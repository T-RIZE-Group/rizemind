from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .coalition_experiment import (
    CoalitionEvaluation,
    ExperimentConfig,
    add_shared_experiment_args,
    evaluate_coalitions,
    get_execution_mode,
    get_one_round_pretraining_duration,
    has_one_round_local_results,
)
from .coalition_strategies import coalition_to_label, coalition_to_mask
from .mask_generators import (
    DEFAULT_DETERMINISTIC_ADDRESS,
    AllMasksGenerator,
    AntitheticPairingMaskGenerator,
    DeterministicContractMaskGenerator,
    EvaluationMask,
    MonteCarloMaskGenerator,
    SampledMaskGenerator,
    StratifiedSeedKeyedMaskGenerator,
    mask_to_coalition,
)
from .mask_generators_extended import (
    ContractParityStratifiedMaskGenerator,
    ContractParityStratifiedWithDuplicatesMaskGenerator,
    StratifiedAntitheticMaskGenerator,
)
from .shapley_metrics import (
    compute_nrmse,
    compute_shapley_values,
    compute_stratified_shapley_values,
)
from .task import DatasetName, TaskProfile

AVAILABLE_METHODS = (
    "exact",
    "monte_carlo",
    "deterministic",
    "antithetic",
    "antithetic_stratified",
    "stratified",
    "stratified_with_duplicates",
    "ssk",
    "stratified_antithetic",
)


def _float_or_blank(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _int_or_blank(value: int | None) -> str:
    return "" if value is None else str(value)


def _parse_int_list(raw_value: str) -> tuple[int, ...]:
    counts: list[int] = []
    for value in raw_value.split(","):
        cleaned = value.strip()
        if not cleaned:
            continue
        count = int(cleaned)
        if count <= 0:
            raise ValueError("Counts must be positive integers.")
        counts.append(count)
    if not counts:
        raise ValueError("Provide at least one count.")
    return tuple(counts)


def _parse_methods(raw_value: str) -> tuple[str, ...]:
    methods: list[str] = []
    for value in raw_value.split(","):
        cleaned = value.strip()
        if not cleaned:
            continue
        if cleaned not in AVAILABLE_METHODS:
            raise ValueError(
                f"Unknown method '{cleaned}'. Available methods: "
                f"{', '.join(AVAILABLE_METHODS)}"
            )
        methods.append(cleaned)
    if not methods:
        raise ValueError("Provide at least one benchmark method.")
    return tuple(dict.fromkeys(methods))


def _parse_method_sample_budgets(raw_value: str | None) -> dict[str, int]:
    if raw_value is None or not raw_value.strip():
        return {}

    budgets: dict[str, int] = {}
    for value in raw_value.split(","):
        cleaned = value.strip()
        if not cleaned:
            continue
        if "=" not in cleaned:
            raise ValueError(
                "Method sample budgets must use the format method=budget."
            )
        method, budget_raw = cleaned.split("=", 1)
        method = method.strip()
        budget_raw = budget_raw.strip()
        if method not in AVAILABLE_METHODS:
            raise ValueError(
                f"Unknown method '{method}' in --method-sample-budgets. "
                f"Available methods: {', '.join(AVAILABLE_METHODS)}"
            )
        budget = int(budget_raw)
        if budget <= 0:
            raise ValueError(
                "Method sample budgets must be positive integers."
            )
        budgets[method] = budget

    return budgets


@dataclass(frozen=True)
class BenchmarkConfig:
    trainer_counts: tuple[int, ...]
    exact_trainer_counts: tuple[int, ...]
    methods: tuple[str, ...]
    sample_budget: int
    method_sample_budgets: dict[str, int]
    min_sampled_coalition_size: int
    utility_source: str
    precomputed_coalitions_dir: str | None
    deterministic_address: str
    round_id: int
    ssk_max_missing_trainers: int
    num_rounds: int
    local_epochs: int
    batch_size: int
    learning_rate: float
    dataset: DatasetName
    task_profile: TaskProfile
    data_dir: str
    output_dir: str
    max_train_samples_per_trainer: int | None
    max_test_samples: int | None
    seed: int
    device: str
    client_num_cpus: float
    client_num_gpus: float
    train_loader_workers: int
    test_loader_workers: int
    persistent_workers: bool
    evaluation_workers: int
    torch_threads_per_worker: int
    evaluation_device: str | None
    combined_logs_dir: str | None


@dataclass(frozen=True)
class BenchmarkCoalitionRecord:
    mask: int
    coalition_label: str
    coalition_size: int
    accuracy: float
    loss: float
    duration_seconds: float
    sample_role: str
    sample_order: int | None
    execution_mode: str
    aggregation_duration_seconds: float | None
    evaluation_duration_seconds: float | None


@dataclass(frozen=True)
class BenchmarkMethodResult:
    trainer_count: int
    method_key: str
    method_name: str
    sampled_masks: list[int]
    coalition_records: list[BenchmarkCoalitionRecord]
    shapley_values: np.ndarray
    execution_mode: str
    total_duration_seconds: float
    pretraining_duration_seconds: float
    aggregation_duration_seconds: float
    evaluation_duration_seconds: float
    coalitions_per_second: float
    coalition_analysis: dict[str, object]


@dataclass(frozen=True)
class BenchmarkSummaryRow:
    trainer_count: int
    method: str
    rmse: float | None
    nrmse_pct: float | None
    mean_exact_shapley: float | None
    sample_budget: int
    sampled_mask_count: int
    evaluated_coalition_count: int
    ground_truth_available: bool
    execution_mode: str
    total_duration_seconds: float
    pretraining_duration_seconds: float
    aggregation_duration_seconds: float
    evaluation_duration_seconds: float
    coalitions_per_second: float


@dataclass(frozen=True)
class BenchmarkMethodSpec:
    key: str
    output_name: str
    csv_method_name: str
    generator: SampledMaskGenerator
    sample_role: str
    estimator: str = "weighted"


def _filter_sampled_masks_by_min_size(
    sampled_masks: list[int],
    *,
    min_sampled_coalition_size: int,
) -> list[int]:
    if min_sampled_coalition_size <= 0:
        return sampled_masks
    return [
        mask
        for mask in sampled_masks
        if mask.bit_count() >= min_sampled_coalition_size
    ]


def _build_sampled_only_evaluation_masks(
    sampled_masks: list[int],
    *,
    sample_role: str,
) -> list[EvaluationMask]:
    return [
        EvaluationMask(mask=mask, sample_role=sample_role, sample_order=sample_order)
        for sample_order, mask in enumerate(sampled_masks)
    ]


@dataclass(frozen=True)
class PrecomputedCoalitionRow:
    coalition_label: str
    coalition_size: int
    accuracy: float
    loss: float
    duration_seconds: float


def _mask_size_histogram(masks: list[int]) -> dict[str, int]:
    return {
        str(size): count
        for size, count in sorted(Counter(mask.bit_count() for mask in masks).items())
    }


def _record_size_histogram(
    records: list[BenchmarkCoalitionRecord],
    *,
    sample_role: str | None = None,
) -> dict[str, int]:
    counts: Counter[int] = Counter()
    for record in records:
        if sample_role is not None and record.sample_role != sample_role:
            continue
        counts[record.coalition_size] += 1
    return {str(size): count for size, count in sorted(counts.items())}


def _sampled_mask_preview(sampled_masks: list[int], trainer_count: int) -> list[dict[str, object]]:
    preview: list[dict[str, object]] = []
    for order, mask in enumerate(sampled_masks[:12]):
        coalition = mask_to_coalition(mask, trainer_count)
        preview.append(
            {
                "order": order,
                "mask": mask,
                "coalition": coalition_to_label(coalition),
                "coalition_size": len(coalition),
            }
        )
    return preview


def _build_antithetic_pair_diagnostics(
    sampled_masks: list[int],
    trainer_count: int,
) -> dict[str, object]:
    full_mask = (1 << trainer_count) - 1
    consecutive_pair_count = len(sampled_masks) // 2
    consecutive_complement_pair_count = 0
    for pair_start in range(0, consecutive_pair_count * 2, 2):
        if sampled_masks[pair_start + 1] == (full_mask ^ sampled_masks[pair_start]):
            consecutive_complement_pair_count += 1

    unique_masks = set(sampled_masks)
    complement_covered_count = sum(
        1 for mask in unique_masks if (full_mask ^ mask) in unique_masks
    )
    return {
        "consecutive_pair_count": consecutive_pair_count,
        "consecutive_complement_pair_count": consecutive_complement_pair_count,
        "consecutive_complement_pair_fraction": (
            0.0
            if consecutive_pair_count == 0
            else consecutive_complement_pair_count / consecutive_pair_count
        ),
        "unique_masks_with_complement_present": complement_covered_count,
        "all_unique_masks_have_complements": (
            complement_covered_count == len(unique_masks)
        ),
    }


def _build_allocation_diagnostics(
    generator: SampledMaskGenerator,
    trainer_count: int,
) -> dict[str, object] | None:
    diagnostics: dict[str, object] = {}

    if hasattr(generator, "allocation_for"):
        allocation = getattr(generator, "allocation_for")(trainer_count)
        diagnostics["planned_allocation"] = {
            str(size): int(count)
            for size, count in sorted(allocation.items())
            if int(count) > 0
        }

    if hasattr(generator, "lower_strata_allocation_for"):
        lower = getattr(generator, "lower_strata_allocation_for")(trainer_count)
        diagnostics["lower_strata_allocation"] = {
            str(size): int(count)
            for size, count in sorted(lower.items())
            if int(count) > 0
        }

    if hasattr(generator, "emitted_size_allocation_for"):
        emitted = getattr(generator, "emitted_size_allocation_for")(trainer_count)
        diagnostics["emitted_size_allocation"] = {
            str(size): int(count)
            for size, count in sorted(emitted.items())
            if int(count) > 0
        }

    return diagnostics or None


def _build_coalition_analysis(
    *,
    trainer_count: int,
    method_spec: BenchmarkMethodSpec,
    sampled_masks: list[int],
    coalition_records: list[BenchmarkCoalitionRecord],
) -> dict[str, object]:
    analysis: dict[str, object] = {
        "trainer_count": trainer_count,
        "method": method_spec.output_name,
        "estimator": method_spec.estimator,
        "emitted_sample_count": len(sampled_masks),
        "unique_sampled_mask_count": len(set(sampled_masks)),
        "duplicate_sampled_mask_count": len(sampled_masks) - len(set(sampled_masks)),
        "evaluated_coalition_count": len(coalition_records),
        "sampled_size_histogram": _mask_size_histogram(sampled_masks),
        "evaluated_size_histogram": _record_size_histogram(coalition_records),
        "support_size_histogram": _record_size_histogram(
            coalition_records, sample_role="support"
        ),
        "sampled_size_histogram_unique_evaluated_rows": _record_size_histogram(
            coalition_records, sample_role="sampled"
        ),
        "first_sampled_masks": _sampled_mask_preview(sampled_masks, trainer_count),
    }

    if method_spec.key in {
        "antithetic",
        "antithetic_stratified",
        "stratified_antithetic",
    }:
        analysis["antithetic_pair_diagnostics"] = _build_antithetic_pair_diagnostics(
            sampled_masks,
            trainer_count,
        )

    allocation_diagnostics = _build_allocation_diagnostics(
        method_spec.generator, trainer_count
    )
    if allocation_diagnostics is not None:
        analysis["allocation_by_stratum"] = allocation_diagnostics

    return analysis


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark exact and approximate DP-Shapley values with nRMSE."
    )
    parser.add_argument(
        "--trainer-counts",
        default="8,9,10,11,12,13,14,15,16",
        help="Comma-separated trainer counts to benchmark.",
    )
    parser.add_argument(
        "--exact-trainer-counts",
        default="8,9,10,11,12,13,14,15,16",
        help="Comma-separated trainer counts that should run exhaustive ground truth.",
    )
    parser.add_argument(
        "--methods",
        default="exact,monte_carlo,deterministic",
        help=(
            "Comma-separated benchmark methods. Available: "
            "exact, monte_carlo, deterministic, antithetic, "
            "antithetic_stratified, stratified, ssk, stratified_antithetic."
        ),
    )
    parser.add_argument(
        "--sample-budget",
        type=int,
        default=837,
        help="Sample budget for approximate methods. Default: 837",
    )
    parser.add_argument(
        "--method-sample-budgets",
        default=None,
        help=(
            "Optional per-method sample budgets in the format "
            "method=budget,method=budget. Methods not listed fall back to "
            "--sample-budget."
        ),
    )
    parser.add_argument(
        "--min-sampled-coalition-size",
        type=int,
        default=0,
        help=(
            "Exclude sampled coalitions smaller than this size before evaluation. "
            "Default: 0"
        ),
    )
    parser.add_argument(
        "--utility-source",
        choices=["live", "precomputed"],
        default="live",
        help=(
            "How to obtain coalition utilities. 'live' evaluates coalitions by "
            "running FL. 'precomputed' loads them from an existing exact "
            "coalitions.csv table. Default: live"
        ),
    )
    parser.add_argument(
        "--precomputed-coalitions-dir",
        default=None,
        help=(
            "Root directory containing trainers-N/exact/coalitions.csv files "
            "used when --utility-source=precomputed."
        ),
    )
    parser.add_argument(
        "--deterministic-address",
        default=DEFAULT_DETERMINISTIC_ADDRESS,
        help="Synthetic address seed used to mirror Solidity _getMask offline.",
    )
    parser.add_argument(
        "--round-id",
        type=int,
        default=0,
        help="Round ID used in the deterministic mask seed. Default: 0",
    )
    parser.add_argument(
        "--ssk-max-missing-trainers",
        type=int,
        default=5,
        help=(
            "For the ssk method, sample only coalition sizes n-1 down to n-k. "
            "Default: 5"
        ),
    )
    parser.add_argument(
        "--combined-logs-dir",
        default=None,
        help=(
            "Optional canonical output root. When set, writes "
            "trainers-N/<method>/coalitions.csv and shapley_values.csv "
            "alongside the timestamped benchmark run."
        ),
    )
    add_shared_experiment_args(parser)
    return parser


def _build_benchmark_config(args: argparse.Namespace) -> BenchmarkConfig:
    if args.sample_budget <= 0:
        raise ValueError("--sample-budget must be positive.")
    if args.min_sampled_coalition_size < 0:
        raise ValueError("--min-sampled-coalition-size must be non-negative.")
    if args.train_loader_workers < 0 or args.test_loader_workers < 0:
        raise ValueError("DataLoader worker counts must be non-negative.")
    if args.evaluation_workers <= 0:
        raise ValueError("--evaluation-workers must be positive.")
    if args.torch_threads_per_worker <= 0:
        raise ValueError("--torch-threads-per-worker must be positive.")
    if args.ssk_max_missing_trainers <= 0:
        raise ValueError("--ssk-max-missing-trainers must be positive.")
    if args.utility_source == "precomputed" and (
        args.precomputed_coalitions_dir is None and args.combined_logs_dir is None
    ):
        raise ValueError(
            "--utility-source=precomputed requires --precomputed-coalitions-dir "
            "or --combined-logs-dir."
        )

    return BenchmarkConfig(
        trainer_counts=_parse_int_list(args.trainer_counts),
        exact_trainer_counts=_parse_int_list(args.exact_trainer_counts),
        methods=_parse_methods(args.methods),
        sample_budget=args.sample_budget,
        method_sample_budgets=_parse_method_sample_budgets(
            args.method_sample_budgets
        ),
        min_sampled_coalition_size=args.min_sampled_coalition_size,
        utility_source=args.utility_source,
        precomputed_coalitions_dir=args.precomputed_coalitions_dir,
        deterministic_address=args.deterministic_address,
        round_id=args.round_id,
        ssk_max_missing_trainers=args.ssk_max_missing_trainers,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dataset=args.dataset,
        task_profile=args.task_profile,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_train_samples_per_trainer=args.max_train_samples_per_trainer,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
        device=args.device,
        client_num_cpus=args.client_num_cpus,
        client_num_gpus=args.client_num_gpus,
        train_loader_workers=args.train_loader_workers,
        test_loader_workers=args.test_loader_workers,
        persistent_workers=args.persistent_workers,
        evaluation_workers=args.evaluation_workers,
        torch_threads_per_worker=args.torch_threads_per_worker,
        evaluation_device=args.evaluation_device,
        combined_logs_dir=args.combined_logs_dir,
    )


def _sample_budget_for_method(config: BenchmarkConfig, method: str) -> int:
    return config.method_sample_budgets.get(method, config.sample_budget)


def _build_experiment_config(
    benchmark_config: BenchmarkConfig,
    trainer_count: int,
    *,
    strategy: str,
) -> ExperimentConfig:
    return ExperimentConfig(
        total_trainers=trainer_count,
        strategy=strategy,
        num_rounds=benchmark_config.num_rounds,
        local_epochs=benchmark_config.local_epochs,
        batch_size=benchmark_config.batch_size,
        learning_rate=benchmark_config.learning_rate,
        dataset=benchmark_config.dataset,
        task_profile=benchmark_config.task_profile,
        data_dir=benchmark_config.data_dir,
        output_dir=benchmark_config.output_dir,
        max_train_samples_per_trainer=benchmark_config.max_train_samples_per_trainer,
        max_test_samples=benchmark_config.max_test_samples,
        seed=benchmark_config.seed,
        budget=None,
        budget_per_size=None,
        min_size=0,
        max_size=None,
        coalitions=None,
        coalitions_file=None,
        include_empty=True,
        include_grand=True,
        device=benchmark_config.device,
        client_num_cpus=benchmark_config.client_num_cpus,
        client_num_gpus=benchmark_config.client_num_gpus,
        train_loader_workers=benchmark_config.train_loader_workers,
        test_loader_workers=benchmark_config.test_loader_workers,
        persistent_workers=benchmark_config.persistent_workers,
        evaluation_workers=benchmark_config.evaluation_workers,
        torch_threads_per_worker=benchmark_config.torch_threads_per_worker,
        evaluation_device=benchmark_config.evaluation_device,
    )


def _resolve_method_spec(method: str, config: BenchmarkConfig) -> BenchmarkMethodSpec:
    sample_budget = _sample_budget_for_method(config, method)

    if method == "exact":
        return BenchmarkMethodSpec(
            key=method,
            output_name="exact",
            csv_method_name="exact",
            generator=AllMasksGenerator(),
            sample_role="exact",
        )
    if method == "monte_carlo":
        return BenchmarkMethodSpec(
            key=method,
            output_name="monte_carlo",
            csv_method_name="monte_carlo",
            generator=MonteCarloMaskGenerator(sample_budget, config.seed),
            sample_role="sampled",
        )
    if method == "deterministic":
        return BenchmarkMethodSpec(
            key=method,
            output_name="deterministic",
            csv_method_name="deterministic",
            generator=DeterministicContractMaskGenerator(
                sample_budget,
                address_seed=config.deterministic_address,
                round_id=config.round_id,
            ),
            sample_role="sampled",
        )
    if method == "antithetic":
        return BenchmarkMethodSpec(
            key=method,
            output_name="antithetic",
            csv_method_name="antithetic",
            generator=AntitheticPairingMaskGenerator(
                sample_budget,
                address_seed=config.deterministic_address,
                round_id=config.round_id,
            ),
            sample_role="sampled",
        )
    if method == "antithetic_stratified":
        return BenchmarkMethodSpec(
            key=method,
            output_name="antithetic_stratified",
            csv_method_name="antithetic_stratified",
            generator=AntitheticPairingMaskGenerator(
                sample_budget,
                address_seed=config.deterministic_address,
                round_id=config.round_id,
            ),
            sample_role="sampled",
            estimator="stratified",
        )
    if method == "stratified":
        return BenchmarkMethodSpec(
            key=method,
            output_name="stratified",
            csv_method_name="stratified",
            generator=ContractParityStratifiedMaskGenerator(
                sample_budget,
                address_seed=config.deterministic_address,
                round_id=config.round_id,
            ),
            sample_role="sampled",
            estimator="stratified",
        )
    if method == "stratified_with_duplicates":
        return BenchmarkMethodSpec(
            key=method,
            output_name="stratified_with_duplicates",
            csv_method_name="stratified_with_duplicates",
            generator=ContractParityStratifiedWithDuplicatesMaskGenerator(
                sample_budget,
                address_seed=config.deterministic_address,
                round_id=config.round_id,
            ),
            sample_role="sampled",
            estimator="stratified",
        )
    if method == "ssk":
        output_name = f"ssk-last{config.ssk_max_missing_trainers}"
        return BenchmarkMethodSpec(
            key=method,
            output_name=output_name,
            csv_method_name=output_name,
            generator=StratifiedSeedKeyedMaskGenerator(
                sample_budget,
                config.seed,
                max_missing_trainers=config.ssk_max_missing_trainers,
            ),
            sample_role="sampled",
        )
    if method == "stratified_antithetic":
        return BenchmarkMethodSpec(
            key=method,
            output_name="stratified_antithetic",
            csv_method_name="stratified_antithetic",
            generator=StratifiedAntitheticMaskGenerator(
                sample_budget,
                config.seed,
                allocation="proportional",
                address_seed=config.deterministic_address,
                round_id=config.round_id,
            ),
            sample_role="sampled",
            estimator="stratified",
        )
    raise ValueError(f"Unsupported method '{method}'.")


def _write_root_config(output_root: Path, config: BenchmarkConfig) -> None:
    (output_root / "config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )


def _load_precomputed_exact_rows(
    config: BenchmarkConfig,
    trainer_count: int,
) -> dict[int, PrecomputedCoalitionRow] | None:
    if config.utility_source != "precomputed":
        return None

    lookup_root = config.precomputed_coalitions_dir or config.combined_logs_dir
    if lookup_root is None:
        return None

    exact_csv = (
        Path(lookup_root)
        / f"trainers-{trainer_count}"
        / "exact"
        / "coalitions.csv"
    )
    if not exact_csv.exists():
        return None

    rows_by_mask: dict[int, PrecomputedCoalitionRow] = {}
    with exact_csv.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            mask = int(row["coalition_mask"])
            coalition_label = row.get("coalition") or coalition_to_label(
                mask_to_coalition(mask, trainer_count)
            )
            coalition_size_raw = row.get("coalition_size")
            coalition_size = (
                int(coalition_size_raw)
                if coalition_size_raw not in {None, ""}
                else len(mask_to_coalition(mask, trainer_count))
            )
            loss_raw = row.get("loss")
            duration_raw = row.get("duration_seconds")
            rows_by_mask[mask] = PrecomputedCoalitionRow(
                coalition_label=coalition_label,
                coalition_size=coalition_size,
                accuracy=float(row["accuracy"]),
                loss=float(loss_raw) if loss_raw not in {None, ""} else 0.0,
                duration_seconds=(
                    float(duration_raw) if duration_raw not in {None, ""} else 0.0
                ),
            )

    return rows_by_mask


def _write_method_coalitions_csv(
    output_root: Path,
    records: list[BenchmarkCoalitionRecord],
) -> None:
    with (output_root / "coalitions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "order",
                "coalition",
                "coalition_mask",
                "coalition_size",
                "accuracy",
                "loss",
                "duration_seconds",
                "sample_role",
                "sample_order",
                "execution_mode",
                "aggregation_duration_seconds",
                "evaluation_duration_seconds",
            ]
        )
        for order, record in enumerate(records):
            writer.writerow(
                [
                    order,
                    record.coalition_label,
                    record.mask,
                    record.coalition_size,
                    _float_or_blank(record.accuracy),
                    _float_or_blank(record.loss),
                    _float_or_blank(record.duration_seconds),
                    record.sample_role,
                    _int_or_blank(record.sample_order),
                    record.execution_mode,
                    _float_or_blank(record.aggregation_duration_seconds),
                    _float_or_blank(record.evaluation_duration_seconds),
                ]
            )


def _write_canonical_method_coalitions_csv(
    output_root: Path,
    records: list[BenchmarkCoalitionRecord],
) -> None:
    with (output_root / "coalitions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "order",
                "coalition",
                "coalition_mask",
                "coalition_size",
                "accuracy",
                "loss",
                "duration_seconds",
                "sample_role",
                "sample_order",
            ]
        )
        for order, record in enumerate(records):
            writer.writerow(
                [
                    order,
                    record.coalition_label,
                    record.mask,
                    record.coalition_size,
                    _float_or_blank(record.accuracy),
                    _float_or_blank(record.loss),
                    _float_or_blank(record.duration_seconds),
                    record.sample_role,
                    _int_or_blank(record.sample_order),
                ]
            )


def _write_shapley_values_csv(
    output_root: Path,
    *,
    trainer_count: int,
    method: str,
    shapley_values: np.ndarray,
) -> None:
    with (output_root / "shapley_values.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["trainer_index", "method", "trainer_count", "shapley_value"])
        for trainer_index, shapley_value in enumerate(shapley_values):
            writer.writerow(
                [
                    trainer_index,
                    method,
                    trainer_count,
                    _float_or_blank(float(shapley_value)),
                ]
            )


def _write_coalition_analysis_json(
    output_root: Path,
    analysis: dict[str, object],
) -> None:
    (output_root / "coalition_analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_canonical_method_outputs(
    combined_logs_root: Path,
    *,
    trainer_count: int,
    method_name: str,
    coalition_records: list[BenchmarkCoalitionRecord],
    shapley_values: np.ndarray,
    coalition_analysis: dict[str, object],
) -> None:
    method_output_root = combined_logs_root / f"trainers-{trainer_count}" / method_name
    method_output_root.mkdir(parents=True, exist_ok=True)
    _write_canonical_method_coalitions_csv(method_output_root, coalition_records)
    _write_shapley_values_csv(
        method_output_root,
        trainer_count=trainer_count,
        method=method_name,
        shapley_values=shapley_values,
    )
    _write_coalition_analysis_json(method_output_root, coalition_analysis)


def _write_summary_csv(
    output_root: Path,
    rows: list[BenchmarkSummaryRow],
) -> None:
    with (output_root / "nrmse_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "trainer_count",
                "method",
                "rmse",
                "nrmse_pct",
                "mean_exact_shapley",
                "sample_budget",
                "sampled_mask_count",
                "evaluated_coalition_count",
                "ground_truth_available",
                "execution_mode",
                "total_duration_seconds",
                "pretraining_duration_seconds",
                "aggregation_duration_seconds",
                "evaluation_duration_seconds",
                "coalitions_per_second",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.trainer_count,
                    row.method,
                    _float_or_blank(row.rmse),
                    _float_or_blank(row.nrmse_pct),
                    _float_or_blank(row.mean_exact_shapley),
                    row.sample_budget,
                    row.sampled_mask_count,
                    row.evaluated_coalition_count,
                    str(row.ground_truth_available).lower(),
                    row.execution_mode,
                    _float_or_blank(row.total_duration_seconds),
                    _float_or_blank(row.pretraining_duration_seconds),
                    _float_or_blank(row.aggregation_duration_seconds),
                    _float_or_blank(row.evaluation_duration_seconds),
                    _float_or_blank(row.coalitions_per_second),
                ]
            )


def _compute_rmse(exact_values: np.ndarray, estimated_values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(estimated_values - exact_values))))


def _normalize_method_order(
    trainer_count: int,
    config: BenchmarkConfig,
) -> list[str]:
    ordered: list[str] = []
    if (
        trainer_count in config.exact_trainer_counts
        and "exact" in config.methods
        and "exact" not in ordered
    ):
        ordered.append("exact")
    for method in config.methods:
        if method == "exact" and trainer_count not in config.exact_trainer_counts:
            continue
        if method not in ordered:
            ordered.append(method)
    return ordered


def _run_single_method(
    *,
    trainer_count: int,
    method: str,
    benchmark_config: BenchmarkConfig,
    experiment_config: ExperimentConfig,
    cache: dict[tuple[int, ...], CoalitionEvaluation],
) -> BenchmarkMethodResult:
    method_spec = _resolve_method_spec(method, benchmark_config)
    sampled_masks = _filter_sampled_masks_by_min_size(
        method_spec.generator.generate(trainer_count),
        min_sampled_coalition_size=benchmark_config.min_sampled_coalition_size,
    )
    evaluation_masks = _build_sampled_only_evaluation_masks(
        sampled_masks,
        sample_role=method_spec.sample_role,
    )
    coalitions = [
        mask_to_coalition(plan.mask, trainer_count) for plan in evaluation_masks
    ]
    precomputed_rows = _load_precomputed_exact_rows(benchmark_config, trainer_count)

    if precomputed_rows is not None:
        print(
            f"Running method '{method_spec.output_name}' for {trainer_count} trainers "
            f"using precomputed exact coalition utilities from combined_logs."
        )
        result_start = time.perf_counter()
        utility_by_mask: dict[int, float] = {}
        coalition_records: list[BenchmarkCoalitionRecord] = []
        for plan in evaluation_masks:
            row = precomputed_rows.get(plan.mask)
            if row is None:
                raise KeyError(
                    "Precomputed exact coalition table is missing mask "
                    f"{plan.mask} for trainer_count={trainer_count}."
                )
            utility_by_mask[plan.mask] = row.accuracy
            coalition_records.append(
                BenchmarkCoalitionRecord(
                    mask=plan.mask,
                    coalition_label=row.coalition_label,
                    coalition_size=row.coalition_size,
                    accuracy=row.accuracy,
                    loss=row.loss,
                    duration_seconds=row.duration_seconds,
                    sample_role=plan.sample_role,
                    sample_order=plan.sample_order,
                    execution_mode="precomputed-exact-lookup",
                    aggregation_duration_seconds=None,
                    evaluation_duration_seconds=None,
                )
            )

        if method_spec.estimator == "stratified":
            shapley_values = compute_stratified_shapley_values(
                sampled_masks,
                utility_by_mask,
                trainer_count,
                missing_policy="skip",
            )
        else:
            shapley_values = compute_shapley_values(
                sampled_masks,
                utility_by_mask,
                trainer_count,
                missing_policy="skip",
            )
        coalition_analysis = _build_coalition_analysis(
            trainer_count=trainer_count,
            method_spec=method_spec,
            sampled_masks=sampled_masks,
            coalition_records=coalition_records,
        )
        total_duration_seconds = time.perf_counter() - result_start
        effective_duration_seconds = max(total_duration_seconds, 1e-9)
        return BenchmarkMethodResult(
            trainer_count=trainer_count,
            method_key=method_spec.key,
            method_name=method_spec.output_name,
            sampled_masks=sampled_masks,
            coalition_records=coalition_records,
            shapley_values=shapley_values,
            execution_mode="precomputed-exact-lookup",
            total_duration_seconds=total_duration_seconds,
            pretraining_duration_seconds=0.0,
            aggregation_duration_seconds=0.0,
            evaluation_duration_seconds=0.0,
            coalitions_per_second=len(coalition_records) / effective_duration_seconds,
            coalition_analysis=coalition_analysis,
        )

    print(
        f"Running method '{method_spec.output_name}' for {trainer_count} trainers with "
        f"{len(sampled_masks)} sampled masks and {len(evaluation_masks)} evaluated coalitions."
    )
    execution_mode = get_execution_mode(experiment_config)
    cache_was_warm = has_one_round_local_results(experiment_config)
    pretraining_before = get_one_round_pretraining_duration(experiment_config)
    evaluate_start = time.perf_counter()
    evaluations = evaluate_coalitions(
        coalitions,
        experiment_config,
        cache=cache,
        progress_callback=lambda index, total, coalition: print(
            f"[{trainer_count}:{method_spec.output_name} {index}/{total}] Evaluating coalition "
            f"{coalition_to_label(coalition)}"
        ),
    )
    evaluate_duration_seconds = time.perf_counter() - evaluate_start
    pretraining_after = get_one_round_pretraining_duration(experiment_config)
    pretraining_duration_seconds = (
        0.0 if cache_was_warm else max(pretraining_after - pretraining_before, 0.0)
    )

    result_start = time.perf_counter()
    utility_by_mask: dict[int, float] = {}
    coalition_records: list[BenchmarkCoalitionRecord] = []
    for plan, evaluation in zip(evaluation_masks, evaluations):
        coalition_mask = coalition_to_mask(evaluation.coalition)
        if coalition_mask != plan.mask:
            raise ValueError(
                "Coalition mask mismatch while building benchmark outputs: "
                f"expected {plan.mask}, got {coalition_mask}."
            )
        utility_by_mask[plan.mask] = evaluation.accuracy
        coalition_records.append(
            BenchmarkCoalitionRecord(
                mask=plan.mask,
                coalition_label=coalition_to_label(evaluation.coalition),
                coalition_size=len(evaluation.coalition),
                accuracy=evaluation.accuracy,
                loss=evaluation.loss,
                duration_seconds=evaluation.duration_seconds,
                sample_role=plan.sample_role,
                sample_order=plan.sample_order,
                execution_mode=evaluation.execution_mode,
                aggregation_duration_seconds=evaluation.aggregation_duration_seconds,
                evaluation_duration_seconds=evaluation.evaluation_duration_seconds,
            )
        )

    if method_spec.estimator == "stratified":
        shapley_values = compute_stratified_shapley_values(
            sampled_masks,
            utility_by_mask,
            trainer_count,
            missing_policy="skip",
        )
    else:
        shapley_values = compute_shapley_values(
            sampled_masks,
            utility_by_mask,
            trainer_count,
            missing_policy="skip",
        )
    coalition_analysis = _build_coalition_analysis(
        trainer_count=trainer_count,
        method_spec=method_spec,
        sampled_masks=sampled_masks,
        coalition_records=coalition_records,
    )
    total_duration_seconds = evaluate_duration_seconds + (
        time.perf_counter() - result_start
    )
    aggregation_duration_seconds = float(
        sum(record.aggregation_duration_seconds or 0.0 for record in coalition_records)
    )
    centralized_evaluation_duration_seconds = float(
        sum(record.evaluation_duration_seconds or 0.0 for record in coalition_records)
    )
    effective_evaluation_seconds = max(
        evaluate_duration_seconds - pretraining_duration_seconds,
        1e-9,
    )
    coalitions_per_second = len(coalition_records) / effective_evaluation_seconds

    return BenchmarkMethodResult(
        trainer_count=trainer_count,
        method_key=method_spec.key,
        method_name=method_spec.output_name,
        sampled_masks=sampled_masks,
        coalition_records=coalition_records,
        shapley_values=shapley_values,
        execution_mode=execution_mode,
        total_duration_seconds=total_duration_seconds,
        pretraining_duration_seconds=pretraining_duration_seconds,
        aggregation_duration_seconds=aggregation_duration_seconds,
        evaluation_duration_seconds=centralized_evaluation_duration_seconds,
        coalitions_per_second=coalitions_per_second,
        coalition_analysis=coalition_analysis,
    )


def run_benchmark(config: BenchmarkConfig) -> list[BenchmarkSummaryRow]:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_root = Path(config.output_dir) / f"shapley-benchmark-{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    _write_root_config(output_root, config)

    summary_rows: list[BenchmarkSummaryRow] = []

    for trainer_count in config.trainer_counts:
        experiment_cache: dict[tuple[int, ...], CoalitionEvaluation] = {}
        experiment_config = _build_experiment_config(
            config,
            trainer_count,
            strategy="shapley_benchmark",
        )
        trainer_output_root = output_root / f"trainers-{trainer_count}"
        trainer_output_root.mkdir(parents=True, exist_ok=True)

        exact_values: np.ndarray | None = None
        for method in _normalize_method_order(trainer_count, config):
            result = _run_single_method(
                trainer_count=trainer_count,
                method=method,
                benchmark_config=config,
                experiment_config=experiment_config,
                cache=experiment_cache,
            )

            method_output_root = trainer_output_root / result.method_name
            method_output_root.mkdir(parents=True, exist_ok=True)
            _write_method_coalitions_csv(method_output_root, result.coalition_records)
            _write_shapley_values_csv(
                method_output_root,
                trainer_count=trainer_count,
                method=result.method_name,
                shapley_values=result.shapley_values,
            )
            _write_coalition_analysis_json(
                method_output_root,
                result.coalition_analysis,
            )
            if config.combined_logs_dir:
                _write_canonical_method_outputs(
                    Path(config.combined_logs_dir),
                    trainer_count=trainer_count,
                    method_name=result.method_name,
                    coalition_records=result.coalition_records,
                    shapley_values=result.shapley_values,
                    coalition_analysis=result.coalition_analysis,
                )

            if method == "exact":
                exact_values = result.shapley_values

            ground_truth_available = exact_values is not None
            rmse: float | None = None
            nrmse_pct: float | None = None
            mean_exact_shapley: float | None = None

            if ground_truth_available:
                mean_exact_shapley = float(np.mean(exact_values))
                if method == "exact":
                    rmse = 0.0
                    nrmse_pct = 0.0
                else:
                    rmse = _compute_rmse(exact_values, result.shapley_values)
                    nrmse_pct = compute_nrmse(exact_values, result.shapley_values)

            summary_rows.append(
                BenchmarkSummaryRow(
                    trainer_count=trainer_count,
                    method=result.method_name,
                    rmse=rmse,
                    nrmse_pct=nrmse_pct,
                    mean_exact_shapley=mean_exact_shapley,
                    sample_budget=len(result.sampled_masks),
                    sampled_mask_count=len(result.sampled_masks),
                    evaluated_coalition_count=len(result.coalition_records),
                    ground_truth_available=ground_truth_available,
                    execution_mode=result.execution_mode,
                    total_duration_seconds=result.total_duration_seconds,
                    pretraining_duration_seconds=result.pretraining_duration_seconds,
                    aggregation_duration_seconds=result.aggregation_duration_seconds,
                    evaluation_duration_seconds=result.evaluation_duration_seconds,
                    coalitions_per_second=result.coalitions_per_second,
                )
            )

    _write_summary_csv(output_root, summary_rows)
    print(f"Benchmark output: {output_root.as_posix()}")
    print(f"nRMSE summary: {(output_root / 'nrmse_summary.csv').as_posix()}")
    return summary_rows


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = _build_benchmark_config(args)
    run_benchmark(config)


if __name__ == "__main__":
    main()
