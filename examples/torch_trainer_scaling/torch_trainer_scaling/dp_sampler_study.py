from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .ablation_metrics import compute_all_metrics
from .coalition_experiment import add_shared_experiment_args
from .mask_generators import DEFAULT_DETERMINISTIC_ADDRESS
from .sampler_budgets import configured_budget_for_method
from .shapley_benchmark import (
    AVAILABLE_METHODS,
    BenchmarkConfig,
    _parse_int_list,
    _parse_method_sample_budgets,
    _parse_methods,
    run_benchmark,
)

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]

METHOD_OUTPUT_NAMES = {
    "exact": "exact",
    "monte_carlo": "monte_carlo",
    "deterministic": "deterministic",
    "antithetic": "antithetic",
    "antithetic_stratified": "antithetic_stratified",
    "stratified": "stratified",
    "stratified_with_duplicates": "stratified_with_duplicates",
    "ssk": "ssk-last3",
    "stratified_antithetic": "stratified_antithetic",
}

METHOD_DISPLAY_LABELS = {
    "exact": "Exact",
    "monte_carlo": "MC",
    "deterministic": "DT",
    "antithetic": "A-",
    "antithetic_stratified": "A",
    "stratified": "S",
    "stratified_with_duplicates": "SR",
    "ssk": "SSK-L3",
    "stratified_antithetic": "SA",
}

DEFAULT_METHODS = (
    "exact",
    "monte_carlo",
    "deterministic",
    "antithetic_stratified",
    "stratified",
    "stratified_with_duplicates",
)

DEFAULT_SAMPLE_BUDGET = 3725


@dataclass(frozen=True)
class SamplerMetricRow:
    trainer_count: int
    method: str
    method_output_name: str
    method_label: str
    configured_sample_budget: int
    sampled_mask_count: int
    evaluated_coalition_count: int
    rmse: float
    nrmse_pct: float
    kendall_tau: float
    spearman_rho: float
    top_k_precision: float
    bias: float


@dataclass(frozen=True)
class AveragedSamplerMetricRow:
    method: str
    method_output_name: str
    method_label: str
    configured_sample_budget_mean: float
    sampled_mask_count_mean: float
    evaluated_coalition_count_mean: float
    rmse_mean: float
    nrmse_pct_mean: float
    kendall_tau_mean: float
    spearman_rho_mean: float
    top_k_precision_mean: float
    bias_mean: float


def _load_shapley_values(path: Path) -> np.ndarray:
    values_by_index: dict[int, float] = {}
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            values_by_index[int(row["trainer_index"])] = float(row["shapley_value"])
    if not values_by_index:
        raise ValueError(f"No Shapley values found in {path}.")
    return np.asarray(
        [values_by_index[index] for index in sorted(values_by_index)],
        dtype=np.float64,
    )


def _count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        return sum(1 for _ in reader)


def _resolve_dp_exact_root(
    *,
    calibration_root: Path | None,
    dp_exact_root: Path | None,
) -> Path:
    if dp_exact_root is not None:
        return dp_exact_root
    if calibration_root is None:
        raise ValueError("Provide either --calibration-root or --dp-exact-root.")
    selected_privacy_path = calibration_root / "selected_privacy_setting.json"
    if not selected_privacy_path.exists():
        raise FileNotFoundError(
            "Could not find selected_privacy_setting.json under "
            f"{calibration_root}."
        )
    payload = json.loads(selected_privacy_path.read_text(encoding="utf-8"))
    selected_root = payload.get("selected_dp_exact_root")
    if not selected_root:
        raise ValueError(
            "selected_privacy_setting.json is missing selected_dp_exact_root."
        )
    return Path(selected_root)


def _configured_budget_for_method(
    *,
    method: str,
    trainer_count: int,
    override_budgets: dict[str, int],
    fallback_budget: int,
    ssk_max_missing_trainers: int,
) -> int:
    return configured_budget_for_method(
        method=method,
        trainer_count=trainer_count,
        override_budgets=override_budgets,
        fallback_budget=fallback_budget,
        ssk_max_missing_trainers=ssk_max_missing_trainers,
    )


def _build_method_sample_budgets(
    *,
    methods: tuple[str, ...],
    trainer_count: int,
    override_budgets: dict[str, int],
    fallback_budget: int,
    ssk_max_missing_trainers: int,
) -> dict[str, int]:
    budgets: dict[str, int] = {}
    for method in methods:
        if method == "exact":
            continue
        budgets[method] = _configured_budget_for_method(
            method=method,
            trainer_count=trainer_count,
            override_budgets=override_budgets,
            fallback_budget=fallback_budget,
            ssk_max_missing_trainers=ssk_max_missing_trainers,
        )
    return budgets


def _build_benchmark_config_for_trainer_count(
    args: argparse.Namespace,
    *,
    trainer_count: int,
    methods: tuple[str, ...],
    dp_exact_root: Path,
    combined_logs_root: Path,
    benchmark_output_dir: Path,
    override_budgets: dict[str, int],
) -> BenchmarkConfig:
    return BenchmarkConfig(
        trainer_counts=(trainer_count,),
        exact_trainer_counts=(trainer_count,),
        methods=methods,
        sample_budget=args.sample_budget,
        method_sample_budgets=_build_method_sample_budgets(
            methods=methods,
            trainer_count=trainer_count,
            override_budgets=override_budgets,
            fallback_budget=args.sample_budget,
            ssk_max_missing_trainers=args.ssk_max_missing_trainers,
        ),
        min_sampled_coalition_size=args.min_sampled_coalition_size,
        utility_source="precomputed",
        precomputed_coalitions_dir=str(dp_exact_root),
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
        output_dir=str(benchmark_output_dir),
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
        combined_logs_dir=str(combined_logs_root),
    )


def _build_metric_rows(
    *,
    trainer_counts: tuple[int, ...],
    methods: tuple[str, ...],
    combined_logs_root: Path,
    override_budgets: dict[str, int],
    fallback_budget: int,
    ssk_max_missing_trainers: int,
) -> list[SamplerMetricRow]:
    rows: list[SamplerMetricRow] = []
    for trainer_count in trainer_counts:
        exact_values = _load_shapley_values(
            combined_logs_root
            / f"trainers-{trainer_count}"
            / "exact"
            / "shapley_values.csv"
        )
        for method in methods:
            output_name = METHOD_OUTPUT_NAMES[method]
            method_root = combined_logs_root / f"trainers-{trainer_count}" / output_name
            shapley_values = _load_shapley_values(method_root / "shapley_values.csv")
            metrics = compute_all_metrics(exact_values, shapley_values)
            coalitions_csv = method_root / "coalitions.csv"
            evaluated_count = _count_rows(coalitions_csv)
            rows.append(
                SamplerMetricRow(
                    trainer_count=trainer_count,
                    method=method,
                    method_output_name=output_name,
                    method_label=METHOD_DISPLAY_LABELS[method],
                    configured_sample_budget=_configured_budget_for_method(
                        method=method,
                        trainer_count=trainer_count,
                        override_budgets=override_budgets,
                        fallback_budget=fallback_budget,
                        ssk_max_missing_trainers=ssk_max_missing_trainers,
                    ),
                    sampled_mask_count=evaluated_count,
                    evaluated_coalition_count=evaluated_count,
                    rmse=float(metrics["rmse"]),
                    nrmse_pct=float(metrics["nrmse_pct"]),
                    kendall_tau=float(metrics["kendall_tau"]),
                    spearman_rho=float(metrics["spearman_rho"]),
                    top_k_precision=float(metrics["top_k_precision"]),
                    bias=float(metrics["bias"]),
                )
            )
    return rows


def _average_metric_rows(
    rows: list[SamplerMetricRow],
    *,
    methods: tuple[str, ...],
) -> list[AveragedSamplerMetricRow]:
    rows_by_method: dict[str, list[SamplerMetricRow]] = {}
    for row in rows:
        rows_by_method.setdefault(row.method, []).append(row)

    averaged_rows: list[AveragedSamplerMetricRow] = []
    for method in methods:
        method_rows = rows_by_method[method]
        averaged_rows.append(
            AveragedSamplerMetricRow(
                method=method,
                method_output_name=METHOD_OUTPUT_NAMES[method],
                method_label=METHOD_DISPLAY_LABELS[method],
                configured_sample_budget_mean=float(
                    np.mean([row.configured_sample_budget for row in method_rows])
                ),
                sampled_mask_count_mean=float(
                    np.mean([row.sampled_mask_count for row in method_rows])
                ),
                evaluated_coalition_count_mean=float(
                    np.mean([row.evaluated_coalition_count for row in method_rows])
                ),
                rmse_mean=float(np.mean([row.rmse for row in method_rows])),
                nrmse_pct_mean=float(np.mean([row.nrmse_pct for row in method_rows])),
                kendall_tau_mean=float(np.mean([row.kendall_tau for row in method_rows])),
                spearman_rho_mean=float(np.mean([row.spearman_rho for row in method_rows])),
                top_k_precision_mean=float(
                    np.mean([row.top_k_precision for row in method_rows])
                ),
                bias_mean=float(np.mean([row.bias for row in method_rows])),
            )
        )
    return averaged_rows


def _plot_metric_panels(
    *,
    per_trainer_rows: list[SamplerMetricRow],
    averaged_rows: list[AveragedSamplerMetricRow],
    trainer_counts: tuple[int, ...],
    methods: tuple[str, ...],
    metric_key: str,
    averaged_metric_key: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    panel_count = len(trainer_counts) + 1
    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=(4.4 * panel_count, 5.6),
        sharey=True,
    )
    if panel_count == 1:
        axes = [axes]

    method_labels = [METHOD_DISPLAY_LABELS[method] for method in methods]
    x_positions = np.arange(len(methods))

    for axis, trainer_count in zip(axes[:-1], trainer_counts, strict=True):
        count_rows = {
            row.method: row
            for row in per_trainer_rows
            if row.trainer_count == trainer_count
        }
        axis.bar(
            x_positions,
            [getattr(count_rows[method], metric_key) for method in methods],
            color="#8da0cb",
        )
        axis.set_title(f"N={trainer_count}")
        axis.set_xticks(x_positions, method_labels, rotation=45, ha="right")
        axis.grid(axis="y", alpha=0.25)

    averaged_by_method = {row.method: row for row in averaged_rows}
    axes[-1].bar(
        x_positions,
        [getattr(averaged_by_method[method], averaged_metric_key) for method in methods],
        color="#66c2a5",
    )
    axes[-1].set_title("Average")
    axes[-1].set_xticks(x_positions, method_labels, rotation=45, ha="right")
    axes[-1].grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(y_label)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_per_trainer_csv(path: Path, rows: list[SamplerMetricRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "trainer_count",
                "method",
                "method_output_name",
                "method_label",
                "configured_sample_budget",
                "sampled_mask_count",
                "evaluated_coalition_count",
                "rmse",
                "nrmse_pct",
                "kendall_tau",
                "spearman_rho",
                "top_k_precision",
                "bias",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.trainer_count,
                    row.method,
                    row.method_output_name,
                    row.method_label,
                    row.configured_sample_budget,
                    row.sampled_mask_count,
                    row.evaluated_coalition_count,
                    row.rmse,
                    row.nrmse_pct,
                    row.kendall_tau,
                    row.spearman_rho,
                    row.top_k_precision,
                    row.bias,
                ]
            )


def _write_averaged_csv(path: Path, rows: list[AveragedSamplerMetricRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "method",
                "method_output_name",
                "method_label",
                "configured_sample_budget_mean",
                "sampled_mask_count_mean",
                "evaluated_coalition_count_mean",
                "rmse_mean",
                "nrmse_pct_mean",
                "kendall_tau_mean",
                "spearman_rho_mean",
                "top_k_precision_mean",
                "bias_mean",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.method,
                    row.method_output_name,
                    row.method_label,
                    row.configured_sample_budget_mean,
                    row.sampled_mask_count_mean,
                    row.evaluated_coalition_count_mean,
                    row.rmse_mean,
                    row.nrmse_pct_mean,
                    row.kendall_tau_mean,
                    row.spearman_rho_mean,
                    row.top_k_precision_mean,
                    row.bias_mean,
                ]
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the sampler family against selected DP exact coalition tables "
            "and summarize RMSE/Kendall against DP exact."
        )
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=None,
        help="Path to a dp-calibration-* output root containing selected_privacy_setting.json.",
    )
    parser.add_argument(
        "--dp-exact-root",
        type=Path,
        default=None,
        help="Direct path to a root containing trainers-N/exact/coalitions.csv.",
    )
    parser.add_argument(
        "--trainer-counts",
        default="10,13,15",
        help="Comma-separated trainer counts to evaluate. Default: 10,13,15",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help=(
            "Comma-separated sampler methods. Available: "
            + ", ".join(AVAILABLE_METHODS)
        ),
    )
    parser.add_argument(
        "--sample-budget",
        type=int,
        default=DEFAULT_SAMPLE_BUDGET,
        help=f"Fallback sample budget for methods without per-N tables. Default: {DEFAULT_SAMPLE_BUDGET}",
    )
    parser.add_argument(
        "--method-sample-budgets",
        default="",
        help=(
            "Optional global overrides in method=budget format, e.g. "
            "monte_carlo=256,stratified=256."
        ),
    )
    parser.add_argument(
        "--min-sampled-coalition-size",
        type=int,
        default=0,
        help="Minimum allowed sampled coalition size. Default: 0",
    )
    parser.add_argument(
        "--deterministic-address",
        default=DEFAULT_DETERMINISTIC_ADDRESS,
        help="Deterministic address seed used by contract-parity generators.",
    )
    parser.add_argument(
        "--round-id",
        type=int,
        default=0,
        help="Round id used by contract-parity generators. Default: 0",
    )
    parser.add_argument(
        "--ssk-max-missing-trainers",
        type=int,
        default=3,
        help="Tail width for the SSK generator. Default: 3",
    )
    parser.add_argument(
        "--study-output-dir",
        type=Path,
        default=EXAMPLE_ROOT / "combined_logs" / "AVG" / "results" / "dp_sampler_study",
        help="Directory for sampler-study outputs.",
    )
    add_shared_experiment_args(parser)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    trainer_counts = _parse_int_list(args.trainer_counts)
    methods = _parse_methods(args.methods)
    if "exact" not in methods:
        raise ValueError(
            "DP sampler study requires the 'exact' method so metrics can be computed "
            "against DP exact Shapley."
        )
    override_budgets = _parse_method_sample_budgets(args.method_sample_budgets)
    dp_exact_root = _resolve_dp_exact_root(
        calibration_root=args.calibration_root,
        dp_exact_root=args.dp_exact_root,
    )

    for trainer_count in trainer_counts:
        exact_csv = dp_exact_root / f"trainers-{trainer_count}" / "exact" / "coalitions.csv"
        if not exact_csv.exists():
            raise FileNotFoundError(
                f"Missing DP exact coalition table for trainer_count={trainer_count}: {exact_csv}"
            )

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_root = args.study_output_dir / f"dp-sampler-study-{timestamp}"
    combined_logs_root = output_root / "combined_logs"
    combined_logs_root.mkdir(parents=True, exist_ok=True)

    if args.calibration_root is not None:
        selected_privacy_path = args.calibration_root / "selected_privacy_setting.json"
        if selected_privacy_path.exists():
            shutil.copy2(selected_privacy_path, output_root / "selected_privacy_setting.json")

    benchmark_output_root = output_root / "benchmark_runs"
    benchmark_output_root.mkdir(parents=True, exist_ok=True)

    for trainer_count in trainer_counts:
        print(f"Running DP sampler study for trainer_count={trainer_count}")
        trainer_benchmark_output_dir = benchmark_output_root / f"trainers-{trainer_count}"
        trainer_benchmark_output_dir.mkdir(parents=True, exist_ok=True)
        benchmark_config = _build_benchmark_config_for_trainer_count(
            args,
            trainer_count=trainer_count,
            methods=methods,
            dp_exact_root=dp_exact_root,
            combined_logs_root=combined_logs_root,
            benchmark_output_dir=trainer_benchmark_output_dir,
            override_budgets=override_budgets,
        )
        run_benchmark(benchmark_config)

    metric_rows = _build_metric_rows(
        trainer_counts=trainer_counts,
        methods=methods,
        combined_logs_root=combined_logs_root,
        override_budgets=override_budgets,
        fallback_budget=args.sample_budget,
        ssk_max_missing_trainers=args.ssk_max_missing_trainers,
    )
    averaged_rows = _average_metric_rows(metric_rows, methods=methods)

    _write_per_trainer_csv(output_root / "sampler_metrics_per_trainer.csv", metric_rows)
    _write_averaged_csv(output_root / "sampler_metrics_average.csv", averaged_rows)
    (output_root / "sampler_metrics_per_trainer.json").write_text(
        json.dumps([asdict(row) for row in metric_rows], indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "sampler_metrics_average.json").write_text(
        json.dumps([asdict(row) for row in averaged_rows], indent=2) + "\n",
        encoding="utf-8",
    )

    _plot_metric_panels(
        per_trainer_rows=metric_rows,
        averaged_rows=averaged_rows,
        trainer_counts=trainer_counts,
        methods=methods,
        metric_key="rmse",
        averaged_metric_key="rmse_mean",
        y_label="RMSE vs DP Exact",
        title="Sampler RMSE Against DP Exact",
        output_path=output_root / "sampler_rmse_vs_dp_exact.png",
    )
    _plot_metric_panels(
        per_trainer_rows=metric_rows,
        averaged_rows=averaged_rows,
        trainer_counts=trainer_counts,
        methods=methods,
        metric_key="kendall_tau",
        averaged_metric_key="kendall_tau_mean",
        y_label="Kendall Tau vs DP Exact",
        title="Sampler Kendall Tau Against DP Exact",
        output_path=output_root / "sampler_kendall_vs_dp_exact.png",
    )

    (output_root / "study_config.json").write_text(
        json.dumps(
            {
                "trainer_counts": list(trainer_counts),
                "methods": list(methods),
                "dp_exact_root": str(dp_exact_root),
                "sample_budget": args.sample_budget,
                "method_sample_budgets": override_budgets,
                "min_sampled_coalition_size": args.min_sampled_coalition_size,
                "deterministic_address": args.deterministic_address,
                "round_id": args.round_id,
                "ssk_max_missing_trainers": args.ssk_max_missing_trainers,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"DP sampler study output: {output_root.as_posix()}")


if __name__ == "__main__":
    main()
