from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import rankdata

from .ablation_metrics import compute_all_metrics
from .coalition_dp_subset_interpolation import (
    _build_experiment_config,
    _interpolate_size_effects,
    _load_exact_rows,
    _sample_exact_subset,
)
from .coalition_experiment import (
    ExperimentConfig,
    _get_or_build_one_round_local_results,
    add_shared_experiment_args,
)
from .task import build_model, get_trainloader, set_weights, test
from .coalition_experiment_dp import (
    DPReleaseConfig,
    _aggregate_with_adaptive_dp,
    _evaluate_parameters,
)
from .coalition_strategies import coalition_to_label
from .shapley_metrics import compute_shapley_values

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AttackMetricSummary:
    primary_metric: str
    primary_auc: float
    diagnostic_auc: float
    per_trainer_primary_auc: dict[int, float]
    per_trainer_diagnostic_auc: dict[int, float]
    example_count: int


@dataclass(frozen=True)
class CalibrationRow:
    trainer_count: int
    noise_multiplier: float
    sampled_rows: int
    mean_accuracy_delta: float
    mean_abs_accuracy_delta: float
    max_abs_accuracy_delta: float
    mean_noise_std: float
    attack_auc: float
    attack_auc_l2: float
    rmse: float
    nrmse_pct: float
    kendall_tau: float
    spearman_rho: float
    top_k_precision: float
    bias: float


@dataclass(frozen=True)
class AggregatedCalibrationRow:
    noise_multiplier: float
    attack_auc_by_trainer_count: dict[int, float]
    rmse_by_trainer_count: dict[int, float]
    kendall_tau_by_trainer_count: dict[int, float]
    avg_attack_auc: float
    avg_attack_auc_l2: float
    avg_rmse: float
    avg_nrmse_pct: float
    avg_kendall_tau: float
    avg_spearman_rho: float
    avg_top_k_precision: float
    avg_bias: float
    threshold_met: bool
    selected: bool


def _float_or_blank(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _parse_int_list(raw_value: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in raw_value.split(","):
        cleaned = part.strip()
        if not cleaned:
            continue
        values.append(int(cleaned))
    if not values:
        raise ValueError("Provide at least one trainer count.")
    return tuple(values)


def _parse_float_list(raw_value: str) -> tuple[float, ...]:
    values: list[float] = []
    for part in raw_value.split(","):
        cleaned = part.strip()
        if not cleaned:
            continue
        value = float(cleaned)
        if value < 0:
            raise ValueError("Noise multipliers must be non-negative.")
        values.append(value)
    if not values:
        raise ValueError("Provide at least one noise multiplier.")
    return tuple(values)


def _noise_multiplier_label(noise_multiplier: float) -> str:
    return str(noise_multiplier).replace(".", "_")


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




def _roc_auc(labels: list[int], scores: list[float]) -> float:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length.")
    if not labels:
        raise ValueError("labels and scores must be non-empty.")

    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5

    ranks = rankdata(scores, method="average")
    positive_rank_sum = float(
        sum(rank for rank, label in zip(ranks, labels, strict=True) if label == 1)
    )
    u_statistic = positive_rank_sum - (positives * (positives + 1) / 2.0)
    return float(u_statistic / (positives * negatives))


def _build_attack_summary(
    *,
    sampled_rows,
    noisy_params_list: list[list[np.ndarray]],
    trainer_count: int,
    config: ExperimentConfig,
) -> tuple[list[dict[str, float | int | str]], AttackMetricSummary]:
    """Loss-based membership inference attack (prediction-based, not weight similarity).

    For each coalition's released model, evaluates loss on every trainer's local
    training data. Members have lower loss → negative loss is used as the attack score.
    This matches the threat model of flower_mia_experiment.
    """
    device = torch.device("cpu")
    loss_by_trainer: dict[int, list[tuple[int, float]]] = {i: [] for i in range(trainer_count)}
    attack_rows: list[dict[str, float | int | str]] = []

    for sampled_row, noisy_params in zip(sampled_rows, noisy_params_list, strict=True):
        coalition_members = set(sampled_row.coalition)
        coalition_label = coalition_to_label(sampled_row.coalition)

        model = build_model(config.task_profile)
        set_weights(model, noisy_params)

        for trainer_index in range(trainer_count):
            trainloader = get_trainloader(
                partition_id=trainer_index,
                batch_size=config.batch_size,
                dataset_name=config.dataset,
                data_dir=config.data_dir,
                num_trainers=config.total_trainers,
                max_train_samples_per_trainer=config.max_train_samples_per_trainer,
                max_test_samples=config.max_test_samples,
                seed=config.seed,
                task_profile=config.task_profile,
            )
            loss, _ = test(model, trainloader, device)
            label = 1 if trainer_index in coalition_members else 0
            score = -loss  # lower loss → higher score → predicted member
            loss_by_trainer[trainer_index].append((label, score))
            attack_rows.append(
                {
                    "coalition": coalition_label,
                    "coalition_mask": sampled_row.coalition_mask,
                    "coalition_size": sampled_row.coalition_size,
                    "trainer_index": trainer_index,
                    "member": label,
                    "loss": loss,
                    "attack_score": score,
                }
            )

    per_trainer_auc = {
        trainer_index: _roc_auc(
            [label for label, _ in items],
            [score for _, score in items],
        )
        for trainer_index, items in loss_by_trainer.items()
    }
    avg_auc = float(np.mean(list(per_trainer_auc.values())))
    return attack_rows, AttackMetricSummary(
        primary_metric="loss_based",
        primary_auc=avg_auc,
        diagnostic_auc=avg_auc,
        per_trainer_primary_auc=per_trainer_auc,
        per_trainer_diagnostic_auc=per_trainer_auc,
        example_count=len(attack_rows),
    )


def _compute_dp_exact_shapley(
    *,
    exact_rows,
    interpolated_accuracy_deltas: dict[int, float],
    trainer_count: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    utility_by_mask: dict[int, float] = {}
    benchmark_rows: list[dict[str, object]] = []

    for order, row in enumerate(exact_rows):
        dp_accuracy = min(
            max(row.accuracy + interpolated_accuracy_deltas[row.coalition_size], 0.0),
            1.0,
        )
        utility_by_mask[row.coalition_mask] = dp_accuracy
        benchmark_rows.append(
            {
                "order": order,
                "coalition": coalition_to_label(row.coalition),
                "coalition_mask": row.coalition_mask,
                "coalition_size": row.coalition_size,
                "accuracy": dp_accuracy,
                "loss": row.loss,
                "duration_seconds": row.duration_seconds,
            }
        )

    all_masks = [int(row["coalition_mask"]) for row in benchmark_rows]
    shapley_values = compute_shapley_values(
        all_masks,
        utility_by_mask,
        trainer_count,
        missing_policy="raise",
    )
    return shapley_values, benchmark_rows


def _write_attack_rows_csv(output_path: Path, attack_rows: list[dict[str, float | int | str]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "coalition",
                "coalition_mask",
                "coalition_size",
                "trainer_index",
                "member",
                "loss",
                "attack_score",
            ],
        )
        writer.writeheader()
        writer.writerows(attack_rows)


def _write_dp_exact_root(
    dp_root: Path,
    *,
    trainer_count: int,
    benchmark_rows: list[dict[str, object]],
    shapley_values: np.ndarray,
) -> None:
    exact_root = dp_root / f"trainers-{trainer_count}" / "exact"
    exact_root.mkdir(parents=True, exist_ok=True)

    with (exact_root / "coalitions.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "order",
                "coalition",
                "coalition_mask",
                "coalition_size",
                "accuracy",
                "loss",
                "duration_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(benchmark_rows)

    with (exact_root / "shapley_values.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["trainer_index", "method", "trainer_count", "shapley_value"])
        for trainer_index, value in enumerate(shapley_values):
            writer.writerow(
                [
                    trainer_index,
                    "exact",
                    trainer_count,
                    _float_or_blank(float(value)),
                ]
            )


def _aggregate_rows_by_noise(
    calibration_rows: list[CalibrationRow],
    *,
    trainer_counts: tuple[int, ...],
    auc_threshold: float,
) -> list[AggregatedCalibrationRow]:
    grouped: dict[float, list[CalibrationRow]] = {}
    for row in calibration_rows:
        grouped.setdefault(row.noise_multiplier, []).append(row)

    aggregated: list[AggregatedCalibrationRow] = []
    for noise_multiplier in sorted(grouped):
        rows = grouped[noise_multiplier]
        rows_by_trainer_count = {row.trainer_count: row for row in rows}
        missing_trainer_counts = [
            trainer_count
            for trainer_count in trainer_counts
            if trainer_count not in rows_by_trainer_count
        ]
        if missing_trainer_counts:
            raise ValueError(
                "Missing calibration rows for noise_multiplier="
                f"{noise_multiplier} and trainer counts {missing_trainer_counts}."
            )
        aggregated.append(
            AggregatedCalibrationRow(
                noise_multiplier=noise_multiplier,
                attack_auc_by_trainer_count={
                    trainer_count: rows_by_trainer_count[trainer_count].attack_auc
                    for trainer_count in trainer_counts
                },
                rmse_by_trainer_count={
                    trainer_count: rows_by_trainer_count[trainer_count].rmse
                    for trainer_count in trainer_counts
                },
                kendall_tau_by_trainer_count={
                    trainer_count: rows_by_trainer_count[trainer_count].kendall_tau
                    for trainer_count in trainer_counts
                },
                avg_attack_auc=float(np.mean([row.attack_auc for row in rows])),
                avg_attack_auc_l2=float(np.mean([row.attack_auc_l2 for row in rows])),
                avg_rmse=float(np.mean([row.rmse for row in rows])),
                avg_nrmse_pct=float(np.mean([row.nrmse_pct for row in rows])),
                avg_kendall_tau=float(np.mean([row.kendall_tau for row in rows])),
                avg_spearman_rho=float(np.mean([row.spearman_rho for row in rows])),
                avg_top_k_precision=float(
                    np.mean([row.top_k_precision for row in rows])
                ),
                avg_bias=float(np.mean([row.bias for row in rows])),
                threshold_met=float(np.mean([row.attack_auc for row in rows]))
                <= auc_threshold,
                selected=False,
            )
        )
    return aggregated


def _select_noise_multiplier(
    aggregated_rows: list[AggregatedCalibrationRow],
) -> AggregatedCalibrationRow:
    threshold_met_rows = [row for row in aggregated_rows if row.threshold_met]
    if threshold_met_rows:
        return min(
            threshold_met_rows,
            key=lambda row: (
                -row.avg_kendall_tau,
                row.avg_rmse,
                row.noise_multiplier,
            ),
        )
    return min(
        aggregated_rows,
        key=lambda row: (
            row.avg_attack_auc,
            -row.avg_kendall_tau,
            row.avg_rmse,
            row.noise_multiplier,
        ),
    )


def _plot_metric_vs_noise(
    rows: list[CalibrationRow],
    *,
    trainer_counts: tuple[int, ...],
    metric_key: str,
    y_label: str,
    title: str,
    output_path: Path,
    selected_noise_multiplier: float,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6))
    for trainer_count in trainer_counts:
        count_rows = sorted(
            [row for row in rows if row.trainer_count == trainer_count],
            key=lambda row: row.noise_multiplier,
        )
        ax.plot(
            [row.noise_multiplier for row in count_rows],
            [getattr(row, metric_key) for row in count_rows],
            marker="o",
            linewidth=2,
            label=f"N={trainer_count}",
        )
    ax.axvline(
        selected_noise_multiplier,
        color="#111111",
        linestyle="--",
        linewidth=1.5,
        label="selected",
    )
    ax.set_xlabel("Noise Multiplier")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate Flower DP on trainer-count subsets, measure coalition-membership "
            "attack AUC, and materialize selected DP exact coalition tables."
        )
    )
    parser.add_argument(
        "--exact-root",
        type=Path,
        default=EXAMPLE_ROOT / "combined_logs" / "AVG",
        help="Root containing trainers-N/exact/coalitions.csv and shapley_values.csv.",
    )
    parser.add_argument(
        "--trainer-counts",
        default="10,13,15",
        help="Comma-separated trainer counts to calibrate. Default: 10,13,15",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.1,
        help="Fraction of coalitions to sample per coalition size. Default: 0.1",
    )
    parser.add_argument(
        "--initial-clipping-norm",
        type=float,
        default=0.1,
        help="Initial clipping norm for adaptive clipping (Andrew et al.). Default: 0.1",
    )
    parser.add_argument(
        "--noise-multipliers",
        default="0.25,0.5,0.75,1.0,1.5,2.0",
        help="Comma-separated Flower noise multipliers.",
    )
    parser.add_argument(
        "--auc-threshold",
        type=float,
        default=0.5,
        help="Average AUC threshold for privacy selection. Default: 0.5",
    )
    parser.add_argument(
        "--calibration-output-dir",
        type=Path,
        default=EXAMPLE_ROOT / "combined_logs" / "AVG" / "results" / "dp_calibration",
        help="Directory for calibration outputs.",
    )
    add_shared_experiment_args(parser)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    trainer_counts = _parse_int_list(args.trainer_counts)
    noise_multipliers = _parse_float_list(args.noise_multipliers)
    if args.num_rounds != 1:
        raise ValueError("DP calibration currently supports only --num-rounds=1.")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_root = args.calibration_output_dir / f"dp-calibration-{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    calibration_rows: list[CalibrationRow] = []
    dp_table_roots: dict[tuple[int, float], Path] = {}

    for trainer_count in trainer_counts:
        print(f"Calibrating Flower DP for trainer_count={trainer_count}")
        exact_root = args.exact_root / f"trainers-{trainer_count}" / "exact"
        exact_rows = _load_exact_rows(
            exact_root / "coalitions.csv",
            total_trainers=trainer_count,
        )
        clean_exact_shapley = _load_shapley_values(exact_root / "shapley_values.csv")
        sampled_rows = _sample_exact_subset(
            exact_rows,
            sample_fraction=args.sample_fraction,
            seed=args.seed,
        )

        trainer_specific_config = _build_experiment_config(
            argparse.Namespace(**(vars(args) | {"total_trainers": trainer_count}))
        )
        # Warm the local-results cache so _aggregate_with_adaptive_dp hits it directly.
        _get_or_build_one_round_local_results(trainer_specific_config)

        for noise_multiplier in noise_multipliers:
            noise_label = _noise_multiplier_label(noise_multiplier)
            result_root = (
                output_root
                / f"trainers-{trainer_count}"
                / f"noise_multiplier-{noise_label}"
            )
            result_root.mkdir(parents=True, exist_ok=True)

            dp_config = DPReleaseConfig(
                noise_multiplier=noise_multiplier,
                noise_seed=args.seed,
                initial_clipping_norm=args.initial_clipping_norm,
            )
            sampled_records: list[dict[str, object]] = []
            noisy_params_list: list[list[np.ndarray]] = []

            for index, sampled_row in enumerate(sampled_rows, start=1):
                print(
                    f"[N={trainer_count} nm={noise_multiplier} {index}/{len(sampled_rows)}] "
                    f"{coalition_to_label(sampled_row.coalition)}"
                )
                _, noisy_parameters, dp_noise_std, _ = _aggregate_with_adaptive_dp(
                    sampled_row.coalition,
                    trainer_specific_config,
                    dp_config,
                )
                loss, accuracy = _evaluate_parameters(noisy_parameters, trainer_specific_config)
                noisy_params_list.append(noisy_parameters)
                sampled_records.append(
                    {
                        "coalition": sampled_row.coalition,
                        "coalition_mask": sampled_row.coalition_mask,
                        "coalition_size": sampled_row.coalition_size,
                        "clean_accuracy": sampled_row.accuracy,
                        "clean_loss": sampled_row.loss,
                        "accuracy": accuracy,
                        "loss": loss,
                        "accuracy_delta": accuracy - sampled_row.accuracy,
                        "loss_delta": loss - sampled_row.loss,
                        "dp_noise_std": dp_noise_std,
                    }
                )

            with (result_root / "sampled_exact_vs_dp.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=[
                        "coalition",
                        "coalition_mask",
                        "coalition_size",
                        "clean_accuracy",
                        "clean_loss",
                        "accuracy",
                        "loss",
                        "accuracy_delta",
                        "loss_delta",
                        "dp_noise_std",
                    ],
                )
                writer.writeheader()
                for record in sampled_records:
                    writer.writerow(
                        {
                            "coalition": coalition_to_label(record["coalition"]),
                            "coalition_mask": record["coalition_mask"],
                            "coalition_size": record["coalition_size"],
                            "clean_accuracy": _float_or_blank(float(record["clean_accuracy"])),
                            "clean_loss": _float_or_blank(float(record["clean_loss"])),
                            "accuracy": _float_or_blank(float(record["accuracy"])),
                            "loss": _float_or_blank(float(record["loss"])),
                            "accuracy_delta": _float_or_blank(float(record["accuracy_delta"])),
                            "loss_delta": _float_or_blank(float(record["loss_delta"])),
                            "dp_noise_std": _float_or_blank(float(record["dp_noise_std"])),
                        }
                    )

            attack_rows, attack_summary = _build_attack_summary(
                sampled_rows=sampled_rows,
                noisy_params_list=noisy_params_list,
                trainer_count=trainer_count,
                config=trainer_specific_config,
            )
            _write_attack_rows_csv(result_root / "attack_scores.csv", attack_rows)
            (result_root / "attack_summary.json").write_text(
                json.dumps(asdict(attack_summary), indent=2) + "\n",
                encoding="utf-8",
            )

            observed_accuracy_deltas: dict[int, float] = {}
            observed_loss_deltas: dict[int, float] = {}
            for coalition_size in sorted(
                {int(record["coalition_size"]) for record in sampled_records}
            ):
                size_records = [
                    record
                    for record in sampled_records
                    if int(record["coalition_size"]) == coalition_size
                ]
                observed_accuracy_deltas[coalition_size] = float(
                    np.mean([float(record["accuracy_delta"]) for record in size_records])
                )
                observed_loss_deltas[coalition_size] = float(
                    np.mean([float(record["loss_delta"]) for record in size_records])
                )

            interpolated_accuracy_deltas = _interpolate_size_effects(
                observed_accuracy_deltas,
                min_size=min(row.coalition_size for row in exact_rows),
                max_size=max(row.coalition_size for row in exact_rows),
            )
            interpolated_loss_deltas = _interpolate_size_effects(
                observed_loss_deltas,
                min_size=min(row.coalition_size for row in exact_rows),
                max_size=max(row.coalition_size for row in exact_rows),
            )
            dp_exact_shapley, benchmark_rows = _compute_dp_exact_shapley(
                exact_rows=exact_rows,
                interpolated_accuracy_deltas=interpolated_accuracy_deltas,
                trainer_count=trainer_count,
            )
            with (result_root / "interpolated_coalitions.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [
                        "order",
                        "coalition",
                        "coalition_mask",
                        "coalition_size",
                        "clean_accuracy",
                        "clean_loss",
                        "dp_accuracy_interpolated",
                        "dp_loss_interpolated",
                        "observed_in_sample",
                    ]
                )
                sampled_masks = {row.coalition_mask for row in sampled_rows}
                for benchmark_row, exact_row in zip(benchmark_rows, exact_rows, strict=True):
                    writer.writerow(
                        [
                            benchmark_row["order"],
                            benchmark_row["coalition"],
                            benchmark_row["coalition_mask"],
                            benchmark_row["coalition_size"],
                            _float_or_blank(exact_row.accuracy),
                            _float_or_blank(exact_row.loss),
                            _float_or_blank(float(benchmark_row["accuracy"])),
                            _float_or_blank(
                                max(
                                    exact_row.loss
                                    + interpolated_loss_deltas[exact_row.coalition_size],
                                    0.0,
                                )
                            ),
                            "true" if exact_row.coalition_mask in sampled_masks else "false",
                        ]
                    )

            with (result_root / "dp_exact_shapley_values.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["trainer_index", "method", "trainer_count", "shapley_value"])
                for trainer_index, value in enumerate(dp_exact_shapley):
                    writer.writerow(
                        [
                            trainer_index,
                            "exact",
                            trainer_count,
                            _float_or_blank(float(value)),
                        ]
                    )

            metrics = compute_all_metrics(clean_exact_shapley, dp_exact_shapley)
            calibration_rows.append(
                CalibrationRow(
                    trainer_count=trainer_count,
                    noise_multiplier=noise_multiplier,
                    sampled_rows=len(sampled_rows),
                    mean_accuracy_delta=float(
                        np.mean([float(record["accuracy_delta"]) for record in sampled_records])
                    ),
                    mean_abs_accuracy_delta=float(
                        np.mean([abs(float(record["accuracy_delta"])) for record in sampled_records])
                    ),
                    max_abs_accuracy_delta=float(
                        np.max([abs(float(record["accuracy_delta"])) for record in sampled_records])
                    ),
                    mean_noise_std=float(
                        np.mean([float(record["dp_noise_std"]) for record in sampled_records])
                    ),
                    attack_auc=attack_summary.primary_auc,
                    attack_auc_l2=attack_summary.diagnostic_auc,
                    rmse=float(metrics["rmse"]),
                    nrmse_pct=float(metrics["nrmse_pct"]),
                    kendall_tau=float(metrics["kendall_tau"]),
                    spearman_rho=float(metrics["spearman_rho"]),
                    top_k_precision=float(metrics["top_k_precision"]),
                    bias=float(metrics["bias"]),
                )
            )

            dp_table_root = output_root / "dp_exact_tables" / f"noise_multiplier-{noise_label}"
            _write_dp_exact_root(
                dp_table_root,
                trainer_count=trainer_count,
                benchmark_rows=benchmark_rows,
                shapley_values=dp_exact_shapley,
            )
            dp_table_roots[(trainer_count, noise_multiplier)] = dp_table_root

            (result_root / "calibration_summary.json").write_text(
                json.dumps(
                    {
                        "trainer_count": trainer_count,
                        "noise_multiplier": noise_multiplier,
                        "sample_fraction": args.sample_fraction,
                        "initial_clipping_norm": args.initial_clipping_norm,
                        "attack_summary": asdict(attack_summary),
                        "metrics_vs_clean_exact": metrics,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    aggregated_rows = _aggregate_rows_by_noise(
        calibration_rows,
        trainer_counts=trainer_counts,
        auc_threshold=args.auc_threshold,
    )
    selected = _select_noise_multiplier(aggregated_rows)
    aggregated_rows = [
        AggregatedCalibrationRow(**(asdict(row) | {"selected": row.noise_multiplier == selected.noise_multiplier}))
        for row in aggregated_rows
    ]

    selection_summary_path = output_root / "selection_summary.csv"
    with selection_summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        header = ["noise_multiplier"]
        for trainer_count in trainer_counts:
            header.append(f"AUC@{trainer_count}")
        header.append("avg_AUC")
        for trainer_count in trainer_counts:
            header.append(f"Kendall@{trainer_count}")
        header.append("avg_kendall_tau")
        for trainer_count in trainer_counts:
            header.append(f"RMSE@{trainer_count}")
        header.extend(
            [
                "avg_rmse",
                "avg_attack_auc_l2",
                "avg_nrmse_pct",
                "avg_spearman_rho",
                "avg_top_k_precision",
                "avg_bias",
                "threshold_met",
                "selected",
            ]
        )
        writer.writerow(header)
        for row in aggregated_rows:
            row_values: list[object] = [row.noise_multiplier]
            row_values.extend(
                _float_or_blank(row.attack_auc_by_trainer_count[trainer_count])
                for trainer_count in trainer_counts
            )
            row_values.append(_float_or_blank(row.avg_attack_auc))
            row_values.extend(
                _float_or_blank(row.kendall_tau_by_trainer_count[trainer_count])
                for trainer_count in trainer_counts
            )
            row_values.append(_float_or_blank(row.avg_kendall_tau))
            row_values.extend(
                _float_or_blank(row.rmse_by_trainer_count[trainer_count])
                for trainer_count in trainer_counts
            )
            row_values.extend(
                [
                    _float_or_blank(row.avg_rmse),
                    _float_or_blank(row.avg_attack_auc_l2),
                    _float_or_blank(row.avg_nrmse_pct),
                    _float_or_blank(row.avg_spearman_rho),
                    _float_or_blank(row.avg_top_k_precision),
                    _float_or_blank(row.avg_bias),
                    str(row.threshold_met).lower(),
                    str(row.selected).lower(),
                ]
            )
            writer.writerow(row_values)

    (output_root / "selection_summary.json").write_text(
        json.dumps([asdict(row) for row in aggregated_rows], indent=2) + "\n",
        encoding="utf-8",
    )

    selected_dp_root = output_root / "selected_dp_exact_root"
    source_selected_root = output_root / "dp_exact_tables" / (
        f"noise_multiplier-{_noise_multiplier_label(selected.noise_multiplier)}"
    )
    if selected_dp_root.exists():
        shutil.rmtree(selected_dp_root)
    shutil.copytree(source_selected_root, selected_dp_root)

    (output_root / "selected_privacy_setting.json").write_text(
        json.dumps(
            {
                "noise_multiplier": selected.noise_multiplier,
                "auc_threshold": args.auc_threshold,
                "threshold_met": selected.threshold_met,
                "avg_attack_auc": selected.avg_attack_auc,
                "avg_kendall_tau": selected.avg_kendall_tau,
                "avg_rmse": selected.avg_rmse,
                "attack_auc_by_trainer_count": selected.attack_auc_by_trainer_count,
                "kendall_tau_by_trainer_count": selected.kendall_tau_by_trainer_count,
                "rmse_by_trainer_count": selected.rmse_by_trainer_count,
                "selected_dp_exact_root": str(selected_dp_root),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    _plot_metric_vs_noise(
        calibration_rows,
        trainer_counts=trainer_counts,
        metric_key="attack_auc",
        y_label="Attack AUC",
        title="Coalition-Membership Attack AUC vs Noise Multiplier",
        output_path=output_root / "attack_auc_vs_noise.png",
        selected_noise_multiplier=selected.noise_multiplier,
    )
    _plot_metric_vs_noise(
        calibration_rows,
        trainer_counts=trainer_counts,
        metric_key="kendall_tau",
        y_label="Kendall Tau vs Clean Exact",
        title="Contribution Ranking Stability vs Noise Multiplier",
        output_path=output_root / "kendall_vs_noise.png",
        selected_noise_multiplier=selected.noise_multiplier,
    )
    _plot_metric_vs_noise(
        calibration_rows,
        trainer_counts=trainer_counts,
        metric_key="rmse",
        y_label="RMSE vs Clean Exact",
        title="Contribution RMSE vs Noise Multiplier",
        output_path=output_root / "rmse_vs_noise.png",
        selected_noise_multiplier=selected.noise_multiplier,
    )

    print(f"DP calibration output: {output_root.as_posix()}")
    print(f"Selected noise multiplier: {selected.noise_multiplier}")


if __name__ == "__main__":
    main()
