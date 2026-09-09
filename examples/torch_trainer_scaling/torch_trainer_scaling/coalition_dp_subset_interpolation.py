from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .coalition_experiment import ExperimentConfig, add_shared_experiment_args
from .coalition_experiment_dp import (
    DPReleaseConfig,
    DPCoalitionEvaluation,
    evaluate_coalitions_with_dp,
    write_dp_coalitions_csv,
)
from .coalition_strategies import Coalition, coalition_to_label
from .mask_generators import mask_to_coalition


@dataclass(frozen=True)
class ExactCoalitionRow:
    coalition: Coalition
    coalition_mask: int
    coalition_size: int
    accuracy: float
    loss: float
    duration_seconds: float


def _float_or_blank(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _load_exact_rows(
    exact_csv: Path,
    *,
    total_trainers: int,
) -> list[ExactCoalitionRow]:
    rows: list[ExactCoalitionRow] = []
    with exact_csv.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            coalition_mask = int(row["coalition_mask"])
            coalition = mask_to_coalition(coalition_mask, total_trainers)
            rows.append(
                ExactCoalitionRow(
                    coalition=coalition,
                    coalition_mask=coalition_mask,
                    coalition_size=int(row["coalition_size"]),
                    accuracy=float(row["accuracy"]),
                    loss=float(row["loss"]),
                    duration_seconds=float(row["duration_seconds"]),
                )
            )
    return rows


def _sample_exact_subset(
    rows: list[ExactCoalitionRow],
    *,
    sample_fraction: float,
    seed: int,
) -> list[ExactCoalitionRow]:
    if sample_fraction <= 0 or sample_fraction > 1:
        raise ValueError("--sample-fraction must be in the interval (0, 1].")

    rng = random.Random(seed)
    by_size: dict[int, list[ExactCoalitionRow]] = {}
    for row in rows:
        by_size.setdefault(row.coalition_size, []).append(row)

    selected: list[ExactCoalitionRow] = []
    for size in sorted(by_size):
        group = list(by_size[size])
        group_size = len(group)
        sample_count = min(group_size, max(1, math.ceil(group_size * sample_fraction)))
        selected.extend(rng.sample(group, sample_count))

    selected.sort(key=lambda row: (row.coalition_size, row.coalition_mask))
    return selected


def _mean_delta_by_size(
    sampled_rows: list[ExactCoalitionRow],
    evaluations: list[DPCoalitionEvaluation],
) -> tuple[dict[int, float], dict[int, float], dict[int, int]]:
    accuracy_deltas: dict[int, list[float]] = {}
    loss_deltas: dict[int, list[float]] = {}
    counts: dict[int, int] = {}

    row_by_coalition = {row.coalition: row for row in sampled_rows}
    for evaluation in evaluations:
        row = row_by_coalition[evaluation.coalition]
        size = row.coalition_size
        accuracy_deltas.setdefault(size, []).append(evaluation.accuracy - row.accuracy)
        loss_deltas.setdefault(size, []).append(evaluation.loss - row.loss)
        counts[size] = counts.get(size, 0) + 1

    mean_accuracy = {
        size: float(np.mean(deltas)) for size, deltas in accuracy_deltas.items()
    }
    mean_loss = {size: float(np.mean(deltas)) for size, deltas in loss_deltas.items()}
    return mean_accuracy, mean_loss, counts


def _interpolate_size_effects(
    observed_by_size: dict[int, float],
    *,
    min_size: int,
    max_size: int,
) -> dict[int, float]:
    if not observed_by_size:
        raise ValueError("Cannot interpolate size effects without any observed rows.")

    observed_sizes = np.array(sorted(observed_by_size), dtype=np.float64)
    observed_values = np.array(
        [observed_by_size[int(size)] for size in observed_sizes], dtype=np.float64
    )
    target_sizes = np.arange(min_size, max_size + 1, dtype=np.float64)
    interpolated = np.interp(target_sizes, observed_sizes, observed_values)
    return {
        int(size): float(delta)
        for size, delta in zip(target_sizes.tolist(), interpolated.tolist())
    }


def _build_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        total_trainers=args.total_trainers,
        strategy="manual",
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dataset=args.dataset,
        task_profile=getattr(args, "task_profile", "repo"),
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_train_samples_per_trainer=args.max_train_samples_per_trainer,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
        budget=None,
        budget_per_size=None,
        min_size=0,
        max_size=None,
        coalitions=None,
        coalitions_file=None,
        include_empty=True,
        include_grand=True,
        device=args.device,
        client_num_cpus=args.client_num_cpus,
        client_num_gpus=args.client_num_gpus,
        train_loader_workers=args.train_loader_workers,
        test_loader_workers=args.test_loader_workers,
        persistent_workers=args.persistent_workers,
        evaluation_workers=args.evaluation_workers,
        torch_threads_per_worker=args.torch_threads_per_worker,
        evaluation_device=args.evaluation_device,
    )


def _write_interpolated_coalitions_csv(
    output_root: Path,
    exact_rows: list[ExactCoalitionRow],
    sampled_masks: set[int],
    *,
    interpolated_accuracy_deltas: dict[int, float],
    interpolated_loss_deltas: dict[int, float],
) -> None:
    with (output_root / "interpolated_coalitions.csv").open(
        "w", newline="", encoding="utf-8"
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
        for order, row in enumerate(exact_rows):
            accuracy_delta = interpolated_accuracy_deltas[row.coalition_size]
            loss_delta = interpolated_loss_deltas[row.coalition_size]
            predicted_accuracy = min(max(row.accuracy + accuracy_delta, 0.0), 1.0)
            predicted_loss = max(row.loss + loss_delta, 0.0)
            writer.writerow(
                [
                    order,
                    coalition_to_label(row.coalition),
                    row.coalition_mask,
                    row.coalition_size,
                    _float_or_blank(row.accuracy),
                    _float_or_blank(row.loss),
                    _float_or_blank(predicted_accuracy),
                    _float_or_blank(predicted_loss),
                    "true" if row.coalition_mask in sampled_masks else "false",
                ]
            )


def _write_sampled_exact_rows_csv(
    output_root: Path,
    sampled_rows: list[ExactCoalitionRow],
    evaluations: list[DPCoalitionEvaluation],
) -> None:
    row_by_coalition = {row.coalition: row for row in sampled_rows}
    with (output_root / "sampled_exact_vs_dp.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "coalition",
                "coalition_mask",
                "coalition_size",
                "clean_accuracy_exact",
                "dp_accuracy_observed",
                "accuracy_delta",
                "clean_loss_exact",
                "dp_loss_observed",
                "loss_delta",
                "dp_noise_std",
            ]
        )
        for evaluation in evaluations:
            row = row_by_coalition[evaluation.coalition]
            writer.writerow(
                [
                    coalition_to_label(row.coalition),
                    row.coalition_mask,
                    row.coalition_size,
                    _float_or_blank(row.accuracy),
                    _float_or_blank(evaluation.accuracy),
                    _float_or_blank(evaluation.accuracy - row.accuracy),
                    _float_or_blank(row.loss),
                    _float_or_blank(evaluation.loss),
                    _float_or_blank(evaluation.loss - row.loss),
                    _float_or_blank(evaluation.dp_noise_std),
                ]
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a subset of exact coalitions, evaluate them with Flower "
            "server-side fixed-clipping DP, then interpolate the DP effect across "
            "coalition sizes."
        )
    )
    parser.add_argument(
        "--exact-csv",
        required=True,
        help="Path to trainers-N/exact/coalitions.csv.",
    )
    parser.add_argument(
        "--total-trainers",
        type=int,
        required=True,
        help="Total trainer count for the exact coalition table.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.1,
        help="Fraction of coalitions to evaluate per size stratum. Default: 0.1",
    )
    parser.add_argument(
        "--clip-norm",
        type=float,
        default=1.0,
        help="Flower clipping norm applied before noising. Default: 1.0",
    )
    parser.add_argument(
        "--noise-multiplier",
        type=float,
        default=1.0,
        help="Flower Gaussian noise multiplier. Default: 1.0",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=None,
        help="Optional RNG seed for Gaussian noise. Defaults to --seed.",
    )
    add_shared_experiment_args(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    exact_csv = Path(args.exact_csv)
    exact_rows = _load_exact_rows(exact_csv, total_trainers=args.total_trainers)
    sampled_rows = _sample_exact_subset(
        exact_rows,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
    )
    experiment_config = _build_experiment_config(args)
    dp_config = DPReleaseConfig(
        clip_norm=args.clip_norm,
        noise_multiplier=args.noise_multiplier,
        noise_seed=args.seed if args.noise_seed is None else args.noise_seed,
    )

    sampled_coalitions = [row.coalition for row in sampled_rows]
    evaluations = evaluate_coalitions_with_dp(
        sampled_coalitions,
        experiment_config,
        dp_config,
    )
    observed_accuracy_deltas, observed_loss_deltas, sampled_counts = _mean_delta_by_size(
        sampled_rows,
        evaluations,
    )

    min_size = min(row.coalition_size for row in exact_rows)
    max_size = max(row.coalition_size for row in exact_rows)
    interpolated_accuracy_deltas = _interpolate_size_effects(
        observed_accuracy_deltas,
        min_size=min_size,
        max_size=max_size,
    )
    interpolated_loss_deltas = _interpolate_size_effects(
        observed_loss_deltas,
        min_size=min_size,
        max_size=max_size,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_root = (
        Path(args.output_dir)
        / f"coalition-dp-subset-interpolation-trainers-{args.total_trainers}-{timestamp}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    write_dp_coalitions_csv(output_root, evaluations)
    _write_sampled_exact_rows_csv(output_root, sampled_rows, evaluations)
    _write_interpolated_coalitions_csv(
        output_root,
        exact_rows,
        {row.coalition_mask for row in sampled_rows},
        interpolated_accuracy_deltas=interpolated_accuracy_deltas,
        interpolated_loss_deltas=interpolated_loss_deltas,
    )

    summary = {
        "exact_csv": str(exact_csv),
        "total_trainers": args.total_trainers,
        "sample_fraction": args.sample_fraction,
        "selected_coalition_count": len(sampled_rows),
        "dp_release": asdict(dp_config),
        "sampled_counts_by_size": {
            str(size): count for size, count in sorted(sampled_counts.items())
        },
        "observed_accuracy_delta_by_size": {
            str(size): value for size, value in sorted(observed_accuracy_deltas.items())
        },
        "observed_loss_delta_by_size": {
            str(size): value for size, value in sorted(observed_loss_deltas.items())
        },
        "interpolated_accuracy_delta_by_size": {
            str(size): value
            for size, value in sorted(interpolated_accuracy_deltas.items())
        },
        "interpolated_loss_delta_by_size": {
            str(size): value for size, value in sorted(interpolated_loss_deltas.items())
        },
    }
    (output_root / "interpolation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"DP sampled subset: {(output_root / 'coalitions.csv').as_posix()}")
    print(f"Observed subset deltas: {(output_root / 'sampled_exact_vs_dp.csv').as_posix()}")
    print(
        f"Interpolated full table: {(output_root / 'interpolated_coalitions.csv').as_posix()}"
    )


if __name__ == "__main__":
    main()
