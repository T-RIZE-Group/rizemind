from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .shapley_metrics import compute_nrmse


@dataclass(frozen=True)
class StrategyNrmseRow:
    trainer_count: int
    method: str
    rmse: float
    nrmse_pct: float
    mean_exact_shapley: float


def _load_shapley_values(path: Path) -> tuple[str, int, np.ndarray]:
    trainer_count: int | None = None
    method_name: str | None = None
    values_by_index: dict[int, float] = {}

    with path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            trainer_index = int(row["trainer_index"])
            method_name = row["method"]
            trainer_count = int(row["trainer_count"])
            values_by_index[trainer_index] = float(row["shapley_value"])

    if trainer_count is None or method_name is None:
        raise ValueError(f"No shapley rows found in {path}.")

    ordered_values = np.array(
        [values_by_index[index] for index in sorted(values_by_index)],
        dtype=np.float64,
    )
    return method_name, trainer_count, ordered_values


def compute_combined_logs_nrmse_rows(combined_logs_dir: Path) -> list[StrategyNrmseRow]:
    rows: list[StrategyNrmseRow] = []

    for trainer_dir in sorted(combined_logs_dir.iterdir()):
        if not trainer_dir.is_dir() or not trainer_dir.name.startswith("trainers-"):
            continue

        exact_path = trainer_dir / "exact" / "shapley_values.csv"
        if not exact_path.exists():
            continue

        _, trainer_count, exact_values = _load_shapley_values(exact_path)
        mean_exact_shapley = float(np.mean(exact_values))

        method_dirs = sorted(
            child for child in trainer_dir.iterdir() if child.is_dir()
        )
        for method_dir in method_dirs:
            shapley_path = method_dir / "shapley_values.csv"
            if not shapley_path.exists():
                continue

            method_name, file_trainer_count, estimated_values = _load_shapley_values(
                shapley_path
            )
            if file_trainer_count != trainer_count:
                raise ValueError(
                    f"Trainer count mismatch in {shapley_path}: "
                    f"expected {trainer_count}, got {file_trainer_count}."
                )
            if estimated_values.shape != exact_values.shape:
                raise ValueError(
                    f"Length mismatch between exact and {method_name} for "
                    f"trainer_count={trainer_count}."
                )

            if method_name == "exact":
                rmse = 0.0
                nrmse_pct = 0.0
            else:
                rmse = float(np.sqrt(np.mean(np.square(estimated_values - exact_values))))
                nrmse_pct = compute_nrmse(exact_values, estimated_values)

            rows.append(
                StrategyNrmseRow(
                    trainer_count=trainer_count,
                    method=method_name,
                    rmse=rmse,
                    nrmse_pct=nrmse_pct,
                    mean_exact_shapley=mean_exact_shapley,
                )
            )

    return rows


def _write_summary_csv(path: Path, rows: list[StrategyNrmseRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "trainer_count",
                "method",
                "rmse",
                "nrmse_pct",
                "mean_exact_shapley",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.trainer_count,
                    row.method,
                    f"{row.rmse:.6f}",
                    "inf" if math.isinf(row.nrmse_pct) else f"{row.nrmse_pct:.6f}",
                    f"{row.mean_exact_shapley:.6f}",
                ]
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute nRMSE for every strategy in combined_logs using exact as the baseline."
    )
    parser.add_argument(
        "--combined-logs-dir",
        default="combined_logs",
        help="Root directory containing trainers-N/<method>/shapley_values.csv files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path. Defaults to <combined_logs>/nrmse_summary.csv.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    combined_logs_dir = Path(args.combined_logs_dir)
    rows = compute_combined_logs_nrmse_rows(combined_logs_dir)

    output_path = (
        Path(args.output)
        if args.output is not None
        else combined_logs_dir / "nrmse_summary.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(output_path, rows)
    print(output_path.as_posix())


if __name__ == "__main__":
    main()
