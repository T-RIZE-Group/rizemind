# ruff: noqa: E402, I001

import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "torch_trainer_scaling"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from torch_trainer_scaling.dp_sampler_study import (  # noqa: E402
    _configured_budget_for_method,
    main as dp_sampler_study_main,
)
from torch_trainer_scaling.mask_generators import mask_to_coalition  # noqa: E402
from torch_trainer_scaling.coalition_strategies import coalition_to_label  # noqa: E402


def _write_synthetic_exact_root(root: Path, *, trainer_count: int) -> None:
    exact_root = root / f"trainers-{trainer_count}" / "exact"
    exact_root.mkdir(parents=True, exist_ok=True)

    weights = [(index + 1) / 100.0 for index in range(trainer_count)]
    total_weight = sum(weights)

    with (exact_root / "coalitions.csv").open("w", newline="", encoding="utf-8") as csv_file:
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
            ]
        )
        for mask in range(1 << trainer_count):
            coalition = mask_to_coalition(mask, trainer_count)
            accuracy = sum(weights[index] for index in coalition) / total_weight
            writer.writerow(
                [
                    mask,
                    coalition_to_label(coalition),
                    mask,
                    len(coalition),
                    f"{accuracy:.6f}",
                    f"{max(1.0 - accuracy, 0.0):.6f}",
                    "0.0",
                ]
            )

    with (exact_root / "shapley_values.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["trainer_index", "method", "trainer_count", "shapley_value"])
        for trainer_index, weight in enumerate(weights):
            writer.writerow(
                [
                    trainer_index,
                    "exact",
                    trainer_count,
                    f"{(weight / total_weight):.6f}",
                ]
            )


def test_configured_budget_for_method_uses_expected_tables() -> None:
    assert _configured_budget_for_method(
        method="stratified",
        trainer_count=10,
        override_budgets={},
        fallback_budget=3725,
        ssk_max_missing_trainers=3,
    ) == 1024
    assert _configured_budget_for_method(
        method="antithetic_stratified",
        trainer_count=15,
        override_budgets={},
        fallback_budget=3725,
        ssk_max_missing_trainers=3,
    ) == 2504
    assert _configured_budget_for_method(
        method="stratified_antithetic",
        trainer_count=13,
        override_budgets={},
        fallback_budget=3725,
        ssk_max_missing_trainers=3,
    ) == 742


def test_dp_sampler_study_smoke_on_synthetic_precomputed_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dp_exact_root = tmp_path / "dp_exact_root"
    _write_synthetic_exact_root(dp_exact_root, trainer_count=8)

    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dp-sampler-study",
            "--dp-exact-root",
            str(dp_exact_root),
            "--trainer-counts",
            "8",
            "--methods",
            "exact,deterministic,monte_carlo",
            "--method-sample-budgets",
            "deterministic=16,monte_carlo=16",
            "--study-output-dir",
            str(output_dir),
            "--dataset",
            "fake",
            "--data-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "benchmark_logs"),
            "--device",
            "cpu",
        ],
    )

    dp_sampler_study_main()

    study_roots = list(output_dir.glob("dp-sampler-study-*"))
    assert len(study_roots) == 1
    study_root = study_roots[0]

    assert (study_root / "sampler_metrics_per_trainer.csv").exists()
    assert (study_root / "sampler_metrics_average.csv").exists()
    assert (study_root / "sampler_rmse_vs_dp_exact.png").exists()
    assert (study_root / "sampler_kendall_vs_dp_exact.png").exists()
    assert (
        study_root / "combined_logs" / "trainers-8" / "exact" / "shapley_values.csv"
    ).exists()
