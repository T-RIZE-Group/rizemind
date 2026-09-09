# ruff: noqa: E402, I001

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "torch_trainer_scaling"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from torch_trainer_scaling.dp_privacy_calibration import (  # noqa: E402
    AggregatedCalibrationRow,
    CalibrationRow,
    _aggregate_rows_by_noise,
    _roc_auc,
    _select_noise_multiplier,
)


def _build_calibration_row(
    *,
    trainer_count: int,
    noise_multiplier: float,
    attack_auc: float,
    rmse: float,
    kendall_tau: float,
) -> CalibrationRow:
    return CalibrationRow(
        trainer_count=trainer_count,
        noise_multiplier=noise_multiplier,
        sampled_rows=10,
        mean_accuracy_delta=-0.1,
        mean_abs_accuracy_delta=0.1,
        max_abs_accuracy_delta=0.2,
        mean_noise_std=0.5,
        attack_auc=attack_auc,
        attack_auc_l2=attack_auc,
        rmse=rmse,
        nrmse_pct=rmse * 10.0,
        kendall_tau=kendall_tau,
        spearman_rho=kendall_tau,
        top_k_precision=0.5,
        bias=0.0,
    )


def test_roc_auc_returns_one_for_perfect_separation() -> None:
    assert _roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_aggregate_rows_by_noise_keeps_per_trainer_count_metrics() -> None:
    aggregated_rows = _aggregate_rows_by_noise(
        [
            _build_calibration_row(
                trainer_count=10,
                noise_multiplier=0.5,
                attack_auc=0.49,
                rmse=0.3,
                kendall_tau=0.8,
            ),
            _build_calibration_row(
                trainer_count=13,
                noise_multiplier=0.5,
                attack_auc=0.48,
                rmse=0.4,
                kendall_tau=0.7,
            ),
            _build_calibration_row(
                trainer_count=15,
                noise_multiplier=0.5,
                attack_auc=0.47,
                rmse=0.5,
                kendall_tau=0.6,
            ),
        ],
        trainer_counts=(10, 13, 15),
        auc_threshold=0.5,
    )

    assert len(aggregated_rows) == 1
    aggregated = aggregated_rows[0]
    assert aggregated.attack_auc_by_trainer_count == pytest.approx(
        {10: 0.49, 13: 0.48, 15: 0.47}
    )
    assert aggregated.kendall_tau_by_trainer_count == pytest.approx(
        {10: 0.8, 13: 0.7, 15: 0.6}
    )
    assert aggregated.rmse_by_trainer_count == pytest.approx(
        {10: 0.3, 13: 0.4, 15: 0.5}
    )
    assert aggregated.avg_attack_auc == pytest.approx((0.49 + 0.48 + 0.47) / 3.0)
    assert aggregated.threshold_met is True


def test_select_noise_multiplier_uses_threshold_then_kendall_then_rmse() -> None:
    selected = _select_noise_multiplier(
        [
            AggregatedCalibrationRow(
                noise_multiplier=0.5,
                attack_auc_by_trainer_count={10: 0.49, 13: 0.49, 15: 0.49},
                rmse_by_trainer_count={10: 0.7, 13: 0.7, 15: 0.7},
                kendall_tau_by_trainer_count={10: 0.6, 13: 0.6, 15: 0.6},
                avg_attack_auc=0.49,
                avg_attack_auc_l2=0.49,
                avg_rmse=0.7,
                avg_nrmse_pct=7.0,
                avg_kendall_tau=0.6,
                avg_spearman_rho=0.6,
                avg_top_k_precision=0.5,
                avg_bias=0.0,
                threshold_met=True,
                selected=False,
            ),
            AggregatedCalibrationRow(
                noise_multiplier=1.0,
                attack_auc_by_trainer_count={10: 0.48, 13: 0.48, 15: 0.48},
                rmse_by_trainer_count={10: 0.9, 13: 0.9, 15: 0.9},
                kendall_tau_by_trainer_count={10: 0.8, 13: 0.8, 15: 0.8},
                avg_attack_auc=0.48,
                avg_attack_auc_l2=0.48,
                avg_rmse=0.9,
                avg_nrmse_pct=9.0,
                avg_kendall_tau=0.8,
                avg_spearman_rho=0.8,
                avg_top_k_precision=0.5,
                avg_bias=0.0,
                threshold_met=True,
                selected=False,
            ),
        ]
    )

    assert selected.noise_multiplier == pytest.approx(1.0)


def test_select_noise_multiplier_falls_back_to_lowest_auc_when_threshold_not_met() -> None:
    selected = _select_noise_multiplier(
        [
            AggregatedCalibrationRow(
                noise_multiplier=0.5,
                attack_auc_by_trainer_count={10: 0.62, 13: 0.62, 15: 0.62},
                rmse_by_trainer_count={10: 0.4, 13: 0.4, 15: 0.4},
                kendall_tau_by_trainer_count={10: 0.9, 13: 0.9, 15: 0.9},
                avg_attack_auc=0.62,
                avg_attack_auc_l2=0.62,
                avg_rmse=0.4,
                avg_nrmse_pct=4.0,
                avg_kendall_tau=0.9,
                avg_spearman_rho=0.9,
                avg_top_k_precision=0.5,
                avg_bias=0.0,
                threshold_met=False,
                selected=False,
            ),
            AggregatedCalibrationRow(
                noise_multiplier=1.0,
                attack_auc_by_trainer_count={10: 0.55, 13: 0.55, 15: 0.55},
                rmse_by_trainer_count={10: 0.8, 13: 0.8, 15: 0.8},
                kendall_tau_by_trainer_count={10: 0.2, 13: 0.2, 15: 0.2},
                avg_attack_auc=0.55,
                avg_attack_auc_l2=0.55,
                avg_rmse=0.8,
                avg_nrmse_pct=8.0,
                avg_kendall_tau=0.2,
                avg_spearman_rho=0.2,
                avg_top_k_precision=0.5,
                avg_bias=0.0,
                threshold_met=False,
                selected=False,
            ),
        ]
    )

    assert selected.noise_multiplier == pytest.approx(1.0)
