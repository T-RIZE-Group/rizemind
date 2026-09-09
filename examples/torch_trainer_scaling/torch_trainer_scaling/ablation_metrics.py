"""Extended metrics for the DP-Shapley ablation study.

Complements the existing compute_nrmse in shapley_metrics.py with
Kendall tau, top-k precision, max absolute error, and bias.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr

from .shapley_metrics import compute_nrmse


def compute_rmse(
    exact_values: Sequence[float],
    estimated_values: Sequence[float],
) -> float:
    exact = np.asarray(exact_values, dtype=np.float64)
    estimated = np.asarray(estimated_values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(estimated - exact))))


def compute_max_absolute_error(
    exact_values: Sequence[float],
    estimated_values: Sequence[float],
) -> float:
    """max_i |phi_hat_i - phi_i| / mean(|phi|)."""
    exact = np.asarray(exact_values, dtype=np.float64)
    estimated = np.asarray(estimated_values, dtype=np.float64)
    exact_mean = float(np.mean(np.abs(exact)))
    max_err = float(np.max(np.abs(estimated - exact)))
    if np.isclose(exact_mean, 0.0):
        return 0.0 if np.isclose(max_err, 0.0) else float("inf")
    return max_err / exact_mean


def compute_kendall_tau(
    exact_values: Sequence[float],
    estimated_values: Sequence[float],
) -> float:
    """Kendall rank correlation coefficient."""
    exact = np.asarray(exact_values, dtype=np.float64)
    estimated = np.asarray(estimated_values, dtype=np.float64)
    if len(exact) < 2:
        return 1.0
    tau, _ = kendalltau(exact, estimated)
    return float(tau) if not np.isnan(tau) else 0.0


def compute_spearman_rho(
    exact_values: Sequence[float],
    estimated_values: Sequence[float],
) -> float:
    """Spearman rank correlation coefficient."""
    exact = np.asarray(exact_values, dtype=np.float64)
    estimated = np.asarray(estimated_values, dtype=np.float64)
    if len(exact) < 2:
        return 1.0
    rho, _ = spearmanr(exact, estimated)
    return float(rho) if not np.isnan(rho) else 0.0


def compute_top_k_precision(
    exact_values: Sequence[float],
    estimated_values: Sequence[float],
    k: int,
) -> float:
    """|topk(exact) ∩ topk(estimated)| / k."""
    exact = np.asarray(exact_values, dtype=np.float64)
    estimated = np.asarray(estimated_values, dtype=np.float64)
    k = min(k, len(exact))
    if k <= 0:
        return 1.0
    exact_topk = set(np.argsort(exact)[-k:])
    estimated_topk = set(np.argsort(estimated)[-k:])
    return len(exact_topk & estimated_topk) / k


def compute_bias(
    exact_values: Sequence[float],
    estimated_values: Sequence[float],
) -> float:
    """mean(estimated - exact) / mean(exact). Systematic over/under-estimation."""
    exact = np.asarray(exact_values, dtype=np.float64)
    estimated = np.asarray(estimated_values, dtype=np.float64)
    exact_mean = float(np.mean(exact))
    bias = float(np.mean(estimated - exact))
    if np.isclose(exact_mean, 0.0):
        return 0.0 if np.isclose(bias, 0.0) else float("inf")
    return bias / exact_mean


def compute_all_metrics(
    exact_values: Sequence[float],
    estimated_values: Sequence[float],
    *,
    top_k: int = 3,
) -> dict[str, float]:
    """Compute all ablation metrics in one call."""
    return {
        "nrmse_pct": compute_nrmse(exact_values, estimated_values),
        "rmse": compute_rmse(exact_values, estimated_values),
        "max_abs_error": compute_max_absolute_error(exact_values, estimated_values),
        "kendall_tau": compute_kendall_tau(exact_values, estimated_values),
        "spearman_rho": compute_spearman_rho(exact_values, estimated_values),
        "top_k_precision": compute_top_k_precision(exact_values, estimated_values, top_k),
        "bias": compute_bias(exact_values, estimated_values),
    }
