"""Shapley value estimators for the ablation study.

Two estimation paths:
- MarginalEstimator: weighted marginal contributions (wraps existing code)
- KernelSHAPEstimator: weighted least squares regression (KernelSHAP)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from .mask_generators import expand_masks_for_evaluation
from .shapley_metrics import compute_shapley_values, compute_stratified_shapley_values


class ShapleyEstimator(ABC):
    """Abstract base for Shapley value estimation from sampled coalitions."""

    name: str

    @abstractmethod
    def estimate(
        self,
        sampled_masks: list[int],
        utility_fn: Callable[[int], float],
        total_trainers: int,
    ) -> np.ndarray:
        """Estimate Shapley values from sampled masks and a utility function.

        Args:
            sampled_masks: List of coalition bitmasks to use for estimation.
            utility_fn: Function mapping a coalition bitmask to its utility value.
            total_trainers: Total number of players/trainers.

        Returns:
            Array of estimated Shapley values, one per trainer.
        """


class MarginalEstimator(ShapleyEstimator):
    """Weighted marginal contribution estimator.

    Wraps the existing compute_shapley_values function. For each sampled
    mask, expands it with support masks (single-bit-flip neighbors) to
    compute marginal contributions.

    Total evaluations: up to K * (n + 1) where K = len(sampled_masks).
    """

    name = "marginal"

    def estimate(
        self,
        sampled_masks: list[int],
        utility_fn: Callable[[int], float],
        total_trainers: int,
    ) -> np.ndarray:
        # Expand masks to include support masks for marginal computation
        evaluation_plans = expand_masks_for_evaluation(
            sampled_masks,
            total_trainers,
            sample_role="sampled",
        )

        # Evaluate all needed coalitions
        utility_by_mask: dict[int, float] = {}
        for plan in evaluation_plans:
            if plan.mask not in utility_by_mask:
                utility_by_mask[plan.mask] = utility_fn(plan.mask)

        return compute_shapley_values(sampled_masks, utility_by_mask, total_trainers)


class StratifiedEstimator(ShapleyEstimator):
    """Stratified mean-of-strata estimator using the sampled mask stream."""

    name = "stratified"

    def estimate(
        self,
        sampled_masks: list[int],
        utility_fn: Callable[[int], float],
        total_trainers: int,
    ) -> np.ndarray:
        evaluation_plans = expand_masks_for_evaluation(
            sampled_masks,
            total_trainers,
            sample_role="sampled",
        )

        utility_by_mask: dict[int, float] = {}
        for plan in evaluation_plans:
            if plan.mask not in utility_by_mask:
                utility_by_mask[plan.mask] = utility_fn(plan.mask)

        return compute_stratified_shapley_values(
            sampled_masks, utility_by_mask, total_trainers
        )


class KernelSHAPEstimator(ShapleyEstimator):
    """KernelSHAP regression-based Shapley value estimator.

    Uses weighted least squares to solve for all Shapley values
    simultaneously from the sampled coalition evaluations.

    Total evaluations: exactly K + 2 (sampled masks + empty + grand).
    No support masks needed.
    """

    name = "kernel_shap"

    def __init__(self, *, regularization: float = 1e-10) -> None:
        self.regularization = regularization

    def estimate(
        self,
        sampled_masks: list[int],
        utility_fn: Callable[[int], float],
        total_trainers: int,
    ) -> np.ndarray:
        n = total_trainers
        grand_mask = (1 << n) - 1

        # Evaluate empty and grand coalitions
        v_empty = utility_fn(0)
        v_grand = utility_fn(grand_mask)

        # Deduplicate masks, exclude empty and grand (handled separately)
        unique_masks = []
        seen = {0, grand_mask}
        for m in sampled_masks:
            if m not in seen:
                unique_masks.append(m)
                seen.add(m)

        K = len(unique_masks)
        if K == 0:
            # No informative masks — return uniform split
            delta = (v_grand - v_empty) / n if n > 0 else 0.0
            return np.full(n, delta, dtype=np.float64)

        # Build design matrix X (K x n) and response vector y (K,)
        X = np.zeros((K, n), dtype=np.float64)
        y = np.zeros(K, dtype=np.float64)
        w = np.zeros(K, dtype=np.float64)

        for k, mask in enumerate(unique_masks):
            s = bin(mask).count("1")
            for i in range(n):
                if mask & (1 << i):
                    X[k, i] = 1.0
            y[k] = utility_fn(mask) - v_empty
            # KernelSHAP weight: pi(S) = (n-1) / (C(n, |S|) * |S| * (n - |S|))
            if 0 < s < n:
                w[k] = (n - 1.0) / (math.comb(n, s) * s * (n - s))
            else:
                w[k] = 1e6  # large weight for boundary coalitions

        # Weighted least squares: phi = (X^T W X + lambda*I)^{-1} X^T W y
        W_diag = w
        XtW = X.T * W_diag  # (n, K) — broadcasting row-wise
        XtWX = XtW @ X  # (n, n)
        XtWy = XtW @ y  # (n,)

        # Regularization for numerical stability
        XtWX += self.regularization * np.eye(n)

        phi = np.linalg.solve(XtWX, XtWy)

        # Enforce efficiency constraint: sum(phi) = v(N) - v(empty)
        efficiency_gap = (v_grand - v_empty) - phi.sum()
        phi += efficiency_gap / n

        return phi
