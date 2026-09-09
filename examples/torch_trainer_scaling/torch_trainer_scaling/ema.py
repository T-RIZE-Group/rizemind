"""Exponential Moving Average (EMA) smoothing for multi-round Shapley values.

Smooths raw per-round Shapley scores across rounds before settlement:
    phi_smoothed(r) = alpha * phi_raw(r) + (1 - alpha) * phi_smoothed(r-1)
"""

from __future__ import annotations

import numpy as np


class EMAShapleyTracker:
    """Track and smooth Shapley values across federated learning rounds."""

    def __init__(self, alpha: float, n_trainers: int) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        if n_trainers <= 0:
            raise ValueError("n_trainers must be positive.")
        self.alpha = alpha
        self.n_trainers = n_trainers
        self._smoothed: np.ndarray | None = None
        self._history: list[np.ndarray] = []
        self._smoothed_history: list[np.ndarray] = []

    def update(self, raw_values: np.ndarray) -> np.ndarray:
        """Apply EMA smoothing to a new round's Shapley values.

        Returns the smoothed values for this round.
        """
        raw = np.asarray(raw_values, dtype=np.float64)
        if raw.shape != (self.n_trainers,):
            raise ValueError(
                f"Expected shape ({self.n_trainers},), got {raw.shape}."
            )

        self._history.append(raw.copy())

        if self._smoothed is None:
            self._smoothed = raw.copy()
        else:
            self._smoothed = self.alpha * raw + (1.0 - self.alpha) * self._smoothed

        self._smoothed_history.append(self._smoothed.copy())
        return self._smoothed.copy()

    def get_smoothed(self) -> np.ndarray | None:
        """Return the current smoothed values, or None if no updates yet."""
        return self._smoothed.copy() if self._smoothed is not None else None

    @property
    def history(self) -> list[np.ndarray]:
        """Raw Shapley values for each round."""
        return self._history

    @property
    def smoothed_history(self) -> list[np.ndarray]:
        """Smoothed Shapley values for each round."""
        return self._smoothed_history

    def reset(self) -> None:
        """Clear all state."""
        self._smoothed = None
        self._history.clear()
        self._smoothed_history.clear()

    def payout_volatility(self) -> np.ndarray:
        """Coefficient of variation of smoothed payouts per trainer across rounds.

        Returns one value per trainer.
        """
        if len(self._smoothed_history) < 2:
            return np.zeros(self.n_trainers, dtype=np.float64)
        stacked = np.stack(self._smoothed_history)  # (rounds, n_trainers)
        means = np.mean(stacked, axis=0)
        stds = np.std(stacked, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(np.abs(means) > 1e-12, stds / np.abs(means), 0.0)
        return cv

    def adaptation_lag(
        self,
        true_values_by_round: list[np.ndarray],
        shift_round: int,
        *,
        threshold: float = 0.5,
    ) -> float:
        """Rounds until smoothed payout reflects a trainer's true change by >threshold.

        Measures the average lag across all trainers that experienced a
        significant shift at *shift_round*.
        """
        if shift_round >= len(self._smoothed_history) or shift_round < 1:
            return float("nan")

        true_before = np.asarray(true_values_by_round[shift_round - 1])
        true_after = np.asarray(true_values_by_round[shift_round])
        shift = np.abs(true_after - true_before)
        shifted_trainers = np.where(shift > 1e-8)[0]

        if len(shifted_trainers) == 0:
            return 0.0

        lags: list[int] = []
        for i in shifted_trainers:
            target_change = shift[i] * threshold
            smoothed_before = self._smoothed_history[shift_round - 1][i]
            found = False
            for r in range(shift_round, len(self._smoothed_history)):
                if abs(self._smoothed_history[r][i] - smoothed_before) >= target_change:
                    lags.append(r - shift_round)
                    found = True
                    break
            if not found:
                lags.append(len(self._smoothed_history) - shift_round)

        return float(np.mean(lags))
