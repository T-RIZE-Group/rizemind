"""Synthetic cooperative game generators for Shapley value ablation studies.

Each game defines a utility function v(S) for any coalition S (represented as
a bitmask) and can compute exact Shapley values via brute-force enumeration.
"""

from __future__ import annotations

import csv
import math
import random
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class SyntheticGame(ABC):
    """Abstract base for cooperative games with bitmask coalition representation."""

    def __init__(self, n: int, seed: int) -> None:
        if n <= 0:
            raise ValueError("n must be positive.")
        self.n = n
        self.seed = seed
        self._utility_cache: dict[int, float] = {}

    @abstractmethod
    def _compute_utility(self, mask: int) -> float:
        """Compute v(S) for the coalition represented by *mask*."""

    def utility(self, mask: int) -> float:
        """Return v(S) with caching."""
        cached = self._utility_cache.get(mask)
        if cached is not None:
            return cached
        value = self._compute_utility(mask)
        self._utility_cache[mask] = value
        return value

    def exact_shapley(self) -> np.ndarray:
        """Brute-force exact Shapley values by enumerating all 2^n coalitions."""
        n = self.n
        total_masks = 1 << n
        shapley = np.zeros(n, dtype=np.float64)

        for i in range(n):
            player_bit = 1 << i
            weighted_sum = 0.0
            weight_total = 0.0
            for mask in range(total_masks):
                if mask & player_bit:
                    with_mask = mask
                    without_mask = mask & ~player_bit
                else:
                    without_mask = mask
                    with_mask = mask | player_bit

                s = bin(without_mask).count("1")
                weight = math.factorial(s) * math.factorial(n - 1 - s) / math.factorial(n)
                contribution = self.utility(with_mask) - self.utility(without_mask)
                weighted_sum += weight * contribution
                weight_total += weight

            shapley[i] = weighted_sum / weight_total if weight_total > 0 else 0.0

        return shapley


def _popcount(value: int) -> int:
    return bin(value).count("1")


def _members(mask: int, n: int) -> list[int]:
    """Return the list of player indices present in *mask*."""
    return [i for i in range(n) if mask & (1 << i)]


class SuperadditiveUniformGame(SyntheticGame):
    """v(S) = sum(w_i for i in S) + alpha * sum(w_i*w_j for i<j in S).

    Weights w_i ~ Uniform(0, 1). alpha controls interaction strength.
    """

    def __init__(self, n: int, seed: int, *, alpha: float = 0.5) -> None:
        super().__init__(n, seed)
        self.alpha = alpha
        rng = random.Random(seed)
        self.weights = [rng.random() for _ in range(n)]

    def _compute_utility(self, mask: int) -> float:
        members = _members(mask, self.n)
        if not members:
            return 0.0
        individual = sum(self.weights[i] for i in members)
        interaction = 0.0
        for idx_a in range(len(members)):
            for idx_b in range(idx_a + 1, len(members)):
                interaction += self.weights[members[idx_a]] * self.weights[members[idx_b]]
        return individual + self.alpha * interaction


class SuperadditiveSkewedGame(SyntheticGame):
    """Like SuperadditiveUniformGame but 2-3 'strong' trainers dominate.

    Strong trainers get weights ~5x larger than weak trainers, so they
    contribute 60%+ of total utility.
    """

    def __init__(self, n: int, seed: int, *, alpha: float = 0.5) -> None:
        super().__init__(n, seed)
        self.alpha = alpha
        rng = random.Random(seed)
        n_strong = max(2, min(3, n // 3))
        strong_indices = set(rng.sample(range(n), n_strong))
        self.weights = []
        for i in range(n):
            if i in strong_indices:
                self.weights.append(rng.uniform(3.0, 5.0))
            else:
                self.weights.append(rng.uniform(0.1, 1.0))

    def _compute_utility(self, mask: int) -> float:
        members = _members(mask, self.n)
        if not members:
            return 0.0
        individual = sum(self.weights[i] for i in members)
        interaction = 0.0
        for idx_a in range(len(members)):
            for idx_b in range(idx_a + 1, len(members)):
                interaction += self.weights[members[idx_a]] * self.weights[members[idx_b]]
        return individual + self.alpha * interaction


class SubmodularGame(SyntheticGame):
    """v(S) = 1 - prod(1 - w_i for i in S).

    Diminishing returns: each additional member adds less when the coalition
    is already large. w_i ~ Uniform(0.1, 0.5) so individual contributions
    are moderate.
    """

    def __init__(self, n: int, seed: int) -> None:
        super().__init__(n, seed)
        rng = random.Random(seed)
        self.weights = [rng.uniform(0.1, 0.5) for _ in range(n)]

    def _compute_utility(self, mask: int) -> float:
        members = _members(mask, self.n)
        if not members:
            return 0.0
        product = 1.0
        for i in members:
            product *= 1.0 - self.weights[i]
        return 1.0 - product


class NonMonotoneGame(SyntheticGame):
    """Some trainers have negative marginal contributions.

    v(S) = sum(w_i for i in S) + alpha * sum(w_i*w_j for i<j in S)
    where some w_i are negative, simulating harmful participants.
    """

    def __init__(self, n: int, seed: int, *, alpha: float = 0.3) -> None:
        super().__init__(n, seed)
        self.alpha = alpha
        rng = random.Random(seed)
        n_negative = max(1, n // 4)
        negative_indices = set(rng.sample(range(n), n_negative))
        self.weights = []
        for i in range(n):
            if i in negative_indices:
                self.weights.append(rng.uniform(-1.0, -0.1))
            else:
                self.weights.append(rng.uniform(0.5, 1.5))

    def _compute_utility(self, mask: int) -> float:
        members = _members(mask, self.n)
        if not members:
            return 0.0
        individual = sum(self.weights[i] for i in members)
        interaction = 0.0
        for idx_a in range(len(members)):
            for idx_b in range(idx_a + 1, len(members)):
                interaction += self.weights[members[idx_a]] * self.weights[members[idx_b]]
        return individual + self.alpha * interaction


class PrecomputedFLGame(SyntheticGame):
    """Loads a precomputed utility table from a coalitions CSV file.

    Expects CSV with columns: coalition_mask, accuracy (used as utility).
    Typically from combined_logs/trainers-N/exact/coalitions.csv.
    """

    def __init__(self, coalitions_csv_path: str | Path, n: int) -> None:
        super().__init__(n, seed=0)
        self._load_utilities(Path(coalitions_csv_path))

    def _load_utilities(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mask = int(row["coalition_mask"])
                accuracy = float(row["accuracy"])
                self._utility_cache[mask] = accuracy

    def _compute_utility(self, mask: int) -> float:
        raise KeyError(
            f"Coalition mask {mask} not found in precomputed utility table. "
            f"PrecomputedFLGame requires all coalitions to be in the CSV."
        )


GAME_REGISTRY: dict[str, type[SyntheticGame]] = {
    "superadditive-uniform": SuperadditiveUniformGame,
    "superadditive-skewed": SuperadditiveSkewedGame,
    "submodular": SubmodularGame,
    "non-monotone": NonMonotoneGame,
}


def build_game(utility_type: str, n: int, seed: int) -> SyntheticGame:
    """Build a synthetic game by name."""
    game_cls = GAME_REGISTRY.get(utility_type)
    if game_cls is None:
        raise ValueError(
            f"Unknown utility type '{utility_type}'. "
            f"Available: {', '.join(GAME_REGISTRY)}"
        )
    return game_cls(n, seed)
