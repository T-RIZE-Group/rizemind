from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from web3 import Web3

from .coalition_strategies import Coalition

DEFAULT_DETERMINISTIC_ADDRESS = "0x0000000000000000000000000000000000000001"


@dataclass(frozen=True)
class EvaluationMask:
    mask: int
    sample_role: str
    sample_order: int | None


class SampledMaskGenerator(ABC):
    name: str

    @abstractmethod
    def generate(self, total_trainers: int) -> list[int]:
        """Return the ordered sampled masks used to estimate Shapley values."""


class AllMasksGenerator(SampledMaskGenerator):
    name = "exact"

    def generate(self, total_trainers: int) -> list[int]:
        total_masks = 1 << total_trainers
        return list(range(total_masks))


class MonteCarloMaskGenerator(SampledMaskGenerator):
    name = "monte_carlo"

    def __init__(self, sample_budget: int, seed: int) -> None:
        if sample_budget <= 0:
            raise ValueError("sample_budget must be positive.")
        self.sample_budget = sample_budget
        self.seed = seed

    def generate(self, total_trainers: int) -> list[int]:
        total_masks = 1 << total_trainers
        sample_count = min(self.sample_budget, total_masks)
        rng = random.Random(self.seed)
        return rng.choices(range(total_masks), k=sample_count)


class DeterministicContractMaskGenerator(SampledMaskGenerator):
    name = "deterministic"
    _rounds = 4

    def __init__(
        self,
        sample_budget: int,
        *,
        address_seed: str = DEFAULT_DETERMINISTIC_ADDRESS,
        round_id: int = 0,
    ) -> None:
        if sample_budget <= 0:
            raise ValueError("sample_budget must be positive.")
        self.sample_budget = sample_budget
        self.address_seed = Web3.to_checksum_address(address_seed)
        self.round_id = round_id

    def generate(self, total_trainers: int) -> list[int]:
        total_masks = 1 << total_trainers
        sample_count = min(self.sample_budget, total_masks)
        seed = Web3.solidity_keccak(
            ["address", "uint256"],
            [self.address_seed, self.round_id],
        )
        return [self._rand(seed, index, total_masks) for index in range(sample_count)]

    def _rand(self, seed: bytes, index: int, max_value: int) -> int:
        if max_value <= 0:
            raise ValueError("max_value must be greater than zero.")
        if index >= max_value:
            raise ValueError("index must be smaller than max_value.")

        n_bits = self._bit_len(max_value - 1)
        if n_bits % 2 == 1:
            n_bits += 1

        value = index
        while True:
            value = self._feistel(value, n_bits, seed)
            if value < max_value:
                return value

    def _feistel(self, value: int, n_bits: int, seed: bytes) -> int:
        half = n_bits >> 1
        mask = (1 << half) - 1 if half > 0 else 0

        right = value & mask
        left = value >> half

        for round_index in range(self._rounds):
            round_hash = Web3.solidity_keccak(
                ["bytes32", "uint256", "uint256"],
                [seed, round_index, right],
            )
            function_value = int.from_bytes(round_hash, "big") & mask
            new_left = right
            new_right = (left ^ function_value) & mask
            left = new_left
            right = new_right

        return (left << half) | right

    @staticmethod
    def _bit_len(value: int) -> int:
        return value.bit_length()


class AntitheticPairingMaskGenerator(SampledMaskGenerator):
    """Pair deterministic masks with their complements."""

    name = "antithetic"

    def __init__(
        self,
        sample_budget: int,
        *,
        address_seed: str = DEFAULT_DETERMINISTIC_ADDRESS,
        round_id: int = 0,
    ) -> None:
        if sample_budget <= 0:
            raise ValueError("sample_budget must be positive.")
        self.sample_budget = sample_budget
        self.address_seed = address_seed
        self.round_id = round_id

    def generate(self, total_trainers: int) -> list[int]:
        total_masks = 1 << total_trainers
        sample_count = min(self.sample_budget, total_masks)
        if sample_count == 0:
            return []

        base_budget = (sample_count + 1) // 2
        base_masks = DeterministicContractMaskGenerator(
            base_budget,
            address_seed=self.address_seed,
            round_id=self.round_id,
        ).generate(total_trainers)
        full_mask = total_masks - 1

        masks: list[int] = []
        seen_masks: set[int] = set()
        for base_mask in base_masks:
            for candidate in (base_mask, full_mask ^ base_mask):
                if candidate in seen_masks:
                    continue
                masks.append(candidate)
                seen_masks.add(candidate)
                if len(masks) >= sample_count:
                    return masks

        return masks


def _unrank_combination(total_trainers: int, coalition_size: int, rank: int) -> int:
    """Return the bitmask for the rank-th subset of the requested size."""
    if coalition_size == 0:
        return 0

    mask = 0
    remaining = rank
    start = 0
    for positions_left in range(coalition_size, 0, -1):
        for candidate in range(start, total_trainers - positions_left + 1):
            count = math.comb(total_trainers - candidate - 1, positions_left - 1)
            if remaining < count:
                mask |= 1 << candidate
                start = candidate + 1
                break
            remaining -= count
    return mask


class StratifiedSeedKeyedMaskGenerator(SampledMaskGenerator):
    """Include all "last-k" coalitions, then backfill with deterministic masks."""

    name = "stratified_seed_keyed"

    def __init__(
        self,
        sample_budget: int,
        seed: int,
        *,
        max_missing_trainers: int = 5,
        min_coalition_size: int = 2,
    ) -> None:
        if sample_budget <= 0:
            raise ValueError("sample_budget must be positive.")
        if max_missing_trainers <= 0:
            raise ValueError("max_missing_trainers must be positive.")
        if min_coalition_size <= 0:
            raise ValueError("min_coalition_size must be positive.")
        self.sample_budget = sample_budget
        self.seed = seed
        self.max_missing_trainers = max_missing_trainers
        self.min_coalition_size = min_coalition_size

    def generate(self, total_trainers: int) -> list[int]:
        highest_size = total_trainers - 1
        lowest_size = max(
            total_trainers - self.max_missing_trainers,
            self.min_coalition_size,
        )
        if highest_size < lowest_size:
            return []

        budget = min(self.sample_budget, 1 << total_trainers)
        selected_sizes = list(range(highest_size, lowest_size - 1, -1))

        masks: list[int] = []
        seen_masks: set[int] = set()

        for size in selected_sizes:
            total_of_size = math.comb(total_trainers, size)
            for rank in range(total_of_size):
                mask = _unrank_combination(total_trainers, size, rank)
                if mask in seen_masks:
                    continue
                masks.append(mask)
                seen_masks.add(mask)
                if len(masks) >= budget:
                    return masks

        deterministic_masks = DeterministicContractMaskGenerator(
            1 << total_trainers,
            round_id=self.seed,
        ).generate(total_trainers)
        for mask in deterministic_masks:
            if mask in seen_masks:
                continue
            masks.append(mask)
            seen_masks.add(mask)
            if len(masks) >= budget:
                break

        return masks


def mask_to_coalition(mask: int, total_trainers: int) -> Coalition:
    return tuple(
        trainer_index
        for trainer_index in range(total_trainers)
        if mask & (1 << trainer_index)
    )


def expand_masks_for_evaluation(
    sampled_masks: list[int],
    total_trainers: int,
    *,
    sample_role: str,
) -> list[EvaluationMask]:
    if sample_role not in {"exact", "sampled"}:
        raise ValueError("sample_role must be either 'exact' or 'sampled'.")

    plans_by_mask: dict[int, EvaluationMask] = {}
    ordered_masks: list[int] = []

    def record(mask: int, role: str, order: int | None) -> None:
        existing = plans_by_mask.get(mask)
        if existing is None:
            ordered_masks.append(mask)
            plans_by_mask[mask] = EvaluationMask(
                mask=mask,
                sample_role=role,
                sample_order=order,
            )
            return
        if role in {"exact", "sampled"} and existing.sample_role == "support":
            plans_by_mask[mask] = EvaluationMask(
                mask=mask,
                sample_role=role,
                sample_order=order,
            )

    for sample_order, mask in enumerate(sampled_masks):
        record(mask, sample_role, sample_order)
        if sample_role == "exact":
            continue
        for trainer_index in range(total_trainers):
            record(mask ^ (1 << trainer_index), "support", None)

    return [plans_by_mask[mask] for mask in ordered_masks]
