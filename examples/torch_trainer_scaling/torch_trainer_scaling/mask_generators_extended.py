"""Extended mask generators for DP-Shapley ablation study.

Implements stratified sampling (with combinatorial unranking) and
antithetic (complement) pairing as SampledMaskGenerator subclasses.
"""

from __future__ import annotations

import math
import random

from web3 import Web3

from .mask_generators import (
    DEFAULT_DETERMINISTIC_ADDRESS,
    SampledMaskGenerator,
)

STRATIFIED_WEIGHT_PRECISION = 10**18


def _unrank_combination(n: int, s: int, rank: int) -> int:
    """Return the bitmask for the rank-th subset of size s from {0,...,n-1}.

    Uses lexicographic ordering. O(n*s) per call.
    """
    if s == 0:
        return 0
    mask = 0
    remaining = rank
    start = 0
    for positions_left in range(s, 0, -1):
        for candidate in range(start, n - positions_left + 1):
            count = math.comb(n - candidate - 1, positions_left - 1)
            if remaining < count:
                mask |= 1 << candidate
                start = candidate + 1
                break
            remaining -= count
    return mask


def _allocate_proportional(n: int, K: int) -> dict[int, int]:
    """Allocate K samples across coalition sizes 0..n-1 proportional to Shapley weights.

    Shapley weight for size s: w(s) = s!(n-1-s)!/n! = 1/(n * C(n-1, s)).
    More samples at extreme sizes (s near 0 or n-1) where weights are highest.
    """
    if K <= 0:
        return {}
    raw_weights: dict[int, float] = {}
    for s in range(n):
        raw_weights[s] = 1.0 / (n * math.comb(n - 1, s))
    total_w = sum(raw_weights.values())

    allocation: dict[int, int] = {}
    for s in range(n):
        allocation[s] = max(1, round(K * raw_weights[s] / total_w))

    # Adjust to sum to exactly K by modifying the largest stratum
    diff = K - sum(allocation.values())
    if diff != 0:
        # Find stratum with largest allocation to absorb the difference
        max_stratum = max(allocation, key=lambda s: allocation[s])
        allocation[max_stratum] = max(1, allocation[max_stratum] + diff)

    return allocation


def _allocate_uniform(n: int, K: int) -> dict[int, int]:
    """Allocate K samples uniformly across coalition sizes 0..n-1."""
    if K <= 0:
        return {}
    base = K // n
    remainder = K % n
    allocation: dict[int, int] = {}
    for s in range(n):
        allocation[s] = base + (1 if s < remainder else 0)
    return allocation


def _floor_to_even(value: int) -> int:
    return value & ~1


def _stratum_weight(n: int, coalition_size: int) -> int:
    if n <= 1:
        return STRATIFIED_WEIGHT_PRECISION

    denominator = math.comb(n - 1, coalition_size)
    weight = STRATIFIED_WEIGHT_PRECISION // denominator
    return weight if weight > 0 else 1


def _allocate_lower_strata_proportional(n: int, base_budget: int) -> dict[int, int]:
    strata_count = (n // 2) + 1
    allocations = {size: 0 for size in range(strata_count)}
    if base_budget <= 0:
        return allocations

    remaining_capacities = {
        size: math.comb(n, size) for size in range(strata_count)
    }
    weights = {size: _stratum_weight(n, size) for size in range(strata_count)}
    active = {size: True for size in range(strata_count)}
    remaining_budget = base_budget

    while remaining_budget > 0:
        total_weight = sum(
            weights[size] for size in range(strata_count) if active[size]
        )

        if total_weight == 0:
            for size in range(strata_count):
                if remaining_budget <= 0:
                    break
                capacity = remaining_capacities[size]
                if capacity <= 0:
                    continue
                grant = min(remaining_budget, capacity)
                allocations[size] += grant
                remaining_capacities[size] -= grant
                remaining_budget -= grant
            break

        saturated_any = False
        budget_snapshot = remaining_budget
        saturated_budget = 0
        for size in range(strata_count):
            if not active[size]:
                continue
            quota_floor = (budget_snapshot * weights[size]) // total_weight
            if quota_floor >= remaining_capacities[size]:
                saturated_budget += remaining_capacities[size]
                allocations[size] += remaining_capacities[size]
                remaining_capacities[size] = 0
                active[size] = False
                saturated_any = True

        if saturated_any:
            remaining_budget = budget_snapshot - saturated_budget
            continue

        remainders: dict[int, int] = {}
        distributed = 0
        for size in range(strata_count):
            if not active[size]:
                continue
            numerator = budget_snapshot * weights[size]
            share = numerator // total_weight
            allocations[size] += share
            remaining_capacities[size] -= share
            distributed += share
            remainders[size] = numerator % total_weight
            if remaining_capacities[size] == 0:
                active[size] = False

        leftover = budget_snapshot - distributed
        while leftover > 0:
            candidates = [
                size for size in range(strata_count) if remaining_capacities[size] > 0
            ]
            if not candidates:
                raise ValueError("No lower stratum capacity left for remaining budget.")
            best_size = max(candidates, key=lambda size: (remainders.get(size, 0), -size))
            allocations[best_size] += 1
            remaining_capacities[best_size] -= 1
            remainders[best_size] = 0
            if remaining_capacities[best_size] == 0:
                active[best_size] = False
            leftover -= 1

        remaining_budget = 0

    return allocations


def _allocate_all_strata_proportional(n: int, budget: int) -> dict[int, int]:
    allocations = {size: 0 for size in range(n)}
    if budget <= 0:
        return allocations

    remaining_capacities = {size: math.comb(n, size) for size in range(n)}
    weights = {size: _stratum_weight(n, size) for size in range(n)}
    active = {size: True for size in range(n)}
    remaining_budget = budget

    while remaining_budget > 0:
        total_weight = sum(weights[size] for size in range(n) if active[size])

        if total_weight == 0:
            for size in range(n):
                if remaining_budget <= 0:
                    break
                capacity = remaining_capacities[size]
                if capacity <= 0:
                    continue
                grant = min(remaining_budget, capacity)
                allocations[size] += grant
                remaining_capacities[size] -= grant
                remaining_budget -= grant
            break

        saturated_any = False
        budget_snapshot = remaining_budget
        saturated_budget = 0
        for size in range(n):
            if not active[size]:
                continue
            quota_floor = (budget_snapshot * weights[size]) // total_weight
            if quota_floor >= remaining_capacities[size]:
                saturated_budget += remaining_capacities[size]
                allocations[size] += remaining_capacities[size]
                remaining_capacities[size] = 0
                active[size] = False
                saturated_any = True

        if saturated_any:
            remaining_budget = budget_snapshot - saturated_budget
            continue

        remainders: dict[int, int] = {}
        distributed = 0
        for size in range(n):
            if not active[size]:
                continue
            numerator = budget_snapshot * weights[size]
            share = numerator // total_weight
            allocations[size] += share
            remaining_capacities[size] -= share
            distributed += share
            remainders[size] = numerator % total_weight
            if remaining_capacities[size] == 0:
                active[size] = False

        leftover = budget_snapshot - distributed
        while leftover > 0:
            candidates = [
                size for size in range(n) if remaining_capacities[size] > 0
            ]
            if not candidates:
                raise ValueError("No stratum capacity left for remaining budget.")
            best_size = max(
                candidates, key=lambda size: (remainders.get(size, 0), -size)
            )
            allocations[best_size] += 1
            remaining_capacities[best_size] -= 1
            remainders[best_size] = 0
            if remaining_capacities[best_size] == 0:
                active[best_size] = False
            leftover -= 1

        remaining_budget = 0

    return allocations


def _allocate_all_strata_proportional_with_replacement(
    n: int,
    budget: int,
) -> dict[int, int]:
    allocations = {size: 0 for size in range(n)}
    if budget <= 0:
        return allocations

    weights = {size: _stratum_weight(n, size) for size in range(n)}
    total_weight = sum(weights.values())
    remainders: dict[int, int] = {}
    distributed = 0

    for size in range(n):
        numerator = budget * weights[size]
        allocations[size] = numerator // total_weight
        remainders[size] = numerator % total_weight
        distributed += allocations[size]

    leftover = budget - distributed
    while leftover > 0:
        best_size = max(range(n), key=lambda size: (remainders.get(size, 0), -size))
        allocations[best_size] += 1
        remainders[best_size] = 0
        leftover -= 1

    return allocations


def _bit_len(value: int) -> int:
    return value.bit_length()


def _feistel(value: int, n_bits: int, seed: bytes) -> int:
    half = n_bits >> 1
    mask = (1 << half) - 1 if half > 0 else 0

    right = value & mask
    left = value >> half

    for round_index in range(4):
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


def _rand(seed: bytes, index: int, max_value: int) -> int:
    if max_value <= 0:
        raise ValueError("max_value must be greater than zero.")
    if index >= max_value:
        raise ValueError("index must be smaller than max_value.")

    n_bits = _bit_len(max_value - 1)
    if n_bits % 2 == 1:
        n_bits += 1

    value = index
    while True:
        value = _feistel(value, n_bits, seed)
        if value < max_value:
            return value


def _rng_rand(seed: bytes, index: int, max_value: int) -> int:
    if max_value <= 0:
        raise ValueError("max_value must be greater than zero.")

    limit = ((1 << 256) - 1) - (((1 << 256) - 1) % max_value)
    probe = index
    while True:
        digest = Web3.solidity_keccak(["bytes32", "uint256"], [seed, probe])
        value = int.from_bytes(digest, "big")
        if value < limit:
            return value % max_value
        probe += 1


class StratifiedMaskGenerator(SampledMaskGenerator):
    """Stratified coalition-size sampling with combinatorial unranking.

    Distributes the budget K across coalition sizes s=0,...,n-1.
    Within each stratum, uses unranking to select subsets deterministically
    from a random permutation of subset indices.
    """

    name = "stratified"

    def __init__(
        self,
        sample_budget: int,
        seed: int,
        *,
        allocation: str = "proportional",
    ) -> None:
        if sample_budget <= 0:
            raise ValueError("sample_budget must be positive.")
        if allocation not in ("proportional", "uniform"):
            raise ValueError("allocation must be 'proportional' or 'uniform'.")
        self.sample_budget = sample_budget
        self.seed = seed
        self.allocation = allocation

    def generate(self, total_trainers: int) -> list[int]:
        n = total_trainers
        total_masks = 1 << n
        budget = min(self.sample_budget, total_masks)

        if self.allocation == "proportional":
            alloc = _allocate_proportional(n, budget)
        else:
            alloc = _allocate_uniform(n, budget)

        rng = random.Random(self.seed)
        masks: list[int] = []

        for size in range(n):
            k_s = alloc.get(size, 0)
            if k_s <= 0:
                continue
            total_of_size = math.comb(n, size)
            if k_s >= total_of_size:
                # Include all subsets of this size
                for rank in range(total_of_size):
                    masks.append(_unrank_combination(n, size, rank))
            else:
                # Sample k_s random indices and unrank them
                indices = rng.sample(range(total_of_size), k_s)
                for rank in indices:
                    masks.append(_unrank_combination(n, size, rank))

        return masks


class ContractParityStratifiedMaskGenerator(SampledMaskGenerator):
    """Mirror the Solidity helper-only pure stratified sampler exactly."""

    name = "stratified"

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

    def emitted_budget(self, total_trainers: int) -> int:
        return min(self.sample_budget, (1 << total_trainers) - 1)

    def allocation_for(self, total_trainers: int) -> dict[int, int]:
        return _allocate_all_strata_proportional(
            total_trainers, self.emitted_budget(total_trainers)
        )

    def generate(self, total_trainers: int) -> list[int]:
        allocations = self.allocation_for(total_trainers)
        masks: list[int] = []
        for size in range(total_trainers):
            sample_count = allocations.get(size, 0)
            if sample_count <= 0:
                continue
            total_of_size = math.comb(total_trainers, size)
            stratum_seed = Web3.solidity_keccak(
                ["address", "uint256", "uint8", "uint256"],
                [self.address_seed, self.round_id, total_trainers, size],
            )
            for local_index in range(sample_count):
                rank = _rand(stratum_seed, local_index, total_of_size)
                masks.append(_unrank_combination(total_trainers, size, rank))
        return masks


class ContractParityStratifiedWithDuplicatesMaskGenerator(SampledMaskGenerator):
    """Mirror the Solidity helper-only stratified sampler with replacement."""

    name = "stratified_with_duplicates"

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

    def allocation_for(self, total_trainers: int) -> dict[int, int]:
        total_masks = 1 << total_trainers
        budget = min(self.sample_budget, total_masks)
        return _allocate_all_strata_proportional_with_replacement(
            total_trainers,
            budget,
        )

    def generate(self, total_trainers: int) -> list[int]:
        allocations = self.allocation_for(total_trainers)
        masks: list[int] = []

        for size in range(total_trainers):
            sample_count = allocations.get(size, 0)
            if sample_count <= 0:
                continue
            total_of_size = math.comb(total_trainers, size)
            stratum_seed = Web3.solidity_keccak(
                ["address", "uint256", "uint8", "uint256"],
                [self.address_seed, self.round_id, total_trainers, size],
            )
            for local_index in range(sample_count):
                rank = _rng_rand(stratum_seed, local_index, total_of_size)
                masks.append(_unrank_combination(total_trainers, size, rank))

        return masks

class AntitheticMaskGenerator(SampledMaskGenerator):
    """Wraps a base generator, adding the complement of each sampled mask.

    Requests K/2 masks from the base generator, then for each mask m adds
    the complement ((1 << n) - 1) ^ m. Total output: K masks.
    """

    name = "antithetic"

    def __init__(self, base_generator: SampledMaskGenerator) -> None:
        self.base_generator = base_generator

    def generate(self, total_trainers: int) -> list[int]:
        base_masks = self.base_generator.generate(total_trainers)
        full_mask = (1 << total_trainers) - 1

        # Take K/2 base masks and add complements
        half = len(base_masks) // 2
        base_half = base_masks[:half]

        masks: list[int] = []
        for m in base_half:
            masks.append(m)
            masks.append(full_mask ^ m)

        # If original budget was odd, add one more base mask without complement
        if len(base_masks) % 2 == 1 and half < len(base_masks):
            masks.append(base_masks[half])

        return masks


class StratifiedAntitheticMaskGenerator(SampledMaskGenerator):
    """Stratified sampling with antithetic pairing.

    Samples from strata s=0,...,floor(n/2). For each mask of size s, the
    complement has size n-s, automatically covering the upper strata.
    """

    name = "stratified_antithetic"

    def __init__(
        self,
        sample_budget: int,
        seed: int | None = None,
        *,
        allocation: str = "proportional",
        address_seed: str | None = None,
        round_id: int = 0,
    ) -> None:
        if sample_budget <= 0:
            raise ValueError("sample_budget must be positive.")
        if allocation != "proportional":
            raise ValueError("allocation must be 'proportional'.")
        self.sample_budget = sample_budget
        self.seed = seed
        self.allocation = allocation
        self.address_seed = (
            Web3.to_checksum_address(address_seed)
            if address_seed is not None
            else None
        )
        self.round_id = round_id

    def generate(self, total_trainers: int) -> list[int]:
        n = total_trainers
        total_masks = 1 << n
        budget = _floor_to_even(min(self.sample_budget, total_masks))
        full_mask = (1 << n) - 1
        if budget == 0:
            return []

        half_budget = budget // 2
        allocations = _allocate_lower_strata_proportional(n, half_budget)
        masks: list[int] = []

        if self.address_seed is None:
            rng = random.Random(self.seed)
            for size in range((n // 2) + 1):
                k_s = allocations.get(size, 0)
                if k_s <= 0:
                    continue
                total_of_size = math.comb(n, size)
                if k_s >= total_of_size:
                    ranks = range(total_of_size)
                else:
                    ranks = rng.sample(range(total_of_size), k_s)
                for rank in ranks:
                    mask = _unrank_combination(n, size, rank)
                    masks.append(mask)
                    masks.append(full_mask ^ mask)
            return masks

        for size in range((n // 2) + 1):
            k_s = allocations.get(size, 0)
            if k_s <= 0:
                continue
            total_of_size = math.comb(n, size)
            stratum_seed = Web3.solidity_keccak(
                ["address", "uint256", "uint8", "uint256"],
                [self.address_seed, self.round_id, n, size],
            )
            for local_index in range(k_s):
                rank = _rand(stratum_seed, local_index, total_of_size)
                mask = _unrank_combination(n, size, rank)
                masks.append(mask)
                masks.append(full_mask ^ mask)

        return masks

    def emitted_budget(self, total_trainers: int) -> int:
        total_masks = 1 << total_trainers
        return _floor_to_even(min(self.sample_budget, total_masks))

    def lower_strata_allocation_for(self, total_trainers: int) -> dict[int, int]:
        return _allocate_lower_strata_proportional(
            total_trainers, self.emitted_budget(total_trainers) // 2
        )

    def emitted_size_allocation_for(self, total_trainers: int) -> dict[int, int]:
        allocations: dict[int, int] = {}
        full_size = total_trainers
        for size, count in self.lower_strata_allocation_for(total_trainers).items():
            if count <= 0:
                continue
            allocations[size] = allocations.get(size, 0) + count
            complement_size = full_size - size
            allocations[complement_size] = (
                allocations.get(complement_size, 0) + count
            )
        return allocations
