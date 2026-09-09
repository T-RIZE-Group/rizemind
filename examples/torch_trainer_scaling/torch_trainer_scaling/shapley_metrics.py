from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

WEIGHT_DECIMALS = 18


def _popcount(value: int) -> int:
    count = 0
    while value != 0:
        value &= value - 1
        count += 1
    return count


def _weight(total_trainers: int, coalition_size_without_trainer: int) -> int:
    if total_trainers <= 0:
        raise ValueError("total_trainers must be positive.")
    if coalition_size_without_trainer >= total_trainers:
        raise ValueError(
            "coalition_size_without_trainer must be smaller than total_trainers."
        )

    size = coalition_size_without_trainer
    half = (total_trainers - 1) // 2
    if size > half:
        size = (total_trainers - 1) - size

    weight = (10**WEIGHT_DECIMALS) // total_trainers
    for value in range(size):
        weight = (weight * (value + 1)) // ((total_trainers - 1) - value)
    return weight


def compute_shapley_values(
    sampled_masks: Sequence[int],
    utility_by_mask: Mapping[int, float],
    total_trainers: int,
    *,
    missing_policy: str = "raise",
) -> np.ndarray:
    if total_trainers <= 0:
        raise ValueError("total_trainers must be positive.")
    if missing_policy not in {"raise", "skip"}:
        raise ValueError("missing_policy must be either 'raise' or 'skip'.")

    shapley_values = np.zeros(total_trainers, dtype=np.float64)
    if len(sampled_masks) == 0:
        return shapley_values

    for trainer_index in range(total_trainers):
        weighted_sum = 0.0
        weight_total = 0
        trainer_mask = 1 << trainer_index

        for generated_mask in sampled_masks:
            if generated_mask & trainer_mask:
                with_trainer_mask = generated_mask
                without_trainer_mask = generated_mask & ~trainer_mask
            else:
                without_trainer_mask = generated_mask
                with_trainer_mask = generated_mask | trainer_mask

            if (
                with_trainer_mask not in utility_by_mask
                or without_trainer_mask not in utility_by_mask
            ):
                if missing_policy == "skip":
                    continue
                missing_mask = (
                    with_trainer_mask
                    if with_trainer_mask not in utility_by_mask
                    else without_trainer_mask
                )
                raise KeyError(
                    f"Missing utility for coalition mask {missing_mask} "
                    f"while computing trainer {trainer_index}."
                )

            with_result = float(utility_by_mask[with_trainer_mask])
            without_result = float(utility_by_mask[without_trainer_mask])

            contribution = with_result - without_result
            weight = _weight(total_trainers, _popcount(without_trainer_mask))
            weighted_sum += contribution * weight
            weight_total += weight

        shapley_values[trainer_index] = (
            0.0 if weight_total == 0 else weighted_sum / weight_total
        )

    return shapley_values


def compute_stratified_shapley_values(
    sampled_masks: Sequence[int],
    utility_by_mask: Mapping[int, float],
    total_trainers: int,
    *,
    missing_policy: str = "raise",
) -> np.ndarray:
    if total_trainers <= 0:
        raise ValueError("total_trainers must be positive.")
    if missing_policy not in {"raise", "skip"}:
        raise ValueError("missing_policy must be either 'raise' or 'skip'.")

    shapley_values = np.zeros(total_trainers, dtype=np.float64)
    if len(sampled_masks) == 0:
        return shapley_values

    for trainer_index in range(total_trainers):
        trainer_mask = 1 << trainer_index
        stratum_sums = np.zeros(total_trainers, dtype=np.float64)
        stratum_counts = np.zeros(total_trainers, dtype=np.int64)

        for generated_mask in sampled_masks:
            if generated_mask & trainer_mask:
                with_trainer_mask = generated_mask
                without_trainer_mask = generated_mask & ~trainer_mask
            else:
                without_trainer_mask = generated_mask
                with_trainer_mask = generated_mask | trainer_mask

            if (
                with_trainer_mask not in utility_by_mask
                or without_trainer_mask not in utility_by_mask
            ):
                if missing_policy == "skip":
                    continue
                missing_mask = (
                    with_trainer_mask
                    if with_trainer_mask not in utility_by_mask
                    else without_trainer_mask
                )
                raise KeyError(
                    f"Missing utility for coalition mask {missing_mask} "
                    f"while computing trainer {trainer_index}."
                )

            with_result = float(utility_by_mask[with_trainer_mask])
            without_result = float(utility_by_mask[without_trainer_mask])

            contribution = with_result - without_result
            s = _popcount(without_trainer_mask)

            stratum_sums[s] += contribution
            stratum_counts[s] += 1

        total_stratum_avg = 0.0
        for s in range(total_trainers):
            if stratum_counts[s] > 0:
                total_stratum_avg += stratum_sums[s] / stratum_counts[s]

        # In pure stratified estimators, we always divide by N assuming full rank representation
        shapley_values[trainer_index] = total_stratum_avg / total_trainers

    return shapley_values


def compute_nrmse(
    exact_values: Sequence[float],
    estimated_values: Sequence[float],
) -> float:
    exact = np.asarray(exact_values, dtype=np.float64).reshape(-1)
    estimated = np.asarray(estimated_values, dtype=np.float64).reshape(-1)

    if exact.size == 0 or estimated.size == 0:
        raise ValueError("exact_values and estimated_values must be non-empty.")
    if exact.shape != estimated.shape:
        raise ValueError("exact_values and estimated_values must have equal length.")

    rmse = float(np.sqrt(np.mean(np.square(estimated - exact))))
    exact_mean = float(np.mean(exact))

    if np.isclose(exact_mean, 0.0):
        return 0.0 if np.isclose(rmse, 0.0) else float("inf")

    return (rmse / exact_mean) * 100.0
