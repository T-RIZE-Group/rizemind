// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {EvaluationStorage} from "./EvaluationStorage.sol";
import {RNG} from "../randomness/RNG.sol";
import {RandPerm} from "../randomness/RNGPermutations.sol";
import {IERC165} from "@openzeppelin-contracts-5.2.0/utils/introspection/IERC165.sol";
import {IEvaluationStorage} from "./types.sol";
import {console} from "forge-std/console.sol";

contract ShapleyValueCalculator is EvaluationStorage {
    // we index num. samples per round to make historical querying of past round accurate
    mapping(uint256 => uint256) private _numSamples;
    uint8 private constant DECIMALS = 18;
    uint256 private constant STRATIFIED_WEIGHT_PRECISION = 1e18;

    event NumSamplesSet(uint256 indexed roundId, uint256 numSamples);

    error setIdTooFar(uint256 roundId, uint256 setId, uint256 targetSetId);

    function _getTotalEvaluations(
        uint256,
        uint8 numberOfPlayers
    ) internal view returns (uint256) {
        return 1 << numberOfPlayers;
    }

    function _getEvaluationsRequired(
        uint256 roundId,
        uint8
    ) internal view returns (uint256) {
        return _numSamples[roundId];
    }

    /**
     * @dev Register a result for a given round, sample ID, set ID, model hash, and number of players
     * Shapley Value requires the set with and without the player to be evaluated.
     * So we also accept to register if the set ID is 1 hamming distance away from the target set ID.
     * @param roundId The round ID
     * @param sampleId The sample ID
     * @param setId The set ID
     * @param modelHash The model hash
     * @param result The result
     * @param numberOfPlayers The number of players
     */
    function _registerResult(
        uint256 roundId,
        uint256 sampleId,
        uint256 setId,
        bytes32 modelHash,
        int256 result,
        uint8 numberOfPlayers
    ) internal virtual {
        uint256 targetSetId = _getMask(roundId, sampleId, numberOfPlayers);
        uint256 distance = popcount(setId ^ targetSetId);
        if (distance > 1) {
            revert setIdTooFar(roundId, setId, targetSetId);
        }
        super._registerResult(roundId, setId, modelHash, result);
    }

    function _calcShapley(
        uint256 roundId,
        uint256 trainerIndex,
        uint8 numberOfPlayers
    ) internal view virtual returns (int256) {
        int256 weightedSum = 0;
        uint256 weightTotal = 0;
        uint256 numSamples = _getNumSamples(roundId);

        // console.log("calcShapley start");
        // console.log("  roundId", roundId);
        console.log("  trainerIndex", trainerIndex);
        // console.log("  numberOfPlayers", uint256(numberOfPlayers));
        // console.log("  numSamples", numSamples);

        for (uint256 i = 0; i < numSamples; ++i) {
            console.log("  i", i);
            // console.log("    gasleft_before_mask", gasleft());
            uint256 generatedMask = _getMask(roundId, i, numberOfPlayers);

            // console.log("    generatedMask", generatedMask);
            uint256 playerMask = 1 << trainerIndex;

            uint256 withTrainerMask;
            uint256 withoutTrainerMask;

            if ((generatedMask & playerMask) > 0) {
                // generated mask includes player
                withTrainerMask = generatedMask;
                withoutTrainerMask = generatedMask & ~playerMask;
            } else {
                //generated mask does not include player
                withoutTrainerMask = generatedMask;
                withTrainerMask = generatedMask | playerMask;
            }

            int256 withResult = getResult(roundId, withTrainerMask);
            int256 withoutResult = getResult(roundId, withoutTrainerMask);
            int256 contribution = withResult - withoutResult;
            uint256 w = weight(numberOfPlayers, popcount(withoutTrainerMask));

            // console.log("    withTrainerMask", withTrainerMask);
            // console.log("    withoutTrainerMask", withoutTrainerMask);
            // console.log("    weight", w);
            console.log("    gasleft_after_iteration", gasleft());

            // if (gasleft() < 50000) {
            //     revert("Gas limit exceeded");
            // }

            weightedSum += contribution * int256(w);
            weightTotal += w;
        }

        // console.log("calcShapley done");
        // console.log("  weightTotal", weightTotal);
        return weightTotal == 0 ? int256(0) : weightedSum / int256(weightTotal);
    }

    function weight(uint256 n, uint256 s) internal view returns (uint256 w) {
        require(n > 0, "weight: n == 0");
        require(s < n, "weight: s >= n"); // s must be in [0, n-1]

        // symmetry: C(n-1, s) == C(n-1, (n-1)-s)
        uint256 t = s;
        uint256 half = (n - 1) / 2;
        if (t > half) t = (n - 1) - t;

        w = (10 ** DECIMALS) / n; // α0
        for (uint256 k = 0; k < t; ++k) {
            w = (w * (k + 1)) / ((n - 1) - k);
        }
    }

    function popcount(uint256 x) internal pure returns (uint256 c) {
        unchecked {
            while (x != 0) {
                x &= x - 1;
                ++c;
            }
        }
    }

    function getMask(
        uint256 roundId,
        uint256 i,
        uint8 numberOfPlayers
    ) external view virtual returns (uint256) {
        return _getMask(roundId, i, numberOfPlayers);
    }

    function _getMask(
        uint256 roundId,
        uint256 i,
        uint8 numberOfPlayers
    ) internal view virtual returns (uint256) {
        // Switch the active sampler here when profiling.
        // return RandPerm.rand(keccak256(abi.encodePacked(address(this), roundId)), i, 1 << numberOfPlayers);
        // return _get_mask_monte_carlo(roundId, i, numberOfPlayers);
        // return _get_mask_stratified(roundId, i, numberOfPlayers);
        return _get_mask_stratified_monte_carlo(roundId, i, numberOfPlayers);
    }

    /// @dev Monte Carlo mask sampler matching the Python study's rule:
    ///      sample uniformly from [0, 2^n) with replacement.
    ///      Seed derivation mirrors the deterministic contract sampler.
    function _get_mask_monte_carlo(
        uint256 roundId,
        uint256 i,
        uint8 numberOfPlayers
    ) internal view virtual returns (uint256) {
        uint256 totalMasks = uint256(1) << numberOfPlayers;
        (uint256 mask, ) = RNG.rand(
            keccak256(abi.encodePacked(address(this), roundId)),
            i,
            totalMasks
        );
        return mask;
    }

    /// @dev Stratified mask sampler matching the Python contract-parity helper:
    ///      allocate the emitted budget across sizes 0..n-1 proportionally to
    ///      Shapley weights, sample a permutation within each stratum, then
    ///      unrank the chosen subset index into a coalition mask.
    function _get_mask_stratified(
        uint256 roundId,
        uint256 i,
        uint8 numberOfPlayers
    ) internal view virtual returns (uint256) {
        uint256 emittedBudget = _get_stratified_emitted_budget(
            roundId,
            numberOfPlayers
        );
        require(i < emittedBudget, "stratified sampleId out of range");

        uint256[] memory allocations = _allocate_all_strata_proportional(
            numberOfPlayers,
            emittedBudget
        );
        uint256 offset = 0;

        for (uint256 size = 0; size < numberOfPlayers; ++size) {
            uint256 sampleCount = allocations[size];
            if (i < offset + sampleCount) {
                uint256 localIndex = i - offset;
                uint256 totalOfSize = _comb(numberOfPlayers, size);
                bytes32 stratumSeed = keccak256(
                    abi.encodePacked(
                        address(this),
                        roundId,
                        numberOfPlayers,
                        size
                    )
                );
                uint256 rank = RandPerm.rand(
                    stratumSeed,
                    localIndex,
                    totalOfSize
                );
                return _unrank_combination(numberOfPlayers, size, rank);
            }
            offset += sampleCount;
        }

        revert("stratified sampleId out of range");
    }

    /// @dev Stratified sampler with replacement inside each stratum.
    ///      Uses the same proportional size allocation as `_get_mask_stratified`
    ///      but draws the local rank with RNG, so duplicates are allowed.
    function _get_mask_stratified_monte_carlo(
        uint256 roundId,
        uint256 i,
        uint8 numberOfPlayers
    ) internal view virtual returns (uint256) {
        uint256 requestedBudget = _getNumSamples(roundId);
        require(i < requestedBudget, "stratified_mc sampleId out of range");

        uint256[]
            memory allocations = _allocate_all_strata_proportional_with_replacement(
                numberOfPlayers,
                requestedBudget
            );
        uint256 offset = 0;

        for (uint256 size = 0; size < numberOfPlayers; ++size) {
            uint256 sampleCount = allocations[size];
            if (i < offset + sampleCount) {
                uint256 localIndex = i - offset;
                uint256 totalOfSize = _comb(numberOfPlayers, size);
                bytes32 stratumSeed = keccak256(
                    abi.encodePacked(
                        address(this),
                        roundId,
                        numberOfPlayers,
                        size
                    )
                );
                (uint256 rank, ) = RNG.rand(
                    stratumSeed,
                    localIndex,
                    totalOfSize
                );
                return _unrank_combination(numberOfPlayers, size, rank);
            }
            offset += sampleCount;
        }

        revert("stratified_mc sampleId out of range");
    }

    function _get_stratified_emitted_budget(
        uint256 roundId,
        uint8 numberOfPlayers
    ) internal view returns (uint256) {
        uint256 totalMasks = uint256(1) << numberOfPlayers;
        uint256 maxBudget = totalMasks - 1;
        uint256 requestedBudget = _getNumSamples(roundId);
        return requestedBudget < maxBudget ? requestedBudget : maxBudget;
    }

    function _allocate_all_strata_proportional(
        uint8 numberOfPlayers,
        uint256 budget
    ) internal pure returns (uint256[] memory allocations) {
        allocations = new uint256[](numberOfPlayers);
        if (budget == 0) {
            return allocations;
        }

        uint256[] memory remainingCapacities = new uint256[](numberOfPlayers);
        uint256[] memory weights = new uint256[](numberOfPlayers);

        for (uint256 size = 0; size < numberOfPlayers; ++size) {
            remainingCapacities[size] = _comb(numberOfPlayers, size);
            weights[size] = _stratum_weight(numberOfPlayers, size);
        }

        uint256 remainingBudget = budget;
        while (remainingBudget > 0) {
            uint256 totalWeight = 0;
            for (uint256 size = 0; size < numberOfPlayers; ++size) {
                if (remainingCapacities[size] > 0) {
                    totalWeight += weights[size];
                }
            }

            if (totalWeight == 0) {
                for (uint256 size = 0; size < numberOfPlayers; ++size) {
                    if (remainingBudget == 0) {
                        break;
                    }

                    uint256 capacity = remainingCapacities[size];
                    if (capacity == 0) {
                        continue;
                    }

                    uint256 grant = remainingBudget < capacity
                        ? remainingBudget
                        : capacity;
                    allocations[size] += grant;
                    remainingCapacities[size] -= grant;
                    remainingBudget -= grant;
                }
                break;
            }

            bool saturatedAny = false;
            uint256 budgetSnapshot = remainingBudget;
            uint256 saturatedBudget = 0;

            for (uint256 size = 0; size < numberOfPlayers; ++size) {
                uint256 capacity = remainingCapacities[size];
                if (capacity == 0) {
                    continue;
                }

                uint256 quotaFloor = (budgetSnapshot * weights[size]) /
                    totalWeight;
                if (quotaFloor >= capacity) {
                    allocations[size] += capacity;
                    remainingCapacities[size] = 0;
                    saturatedBudget += capacity;
                    saturatedAny = true;
                }
            }

            if (saturatedAny) {
                remainingBudget = budgetSnapshot - saturatedBudget;
                continue;
            }

            uint256[] memory remainders = new uint256[](numberOfPlayers);
            uint256 distributed = 0;

            for (uint256 size = 0; size < numberOfPlayers; ++size) {
                uint256 capacity = remainingCapacities[size];
                if (capacity == 0) {
                    continue;
                }

                uint256 numerator = budgetSnapshot * weights[size];
                uint256 share = numerator / totalWeight;
                allocations[size] += share;
                remainingCapacities[size] -= share;
                distributed += share;
                remainders[size] = numerator % totalWeight;
            }

            uint256 leftover = budgetSnapshot - distributed;
            while (leftover > 0) {
                bool found = false;
                uint256 bestSize = 0;
                uint256 bestRemainder = 0;

                for (uint256 size = 0; size < numberOfPlayers; ++size) {
                    if (remainingCapacities[size] == 0) {
                        continue;
                    }

                    if (
                        !found ||
                        remainders[size] > bestRemainder ||
                        (remainders[size] == bestRemainder && size < bestSize)
                    ) {
                        found = true;
                        bestSize = size;
                        bestRemainder = remainders[size];
                    }
                }

                require(found, "No stratum capacity left for remaining budget");

                allocations[bestSize] += 1;
                remainingCapacities[bestSize] -= 1;
                remainders[bestSize] = 0;
                leftover -= 1;
            }

            remainingBudget = 0;
        }
    }

    function _allocate_all_strata_proportional_with_replacement(
        uint8 numberOfPlayers,
        uint256 budget
    ) internal pure returns (uint256[] memory allocations) {
        allocations = new uint256[](numberOfPlayers);
        if (budget == 0) {
            return allocations;
        }

        uint256[] memory remainders = new uint256[](numberOfPlayers);
        uint256 totalWeight = 0;
        for (uint256 size = 0; size < numberOfPlayers; ++size) {
            totalWeight += _stratum_weight(numberOfPlayers, size);
        }

        uint256 distributed = 0;
        for (uint256 size = 0; size < numberOfPlayers; ++size) {
            uint256 numerator = budget * _stratum_weight(numberOfPlayers, size);
            allocations[size] = numerator / totalWeight;
            remainders[size] = numerator % totalWeight;
            distributed += allocations[size];
        }

        uint256 leftover = budget - distributed;
        while (leftover > 0) {
            uint256 bestSize = 0;
            uint256 bestRemainder = 0;

            for (uint256 size = 0; size < numberOfPlayers; ++size) {
                if (
                    remainders[size] > bestRemainder ||
                    (remainders[size] == bestRemainder && size < bestSize)
                ) {
                    bestSize = size;
                    bestRemainder = remainders[size];
                }
            }

            allocations[bestSize] += 1;
            remainders[bestSize] = 0;
            leftover -= 1;
        }
    }

    function _stratum_weight(
        uint8 numberOfPlayers,
        uint256 coalitionSize
    ) internal pure returns (uint256) {
        if (numberOfPlayers <= 1) {
            return STRATIFIED_WEIGHT_PRECISION;
        }

        uint256 denominator = _comb(numberOfPlayers - 1, coalitionSize);
        uint256 w = STRATIFIED_WEIGHT_PRECISION / denominator;
        return w > 0 ? w : 1;
    }

    function _unrank_combination(
        uint8 numberOfPlayers,
        uint256 coalitionSize,
        uint256 rank
    ) internal pure returns (uint256 mask) {
        if (coalitionSize == 0) {
            return 0;
        }

        uint256 remaining = rank;
        uint256 start = 0;

        for (
            uint256 positionsLeft = coalitionSize;
            positionsLeft > 0;
            --positionsLeft
        ) {
            for (
                uint256 candidate = start;
                candidate <= numberOfPlayers - positionsLeft;
                ++candidate
            ) {
                uint256 count = _comb(
                    numberOfPlayers - candidate - 1,
                    positionsLeft - 1
                );
                if (remaining < count) {
                    mask |= uint256(1) << candidate;
                    start = candidate + 1;
                    break;
                }
                remaining -= count;
            }
        }
    }

    function _comb(
        uint256 n,
        uint256 k
    ) internal pure returns (uint256 result) {
        if (k > n) {
            return 0;
        }
        if (k == 0 || k == n) {
            return 1;
        }

        uint256 t = k;
        if (t > n - t) {
            t = n - t;
        }

        result = 1;
        for (uint256 i = 1; i <= t; ++i) {
            result = (result * (n - t + i)) / i;
        }
    }

    function _getNumSamples(uint256 roundId) internal view returns (uint256) {
        return _numSamples[roundId];
    }

    function _setNumSamples(uint256 roundId, uint256 numSamples) internal {
        _numSamples[roundId] = numSamples;
        emit NumSamplesSet(roundId, numSamples);
    }

    /// @dev See {IERC165-supportsInterface}
    function supportsInterface(
        bytes4 interfaceId
    ) public view virtual override returns (bool) {
        return EvaluationStorage.supportsInterface(interfaceId);
    }
}
