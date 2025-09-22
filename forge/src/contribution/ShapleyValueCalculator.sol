// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {EvaluationStorage} from "./EvaluationStorage.sol";
import {RandPerm} from "../randomness/RNGPermutations.sol";
import {IERC165} from "@openzeppelin-contracts-5.2.0/utils/introspection/IERC165.sol";
import {IEvaluationStorage} from "./types.sol";
import {console} from "forge-std/console.sol";

contract ShapleyValueCalculator is EvaluationStorage {
    // we index num. samples per round to make historical querying of past round accurate
    mapping(uint256 => uint256) private _numSamples;
    uint8 private constant DECIMALS = 18;

    event NumSamplesSet(uint256 indexed roundId, uint256 numSamples);

    error setIdTooFar(uint256 roundId, uint256 setId, uint256 targetSetId);

    function _getTotalEvaluations(uint256, uint8 numberOfPlayers) internal view returns (uint256) {
        return 1 << numberOfPlayers;
    }

    function _getEvaluationsRequired(uint256 roundId, uint8) internal view returns (uint256) {
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
    ) internal virtual view returns (int256) {
        int256 weightedSum = 0;
        uint256 weightTotal = 0;
        uint256 numSamples = _getNumSamples(roundId);
        for (uint256 i = 0; i < numSamples; ++i) {
            uint256 generatedMask = _getMask(roundId, i, numberOfPlayers);
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
            weightedSum += contribution * int256(w);
            weightTotal += w;
        }
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
        unchecked { while (x != 0) { x &= x - 1; ++c; } }
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
        return
            RandPerm.rand(
                keccak256(abi.encodePacked(address(this), roundId)),
                i,
                1 << numberOfPlayers
            );
    }


    function _getNumSamples(uint256 roundId) internal view returns (uint256) {
        return _numSamples[roundId];
    }

    function _setNumSamples(uint256 roundId, uint256 numSamples) internal {
        _numSamples[roundId] = numSamples;
        emit NumSamplesSet(roundId, numSamples);
    }

    /// @dev See {IERC165-supportsInterface}
    function supportsInterface(bytes4 interfaceId) public view override virtual returns (bool) {
        return EvaluationStorage.supportsInterface(interfaceId);
    }
}
