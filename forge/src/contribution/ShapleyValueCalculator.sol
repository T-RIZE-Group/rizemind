// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {EvaluationStorage} from "./EvaluationStorage.sol";
import {RandPerm} from "../randomness/RNGPermutations.sol";
import {IERC165} from "@openzeppelin-contracts-5.2.0/utils/introspection/IERC165.sol";
import {IEvaluationStorage} from "./types.sol";

// TODO: handle unevaluated sets
contract ShapleyValueCalculator is EvaluationStorage {
    uint256 private _num_samples;
    uint8 private DECIMALS = 18;

    function _calcShapley(
        uint256 roundId,
        uint256 trainerIndex,
        uint8 numberOfPlayers
    ) internal virtual view returns (int256) {
        int256 weightedSum = 0;
        uint256 weightTotal = 0;
        for (uint8 i = 0; i < _num_samples; ++i) {
            uint256 generatedMask = getMask(roundId, i, 1 << numberOfPlayers);
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

    /// @notice Scaled weight = floor(1e18 / (n * C(n-1, s)))
    function weight(uint256 n, uint256 s) internal virtual view returns (uint256 w) {
        w = (10 ** DECIMALS) / n; // α0
        // α_{k+1} = α_k * (k+1) / (n-1-k)
        for (uint256 k = 0; k < s; ++k) {
            // With n ≤ 255 this can't overflow: w ≤ 1e18 and multiply by ≤ n.
            w = (w * (k + 1)) / ((n - 1) - k);
        }
    }

    function popcount(uint256 x) internal pure returns (uint256 c) {
        unchecked { while (x != 0) { x &= x - 1; ++c; } }
    }

    function getMask(
        uint256 roundId,
        uint8 i,
        uint256 numberOfPlayers
    ) public view returns (uint256) {
        return
            RandPerm.rand(
                keccak256(abi.encodePacked(address(this), roundId)),
                i,
                1 << numberOfPlayers
            );
    }

    /// @dev See {IERC165-supportsInterface}
    function supportsInterface(bytes4 interfaceId) public view override virtual returns (bool) {
        return EvaluationStorage.supportsInterface(interfaceId);
    }
}
