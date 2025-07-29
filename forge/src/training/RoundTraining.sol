// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import {Initializable} from "@openzeppelin-contracts-upgradeable-5.2.0/proxy/utils/Initializable.sol";
import {IERC165} from "@openzeppelin-contracts-5.2.0/utils/introspection/IERC165.sol";
import {ITraining, RoundSummary, RoundFinished} from "./ITraining.sol";

contract RoundTraining is ITraining, Initializable, IERC165 {
    uint256 private _round;

    error RoundMismatch(uint256 currentRound, uint256 givenRound);

    function __RoundTraining_init() internal onlyInitializing {
        _round = 1;
    }

    function currentRound() public view returns (uint256) {
        return _round;
    }

    function nextRound(RoundSummary calldata summary) external {
        uint256 round = _round;
        if (round != summary.roundId) {
            revert RoundMismatch(round, summary.roundId);
        }
        _round++;
        emit RoundFinished(
            summary.roundId,
            summary.nTrainers,
            summary.modelScore,
            summary.totalContributions
        );
    }

    function supportsInterface(
        bytes4 interfaceId
    ) public view virtual override returns (bool) {
        return
            interfaceId == this.currentRound.selector ||
            interfaceId == this.nextRound.selector;
    }
}
