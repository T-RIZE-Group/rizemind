// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import {IERC5267} from "@openzeppelin-contracts-5.2.0/interfaces/IERC5267.sol";

struct RoundSummary {
    uint256 roundId;
    uint64 nTrainers;
    uint64 modelScore;
    uint128 totalContributions;
}

interface IModelRegistry is IERC5267 {
    function canTrain(address trainer, uint256 roundId) external returns (bool);

    function curentRound() external view returns (uint256);

    function nextRound(RoundSummary calldata summary) external;
}
