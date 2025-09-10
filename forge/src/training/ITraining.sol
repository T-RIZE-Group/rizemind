// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

struct RoundSummary {
    uint256 roundId;
    uint64 nTrainers;
    uint64 modelScore;
    uint128 totalContributions;
}


event RoundFinished(
    uint256 indexed roundId,
    uint64 nTrainers,
    uint64 modelScore,
    uint128 totalContribution
);


interface ITraining {
    function currentRound() external view returns (uint256);
}
