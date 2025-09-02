// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import { IERC5267 } from "@openzeppelin-contracts-5.2.0/interfaces/IERC5267.sol";

interface IEvaluationStorage {
    function getResult(uint256 roundId, uint256 setId) external view returns (int256);
    function getResultOrThrow(uint256 roundId, uint256 setId) external view returns (int256);
}

interface IContributionCalculator is IEvaluationStorage, IERC5267  {
    function calculateContribution(uint256 roundId, uint256 trainerIndex, uint8 numberOfTrainers) external view returns (int256);
}

event TrainerContributed(address indexed trainer, uint256 contribution);