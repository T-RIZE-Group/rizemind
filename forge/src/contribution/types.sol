// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

interface IEvaluationStorage {
    function getResult(uint256 roundId, uint256 setId) external view returns (int256);
    function getResultOrThrow(uint256 roundId, uint256 setId) external view returns (int256);
}

event TrainerContributed(address indexed trainer, uint256 contribution);