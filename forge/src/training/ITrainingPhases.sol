// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface ITrainingPhases {
    function getCurrentPhase() external returns (bytes32);
    function isTraining() external view returns (bool);
    function isEvaluation() external view returns (bool);
}
