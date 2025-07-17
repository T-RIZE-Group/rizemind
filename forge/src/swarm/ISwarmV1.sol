// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ITraining} from "../training/ITraining.sol";

interface ISwarmV1 is ITraining {
    function canTrain(address trainer, uint256 roundId) external returns (bool);
}
