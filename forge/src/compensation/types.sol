// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import { IERC5267 } from "@openzeppelin-contracts-5.2.0/interfaces/IERC5267.sol";

event CompensationSent(address indexed receiver, uint256 amount);

interface ICompensation is IERC5267 {
    function distribute(uint256 roundId, address[] memory recipients, uint64[] memory contributions) external;
}