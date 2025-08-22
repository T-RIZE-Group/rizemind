// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC5267} from "@openzeppelin-contracts-5.2.0/interfaces/IERC5267.sol";

/// @title Interface for Trainer Sampling
/// @notice Defines the interface for checking if a trainer is selected in a sampling process
interface ISelector is IERC5267{
    /// @dev Checks if a specific trainer address is selected
    /// @param addr The address of the trainer to check
    /// @return True if the trainer is selected, false otherwise
    function isSelected(address addr, uint256 roundId) external view returns (bool);
}
