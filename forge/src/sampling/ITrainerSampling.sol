// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title Interface for Trainer Sampling
/// @notice Defines the interface for checking if a trainer is selected in a sampling process
interface ITrainerSelection{
    /// @dev Checks if a specific trainer address is selected
    /// @param addr The address of the trainer to check
    /// @return True if the trainer is selected, false otherwise
    function isTrainerSelected(address addr, uint256 roundId) external view returns (bool);
}
