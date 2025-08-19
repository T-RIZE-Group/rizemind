// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title Interface for Seed Provider
/// @notice Defines the interface for providing seeds to the RNG system
interface ISeedProvider {

    /// @dev Returns a seed for the RNG system with default context
    /// @param roundId The round ID to get a seed for
    /// @return A 32-byte seed for the RNG system
    function getSeed(uint256 roundId) external view returns (bytes32);
}
