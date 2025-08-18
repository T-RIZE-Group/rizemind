// SPDX-License-Identifier: MIT
// aderyn-fp-next-line(push-zero-opcode)
pragma solidity ^0.8.20;

/// @title Interface for Random Number Generation with deterministic permutations
/// @notice Defines the interface for generating deterministic random permutations
interface IRNG {
    /// @dev Returns the i-th pseudorandom element of a seed-keyed permutation of [0, max)
    /// @param seed Arbitrary 32-byte seed (fix it to "freeze" the permutation)
    /// @param i Index in [0, max)
    /// @param max Size of the domain (> 0). Outputs are in [0, max).
    /// @return A unique value in [0, max) for the given seed and index
    function rand(bytes32 seed, uint256 i, uint256 max) external pure returns (uint256);
}
