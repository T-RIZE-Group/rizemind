// SPDX-License-Identifier: MIT
// aderyn-fp-next-line(push-zero-opcode)
pragma solidity ^0.8.20; 

library RNG {
    /// @notice Error thrown when max parameter is zero
    error MaxCannotBeZero();

    /// @notice Uniform integer in [0, upper) with no modulo bias
    /// @param seed The seed for the random number generator
    /// @param i The starting index for the random number generator
    /// @param max The maximum value for the random number
    /// @return The random number
    /// @return The new index in case a number is skipped to avoid modulo bias
    function rand(bytes32 seed, uint256 i, uint256 max) internal pure returns (uint256, uint256) {
        if (max == 0) {
            revert MaxCannotBeZero();
        }
        // Largest multiple of `max` <= 2^256-1
        uint256 limit = type(uint256).max - (type(uint256).max % max);
        while (true) {
            uint256 x = uint256(keccak256(abi.encodePacked(seed, i)));
            if (x < limit) {
                return (x % max, i);
            }
            i++;
        }
    }
}