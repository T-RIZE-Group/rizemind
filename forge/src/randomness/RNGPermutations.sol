// SPDX-License-Identifier: MIT
// aderyn-fp-next-line(push-zero-opcode)
pragma solidity ^0.8.20;


/// @title Stateless seeded permutation over [0, max)
/// @notice rand(seed, i, max) returns a unique value in [0, max) for each i in [0, max)
library RandPerm {
    uint256 internal constant ROUNDS = 4; // 4 Feistel rounds are usually plenty for mixing

    /// @dev Returns the i-th pseudorandom element of a seed-keyed permutation of [0, max)
    /// @param seed Arbitrary 32-byte seed (fix it to “freeze” the permutation)
    /// @param i Index in [0, max)
    /// @param max Size of the domain (> 0). Outputs are in [0, max).
    function rand(bytes32 seed, uint256 i, uint256 max) internal pure returns (uint256) {
        require(max > 0, "RandPerm: max=0");
        require(i < max, "RandPerm: i>=max");

        // Bit length n = ceil(log2(max)); use balanced Feistel on 2^n domain
        uint256 nBits = _bitLen(max - 1);
        if (nBits & 1 == 1) nBits += 1; // make it even for balanced halves

        uint256 x = i;
        unchecked {
            while (true) {
                x = _feistel(x, nBits, seed);
                if (x < max) return x;       // format-preserving via cycle-walking
                // else keep walking within the larger 2^n domain
            }
        }
    }

    // === internals ===

    /// @dev One application of a balanced Feistel network over nBits (even), keyed by seed.
    function _feistel(uint256 x, uint256 nBits, bytes32 seed) private pure returns (uint256) {
        // Split into two equal halves
        uint256 half = nBits >> 1;                          // nBits/2
        uint256 mask = (uint256(1) << half) - 1;            // half-bit mask

        uint256 R = x & mask;                               // lower half
        uint256 L = x >> half;                              // upper half

        // Standard Feistel: (L,R) -> (R, L ^ F_k(R))
        // Do ROUNDS rounds with keccak as the round function
        for (uint256 r = 0; r < ROUNDS; r++) {
            // Round function: hash(seed, r, R) and mask to half bits
            uint256 F = uint256(keccak256(abi.encodePacked(seed, r, R))) & mask;
            uint256 newL = R;
            uint256 newR = (L ^ F) & mask;
            L = newL;
            R = newR;
        }

        // Recombine (note: even number of rounds ⇒ no final swap needed)
        return (L << half) | R;
    }

    /// @dev Number of bits to represent x (i.e., floor(log2(x)) + 1), with bitLen(0) = 0.
    function _bitLen(uint256 x) private pure returns (uint256 n) {
        while (x != 0) {
            x >>= 1;
            n++;
        }
    }
}
