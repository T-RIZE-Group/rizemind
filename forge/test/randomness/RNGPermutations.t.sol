// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test} from "forge-std/Test.sol";
import {RandPerm} from "../../src/randomness/RNGPermutations.sol";

// Wrapper contract to test library functions
contract RandPermWrapper {
    function rand(bytes32 seed, uint256 i, uint256 max) external pure returns (uint256) {
        return RandPerm.rand(seed, i, max);
    }
}

contract RandPermTest is Test {
    RandPermWrapper wrapper;
    bytes32 constant TEST_SEED = keccak256("test_seed");
    bytes32 constant ANOTHER_SEED = keccak256("another_seed");

    function setUp() public {
        wrapper = new RandPermWrapper();
    }

    function test_RandPerm_BasicFunctionality() public {
        uint256 max = 100;
        uint256 i = 42;
        
        uint256 result = wrapper.rand(TEST_SEED, i, max);
        
        assertTrue(result < max, "Result should be less than max");
        assertTrue(result >= 0, "Result should be non-negative");
    }

    function test_RandPerm_EdgeCases() public {
        // Test with max = 1
        uint256 result1 = wrapper.rand(TEST_SEED, 0, 1);
        assertEq(result1, 0, "With max=1, should always return 0");

        // Test with max = 2
        uint256 result2 = wrapper.rand(TEST_SEED, 0, 2);
        uint256 result3 = wrapper.rand(TEST_SEED, 1, 2);
        assertTrue(result2 < 2, "Result should be less than 2");
        assertTrue(result3 < 2, "Result should be less than 2");
        assertTrue(result2 != result3, "Different indices should produce different results");
    }

    function test_RandPerm_Reversibility() public {
        uint256 max = 100;
        
        // Test that the same seed and index always produces the same result
        uint256 result1 = wrapper.rand(TEST_SEED, 42, max);
        uint256 result2 = wrapper.rand(TEST_SEED, 42, max);
        assertEq(result1, result2, "Same seed and index should produce same result");
    }

    function test_RandPerm_SeedDependency() public {
        uint256 max = 100;
        uint256 i = 42;
        
        uint256 result1 = wrapper.rand(TEST_SEED, i, max);
        uint256 result2 = wrapper.rand(ANOTHER_SEED, i, max);
        
        // Different seeds should produce different results (with high probability)
        // Note: This is probabilistic, but the probability of collision is extremely low
        assertTrue(result1 != result2, "Different seeds should produce different results");
    }

    function test_RandPerm_IndexDependency() public {
        uint256 max = 100;
        
        uint256 result1 = wrapper.rand(TEST_SEED, 0, max);
        uint256 result2 = wrapper.rand(TEST_SEED, 1, max);
        uint256 result3 = wrapper.rand(TEST_SEED, 2, max);
        
        // Different indices should produce different results
        assertTrue(result1 != result2, "Different indices should produce different results");
        assertTrue(result1 != result3, "Different indices should produce different results");
        assertTrue(result2 != result3, "Different indices should produce different results");
    }

    function test_RandPerm_PermutationProperties() public {
        uint256 max = 10; // Small enough to test all values
        
        // Collect all results
        uint256[] memory results = new uint256[](max);
        for (uint256 i = 0; i < max; i++) {
            results[i] = wrapper.rand(TEST_SEED, i, max);
        }
        
        // Check that all results are unique (permutation property)
        for (uint256 i = 0; i < max; i++) {
            for (uint256 j = i + 1; j < max; j++) {
                assertTrue(results[i] != results[j], "Permutation should have unique values");
            }
        }
        
        // Check that all results are in range [0, max)
        for (uint256 i = 0; i < max; i++) {
            assertTrue(results[i] < max, "All results should be less than max");
            assertTrue(results[i] >= 0, "All results should be non-negative");
        }
    }

    function test_RandPerm_LargeMax() public {
        uint256 max = 1000;
        
        // Test with larger max value
        for (uint256 i = 0; i < 100; i++) {
            uint256 result = wrapper.rand(TEST_SEED, i, max);
            assertTrue(result < max, "Result should be less than max");
            assertTrue(result >= 0, "Result should be non-negative");
        }
    }

    function test_RandPerm_PowerOfTwoMax() public {
        uint256 max = 256; // 2^8
        
        // Test with power of 2 max value
        for (uint256 i = 0; i < 100; i++) {
            uint256 result = wrapper.rand(TEST_SEED, i, max);
            assertTrue(result < max, "Result should be less than max");
            assertTrue(result >= 0, "Result should be non-negative");
        }
    }

    function test_RandPerm_NonPowerOfTwoMax() public {
        uint256 max = 100; // Not a power of 2
        
        // Test with non-power of 2 max value
        for (uint256 i = 0; i < 100; i++) {
            uint256 result = wrapper.rand(TEST_SEED, i, max);
            assertTrue(result < max, "Result should be less than max");
            assertTrue(result >= 0, "Result should be non-negative");
        }
    }

    function test_RandPerm_ConsistencyAcrossRuns() public {
        uint256 max = 100;
        uint256[] memory results = new uint256[](10);
        
        // Run multiple times with same parameters
        for (uint256 run = 0; run < 5; run++) {
            for (uint256 i = 0; i < 10; i++) {
                uint256 result = wrapper.rand(TEST_SEED, i, max);
                if (run == 0) {
                    results[i] = result;
                } else {
                    assertEq(result, results[i], "Results should be consistent across runs");
                }
            }
        }
    }

    function test_RandPerm_ErrorConditions_MaxZero() public {
        // Test max = 0 (should revert)
        vm.expectRevert("RandPerm: max=0");
        wrapper.rand(TEST_SEED, 0, 0);
    }

    function test_RandPerm_ErrorConditions_IndexEqualToMax() public {
        // Test i = max (should revert)
        vm.expectRevert("RandPerm: i>=max");
        wrapper.rand(TEST_SEED, 100, 100);
    }

    function test_RandPerm_ErrorConditions_IndexGreaterThanMax() public {
        // Test i > max (should revert)
        vm.expectRevert("RandPerm: i>=max");
        wrapper.rand(TEST_SEED, 101, 100);
    }

    function test_RandPerm_BitLengthEdgeCases() public {
        // Test with very small values
        uint256 result1 = wrapper.rand(TEST_SEED, 0, 1);
        assertEq(result1, 0, "max=1 should return 0");
        
        // Test with max = 2
        uint256 result2 = wrapper.rand(TEST_SEED, 0, 2);
        uint256 result3 = wrapper.rand(TEST_SEED, 1, 2);
        assertTrue(result2 < 2 && result3 < 2, "Results should be in range [0,2)");
        assertTrue(result2 != result3, "Different indices should produce different results");
    }

    function test_RandPerm_FeistelNetwork() public {
        // Test that the Feistel network produces different outputs for different inputs
        uint256 max = 16; // 2^4, so nBits will be 4
        
        uint256 result1 = wrapper.rand(TEST_SEED, 0, max);
        uint256 result2 = wrapper.rand(TEST_SEED, 1, max);
        uint256 result3 = wrapper.rand(TEST_SEED, 2, max);
        uint256 result4 = wrapper.rand(TEST_SEED, 3, max);
        
        // All results should be different (permutation property)
        assertTrue(result1 != result2, "Feistel should produce different outputs");
        assertTrue(result1 != result3, "Feistel should produce different outputs");
        assertTrue(result1 != result4, "Feistel should produce different outputs");
        assertTrue(result2 != result3, "Feistel should produce different outputs");
        assertTrue(result2 != result4, "Feistel should produce different outputs");
        assertTrue(result3 != result4, "Feistel should produce different outputs");
    }

    function test_RandPerm_DeterministicWithSameSeed() public {
        uint256 max = 100;
        
        // Test that same seed produces same sequence
        uint256[] memory firstRun = new uint256[](10);
        uint256[] memory secondRun = new uint256[](10);
        
        for (uint256 i = 0; i < 10; i++) {
            firstRun[i] = wrapper.rand(TEST_SEED, i, max);
            secondRun[i] = wrapper.rand(TEST_SEED, i, max);
            assertEq(firstRun[i], secondRun[i], "Same seed should produce same sequence");
        }
    }
}
