// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test} from "forge-std/Test.sol";
import {RNG} from "../../src/randomness/RNG.sol";

contract RNGTest is Test {
    // Test data
    bytes32 constant TEST_SEED = keccak256("test_seed");
    bytes32 constant ANOTHER_SEED = keccak256("another_seed");
    
    function setUp() public {}

    // Test the custom error
    function test_revertWhenMaxIsZero() public {
        vm.expectRevert(RNG.MaxCannotBeZero.selector);
        RNG.rand(TEST_SEED, 0, 0);
    }

    // Fuzz test for basic functionality with various max values
    function test_randFuzz(uint256 max, uint256 startIndex) public {
        // Skip invalid inputs
        vm.assume(max > 0);
        
        (uint256 result,) = RNG.rand(TEST_SEED, startIndex, max);
        
        // Assert result is in valid range [0, max)
        assertTrue(result < max, "Result should be less than max");
        assertTrue(result >= 0, "Result should be greater than or equal to 0");
    }

    // Fuzz test for edge cases with very large max values
    function test_randFuzzLargeMax(uint256 startIndex) public {
        // Test with max values close to uint256 max
        uint256 max = type(uint256).max - startIndex % 1000; // Ensure max > startIndex
        vm.assume(max > 0);
        
        (uint256 result, ) = RNG.rand(TEST_SEED, startIndex, max);
        
        assertTrue(result < max, "Result should be less than max");
        assertTrue(result >= 0, "Result should be greater than or equal to 0");
    }

    // Fuzz test for different seeds
    function test_randFuzzDifferentSeeds() public {
        uint256 max = type(uint256).max - 1;
        bytes32 seed1 = keccak256("seed1");
        bytes32 seed2 = keccak256("seed2");
        
        (uint256 result1, ) = RNG.rand(seed1, 0, max);
        (uint256 result2, ) = RNG.rand(seed2, 1, max);
        
        assertNotEq(result1, result2, "Results should be different with different seeds");
    }

    function test_randFuzzDifferentIndex() public {
        bytes32 seed = keccak256("seed");
        uint256 max = type(uint256).max - 1;
        uint256 index1 = 0;
        uint256 index2 = 1;
        
        (uint256 result1, ) = RNG.rand(seed, index1, max);
        (uint256 result2, ) = RNG.rand(seed, index2, max);
        
        assertNotEq(result1, result2, "Results should be different with different indices");
    }

    // Fuzz test for deterministic behavior
    function test_randFuzzDeterministic(uint256 max, uint256 startIndex) public {
        vm.assume(max > 0);
        vm.assume(max < type(uint256).max);
        
        (uint256 result1, uint256 newIndex1) = RNG.rand(TEST_SEED, startIndex, max);
        (uint256 result2, uint256 newIndex2) = RNG.rand(TEST_SEED, startIndex, max);
        
        // Same inputs should produce identical results
        assertEq(result1, result2, "Results should be identical for same parameters");
        assertEq(newIndex1, newIndex2, "Indices should be identical for same parameters");
    }

    // Fuzz test for statistical properties - check average is close to max/2
    function test_randFuzzStatistical(uint256 max) public {
        vm.assume(max > 10);
        vm.assume(max < 1000000);
        
        uint8 DECIMALS = 18;
        uint256 numSamples = 100;

        uint256 sum = 0;
        // Generate multiple samples
        for (uint256 i = 0; i < numSamples; i++) {
            (uint256 result, ) = RNG.rand(TEST_SEED, i, max);
            sum += result;
        }

        
        uint256 average = sum * 10 **DECIMALS / numSamples;
        
        // For uniform distribution, average should be close to max/2
        // Allow some tolerance due to finite sample size
        uint256 expectedAverage = max * 10 **DECIMALS / 2;
        uint256 tolerance = max * 10 **DECIMALS / 10; // 10% tolerance
        
        assertTrue(
            average >= expectedAverage - tolerance && average <= expectedAverage + tolerance,
            "Average should be close to max/2 for uniform distribution"
        );
    }

    // Fuzz test for boundary conditions
    function test_randFuzzBoundaries(uint256 startIndex) public {
        // Test with max = 1 (should always return 0)
        (uint256 result, uint256 newIndex) = RNG.rand(TEST_SEED, startIndex, 1);
        assertEq(result, 0, "Result should be 0 when max is 1");
        assertEq(newIndex, startIndex, "Index should be preserved");
        
        // Test with max = 2 (should return 0 or 1)
        (result, newIndex) = RNG.rand(TEST_SEED, startIndex, 2);
        assertTrue(result < 2, "Result should be less than 2");
        assertTrue(result >= 0, "Result should be greater than or equal to 0");
        assertEq(newIndex, startIndex, "Index should be preserved");
    }

    // Test that the function handles cases where keccak256 might produce values >= limit
    function test_randHandlesHashCollisions() public {
        // This test verifies that the function can handle cases where
        // keccak256 produces values that need to be rejected
        (uint256 result, uint256 newI) = RNG.rand(TEST_SEED, 0, 100);
        
        assertTrue(result < 100, "Result should be less than 100");
        assertTrue(result >= 0, "Result should be greater than or equal to 0");
        assertTrue(newI >= 0, "New index should be non-negative");
    }

    // Fuzz test for specific edge case combinations
    function test_randFuzzEdgeCases(
        uint256 max,
        uint256 startIndex,
        bytes32 seed
    ) public {
        vm.assume(max > 0);
        vm.assume(max < type(uint256).max);
        
        // Test with various seed/index combinations
        (uint256 result, uint256 newIndex) = RNG.rand(seed, startIndex, max);
        
        assertTrue(result < max, "Result should be less than max");
        assertTrue(result >= 0, "Result should be greater than or equal to 0");
    }
}
