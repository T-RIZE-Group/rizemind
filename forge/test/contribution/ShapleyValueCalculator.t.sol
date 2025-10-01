// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test} from "forge-std/Test.sol";
import {ShapleyValueCalculator} from "../../src/contribution/ShapleyValueCalculator.sol";
import {ERC1967Proxy} from "@openzeppelin-contracts-5.2.0/proxy/ERC1967/ERC1967Proxy.sol";

/**
 * @title MockShapleyValueCalculator
 * @dev Mock contract that exposes internal functions for testing
 */
contract MockShapleyValueCalculator is ShapleyValueCalculator {
    function initialize() external initializer {
        __EvaluationStorage_init();
    }

    // Expose internal functions for testing
    function exposed_getTotalEvaluations(uint256 roundId, uint8 numberOfPlayers) external view returns (uint256) {
        return _getTotalEvaluations(roundId, numberOfPlayers);
    }

    function exposed_getEvaluationsRequired(uint256 roundId, uint8 numberOfPlayers) external view returns (uint256) {
        return _getEvaluationsRequired(roundId, numberOfPlayers);
    }

    function exposed_registerResult(
        uint256 roundId,
        uint256 sampleId,
        uint256 setId,
        bytes32 modelHash,
        int256 result,
        uint8 numberOfPlayers
    ) external {
        _registerResult(roundId, sampleId, setId, modelHash, result, numberOfPlayers);
    }

    function exposed_calcShapley(
        uint256 roundId,
        uint256 trainerIndex,
        uint8 numberOfPlayers
    ) external view returns (int256) {
        return _calcShapley(roundId, trainerIndex, numberOfPlayers);
    }

    function exposed_weight(uint256 n, uint256 s) external view returns (uint256) {
        return weight(n, s);
    }

    function exposed_popcount(uint256 x) external pure returns (uint256) {
        return popcount(x);
    }

    function exposed_setNumSamples(uint256 roundId, uint256 numSamples) external {
        _setNumSamples(roundId, numSamples);
    }

    /**
     * @dev Override the _getMask function to return deterministic values for tests instead of using randomness
     */
    function _getMask(
        uint256 /* roundId */,
        uint256 i,
        uint8 numberOfPlayers
    ) internal view override returns (uint256) {
        // For testing, return deterministic values based on sample index
        // This ensures we can predict the coalition sets
        return i % (1 << numberOfPlayers);
    }
}

contract ShapleyValueCalculatorTest is Test {
    MockShapleyValueCalculator public implementation;
    MockShapleyValueCalculator public calculator;
    address public admin;
    address public user;

    function setUp() public {
        admin = makeAddr("admin");
        user = makeAddr("user");
        
        // Deploy implementation
        implementation = new MockShapleyValueCalculator();
        
        // Deploy proxy
        bytes memory initData = abi.encodeWithSelector(
            MockShapleyValueCalculator.initialize.selector
        );
        
        ERC1967Proxy proxy = new ERC1967Proxy(
            address(implementation),
            initData
        );
        
        calculator = MockShapleyValueCalculator(address(proxy));
    }

    function test_initialize() public view {
        // Test that the contract initializes properly
        // Since ShapleyValueCalculator doesn't have access control,
        // we just verify it's deployed and initialized
        assertTrue(address(calculator) != address(0));
    }

    function test_basicShapleyCalculation() public {
        uint256 roundId = 1;
        uint8 numberOfPlayers = 2;
        uint256 numSamples = 4;
        
        // Set number of samples for the round
        calculator.exposed_setNumSamples(roundId, numSamples);
        
        // Verify numSamples was set
        assertEq(calculator.exposed_getEvaluationsRequired(roundId, numberOfPlayers), numSamples);
        
        // Verify total evaluations calculation
        uint256 totalEvaluations = calculator.exposed_getTotalEvaluations(roundId, numberOfPlayers);
        assertEq(totalEvaluations, 1 << numberOfPlayers); // Should be 2^2 = 4
        
        // Register some evaluation results
        bytes32 modelHash = keccak256("test_model");
        
        // Get the target set IDs for each sample
        uint256 targetSetId0 = calculator.getMask(roundId, 0, numberOfPlayers);
        uint256 targetSetId1 = calculator.getMask(roundId, 1, numberOfPlayers);
        uint256 targetSetId2 = calculator.getMask(roundId, 2, numberOfPlayers);
        uint256 targetSetId3 = calculator.getMask(roundId, 3, numberOfPlayers);
        
        // Register results for different coalitions using the target set IDs
        calculator.exposed_registerResult(roundId, 0, targetSetId0, modelHash, 0, numberOfPlayers);
        calculator.exposed_registerResult(roundId, 1, targetSetId1, modelHash, 300, numberOfPlayers);
        calculator.exposed_registerResult(roundId, 2, targetSetId2, modelHash, 600, numberOfPlayers);
        calculator.exposed_registerResult(roundId, 3, targetSetId3, modelHash, 1500, numberOfPlayers);
        
        // Calculate Shapley values for both players
        int256 shapleyValue0 = calculator.exposed_calcShapley(roundId, 0, numberOfPlayers);
        int256 shapleyValue1 = calculator.exposed_calcShapley(roundId, 1, numberOfPlayers);
        
        // Verify the Shapley values are calculated correctly
        // Based on the deterministic mask function, we expect:
        // Player 0: 600 (from the logs)
        // Player 1: 900 (from the logs)
        assertEq(shapleyValue0, 600);
        assertEq(shapleyValue1, 900);
    }

    function test_registerResult_withValidDistance() public {
        uint256 roundId = 1;
        uint8 numberOfPlayers = 2;
        bytes32 modelHash = keccak256("test_model");
        
        // Set up a target set ID
        uint256 targetSetId = calculator.getMask(roundId, 0, numberOfPlayers);
        
        // Register result with exact target set ID (distance = 0)
        calculator.exposed_registerResult(roundId, 0, targetSetId, modelHash, 100, numberOfPlayers);
        
        // Register result with hamming distance = 1
        uint256 nearbySetId = targetSetId ^ 1; // Flip one bit
        calculator.exposed_registerResult(roundId, 0, nearbySetId, modelHash, 150, numberOfPlayers);
        
        // Verify results were stored
        assertEq(calculator.getResult(roundId, targetSetId), 100);
        assertEq(calculator.getResult(roundId, nearbySetId), 150);
    }

    function test_registerResult_withInvalidDistance() public {
        uint256 roundId = 1;
        uint8 numberOfPlayers = 2;
        bytes32 modelHash = keccak256("test_model");
        
        // Set up a target set ID
        uint256 targetSetId = calculator.getMask(roundId, 0, numberOfPlayers);
        
        // Try to register result with hamming distance > 1 (should revert)
        uint256 farSetId = targetSetId ^ 3; // Flip two bits (distance = 2)
        
        vm.expectRevert(
            abi.encodeWithSelector(
                ShapleyValueCalculator.setIdTooFar.selector,
                roundId,
                farSetId,
                targetSetId
            )
        );
        
        calculator.exposed_registerResult(roundId, 0, farSetId, modelHash, 200, numberOfPlayers);
    }
}
