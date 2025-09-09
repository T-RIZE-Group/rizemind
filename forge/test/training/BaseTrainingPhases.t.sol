// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test, console} from "forge-std/Test.sol";
import {ERC1967Proxy} from "@openzeppelin-contracts-5.2.0/proxy/ERC1967/ERC1967Proxy.sol";
import {BaseTrainingPhases} from "../../src/training/BaseTrainingPhases.sol";

/// @title MockBaseTrainingPhases
/// @notice Mock contract that exposes internal functions for testing
contract MockBaseTrainingPhases is BaseTrainingPhases {
    /// @notice Expose the internal _startTrainingPhase function for testing
    /// @return The new phase after starting training
    function startTrainingPhase() external returns (bytes32) {
        return _startTrainingPhase();
    }

    function forceStartTrainingPhase() external returns (bytes32) {
        return _forceStartTrainingPhase();
    }

    /// @notice Expose the internal _startEvaluationPhase function for testing
    /// @return The new phase after starting evaluation
    function startEvaluationPhase() external returns (bytes32) {
        return _startEvaluationPhase();
    }

    /// @notice Expose the internal _endEvaluatorRegistrationPhase function for testing
    /// @return The new phase after ending evaluator registration
    function endEvaluatorRegistrationPhase() external returns (bytes32) {
        return _endEvaluatorRegistrationPhase();
    }

    /// @notice Expose the internal _endEvaluationPhase function for testing
    /// @return The new phase after ending evaluation
    function endEvaluationPhase() external returns (bytes32) {
        return _endEvaluationPhase();
    }

    /// @notice Expose the internal _requirePhase function for testing
    /// @param phase The required phase
    function requirePhase(bytes32 phase) external view {
        _requirePhase(phase);
    }

    /// @notice Public wrapper for testing
    function initialize(
        BaseTrainingPhases.TrainingPhaseConfiguration memory trainingConfig,
        BaseTrainingPhases.EvaluationPhaseConfiguration memory evaluationConfig
    ) external initializer {
        __BaseTrainingPhases_init(
            trainingConfig,
            evaluationConfig
        );
    }
}

contract BaseTrainingPhasesTest is Test {
    MockBaseTrainingPhases public implementation;
    ERC1967Proxy public proxy;
    MockBaseTrainingPhases public trainingPhases;
    BaseTrainingPhases.TrainingPhaseConfiguration public trainingConfig;
    BaseTrainingPhases.EvaluationPhaseConfiguration public evaluationConfig;

    function setUp() public {
        // Deploy contracts
        implementation = new MockBaseTrainingPhases();

        trainingConfig = BaseTrainingPhases.TrainingPhaseConfiguration({ttl: 1000});
        evaluationConfig = BaseTrainingPhases.EvaluationPhaseConfiguration({ttl: 1000, registrationTtl: 1000});
        
        // Deploy proxy with initialization data
        bytes memory initData = abi.encodeWithSelector(
            MockBaseTrainingPhases.initialize.selector,
            trainingConfig,
            evaluationConfig
        );
        proxy = new ERC1967Proxy(address(implementation), initData);
        trainingPhases = MockBaseTrainingPhases(address(proxy));
    }

    // ============================================================================
    // INITIALIZATION TESTS
    // ============================================================================

    function test_initialize() public view {
        // Test that initialize works correctly
        assertTrue(trainingPhases.isIdle(), "Should start in IDLE phase");
        assertEq(trainingPhases.getCurrentPhase(), trainingPhases.IDLE_PHASE(), "Current phase should be IDLE");
        assertFalse(trainingPhases.isTraining(), "Should not be in training phase initially");
        assertFalse(trainingPhases.isEvaluation(), "Should not be in evaluation phase initially");
    }

    function test_initialize_canOnlyBeCalledOnce() public {
        // Test that initialize cannot be called twice
        vm.expectRevert();
        trainingPhases.initialize(trainingConfig, evaluationConfig);
    }

    // ============================================================================
    // PHASE STATE TESTS
    // ============================================================================

    function test_getCurrentPhase() public view {
        assertEq(trainingPhases.getCurrentPhase(), trainingPhases.IDLE_PHASE(), "Should return current phase");
    }

    function test_isIdle() public view {
        assertTrue(trainingPhases.isIdle(), "Should be idle initially");
        assertTrue(trainingPhases.isPhase(trainingPhases.IDLE_PHASE()), "Should be in IDLE phase");
    }

    function test_isTraining() public {
        // Initially not in training phase
        assertFalse(trainingPhases.isTraining(), "Should not be in training phase initially");
        assertFalse(trainingPhases.isPhase(trainingPhases.TRAINING_PHASE()), "Should not be in TRAINING phase initially");
        
        // Start training phase
        trainingPhases.forceStartTrainingPhase();
        assertTrue(trainingPhases.isTraining(), "Should be in training phase after starting");
        assertTrue(trainingPhases.isPhase(trainingPhases.TRAINING_PHASE()), "Should be in TRAINING phase after starting");
    }

    function test_isEvaluation() public {
        // Initially not in evaluation phase
        assertFalse(trainingPhases.isEvaluation(), "Should not be in evaluation phase initially");
        assertFalse(trainingPhases.isPhase(trainingPhases.EVALUATOR_REGISTRATION_PHASE()), "Should not be in EVALUATOR_REGISTRATION phase initially");
    }

    function test_isPhase() public {
        assertTrue(trainingPhases.isPhase(trainingPhases.IDLE_PHASE()), "Should be in IDLE phase");
        assertFalse(trainingPhases.isPhase(trainingPhases.TRAINING_PHASE()), "Should not be in TRAINING phase");
        assertFalse(trainingPhases.isPhase(trainingPhases.EVALUATOR_REGISTRATION_PHASE()), "Should not be in EVALUATOR_REGISTRATION phase");
        assertFalse(trainingPhases.isPhase(trainingPhases.EVALUATION_PHASE()), "Should not be in EVALUATION phase");
    }

    // ============================================================================
    // PHASE TRANSITION TESTS
    // ============================================================================

    function test_startTrainingPhase() public {
        bytes32 newPhase = trainingPhases.forceStartTrainingPhase();
        
        assertEq(newPhase, trainingPhases.TRAINING_PHASE(), "Should return TRAINING phase");
        assertTrue(trainingPhases.isTraining(), "Should be in training phase");
        assertEq(trainingPhases.getCurrentPhase(), trainingPhases.TRAINING_PHASE(), "Current phase should be TRAINING");
    }

    function test_updatePhase_trainingPhase_notExpired() public {
      trainingPhases.forceStartTrainingPhase();
        bytes32 newPhase = trainingPhases.updatePhase();
        assertEq(newPhase, trainingPhases.TRAINING_PHASE(), "Should return TRAINING phase");
        assertTrue(trainingPhases.isTraining(), "Should be in training phase");
        assertEq(trainingPhases.getCurrentPhase(), trainingPhases.TRAINING_PHASE(), "Current phase should be TRAINING");
    }

    function test_updatePhase_trainingPhase_expired() public {
        trainingPhases.forceStartTrainingPhase();
        vm.warp(block.timestamp + trainingConfig.ttl);
        bytes32 newPhase = trainingPhases.updatePhase();
        assertEq(newPhase, trainingPhases.EVALUATOR_REGISTRATION_PHASE(), "Should return EVALUATOR_REGISTRATION phase");
        assertTrue(trainingPhases.isEvaluation(), "Should be in evaluation phase");
    }


    function test_updatePhase_endEvaluatorRegistrationPhase() public {
        trainingPhases.forceStartTrainingPhase();
        vm.warp(block.timestamp + trainingConfig.ttl + evaluationConfig.registrationTtl);
        bytes32 newPhase = trainingPhases.updatePhase();
        assertEq(newPhase, trainingPhases.EVALUATION_PHASE(), "Should return EVALUATION phase");
        assertTrue(trainingPhases.isEvaluation(), "Should be in evaluation phase");
    }

    function test_updatePhase_endEvaluationPhase() public {
        trainingPhases.forceStartTrainingPhase();
        vm.warp(block.timestamp + trainingConfig.ttl + evaluationConfig.registrationTtl + evaluationConfig.ttl);
        bytes32 newPhase = trainingPhases.updatePhase();
        
        assertEq(newPhase, trainingPhases.IDLE_PHASE(), "Should return IDLE phase");
        assertTrue(trainingPhases.isIdle(), "Should still be idle");
    }

    // ============================================================================
    // TIMING TESTS (Basic functionality only)
    // ============================================================================

    function test_updatePhase_idlePhase() public {
        // In idle phase, updatePhase should return idle
        bytes32 newPhase = trainingPhases.updatePhase();
        assertEq(newPhase, trainingPhases.IDLE_PHASE(), "Should return IDLE phase when in idle");
        assertTrue(trainingPhases.isIdle(), "Should still be idle");
    }


    // ============================================================================
    // ERROR HANDLING TESTS
    // ============================================================================

    function test_requirePhase_correctPhase() public {
        // Should not revert when in correct phase
        trainingPhases.requirePhase(trainingPhases.IDLE_PHASE());
    }

    function test_requirePhase_wrongPhase() public {
        // Start training phase first
        trainingPhases.forceStartTrainingPhase();
        
        // Verify we're in training phase
        assertTrue(trainingPhases.isTraining(), "Should be in training phase");
        
        // Should revert when in wrong phase, vm.expectRevert wasn't working...
        bool reverted = false;
        try trainingPhases.requirePhase(trainingPhases.IDLE_PHASE()) {
            // If we get here, it didn't revert
        } catch {
            reverted = true;
        }
        
        assertTrue(reverted, "Should have reverted when requiring wrong phase");
    }

    // ============================================================================
    // INTEGRATION TESTS
    // ============================================================================

    function test_basicPhaseTransitions() public {
        // Start in idle
        assertTrue(trainingPhases.isIdle(), "Should start in idle");
        
        // Start training phase
        bytes32 trainingPhase = trainingPhases.forceStartTrainingPhase();
        assertEq(trainingPhase, trainingPhases.TRAINING_PHASE(), "Should return training phase");
        assertTrue(trainingPhases.isTraining(), "Should be in training phase");
        
        // Test other internal functions return correct phases
        bytes32 evalPhase = trainingPhases.startEvaluationPhase();
        assertEq(evalPhase, trainingPhases.EVALUATOR_REGISTRATION_PHASE(), "Should return evaluator registration phase");
        
        bytes32 endEvalPhase = trainingPhases.endEvaluationPhase();
        assertEq(endEvalPhase, trainingPhases.IDLE_PHASE(), "Should return idle phase");
        
        // Still in training since we used forceStartTrainingPhase
        assertTrue(trainingPhases.isTraining(), "Should still be in training");
    }

    function test_phaseConstants() public view {
        // Test that phase constants are correctly defined
        assertEq(trainingPhases.IDLE_PHASE(), keccak256("IDLE"), "IDLE_PHASE should be correct");
        assertEq(trainingPhases.TRAINING_PHASE(), keccak256("TRAINING"), "TRAINING_PHASE should be correct");
        assertEq(trainingPhases.EVALUATOR_REGISTRATION_PHASE(), keccak256("EVALUATOR_REGISTRATION"), "EVALUATOR_REGISTRATION_PHASE should be correct");
        assertEq(trainingPhases.EVALUATION_PHASE(), keccak256("EVALUATION"), "EVALUATION_PHASE should be correct");
    }

}
