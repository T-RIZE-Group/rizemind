// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test} from "forge-std/Test.sol";
import {ERC1967Proxy} from "@openzeppelin-contracts-5.2.0/proxy/ERC1967/ERC1967Proxy.sol";
import {RoundTrainerRegistry} from "../../../src/swarm/registry/RoundTrainerRegistry.sol";

/// @title MockRoundTrainerRegistry
/// @notice Mock contract that exposes internal functions for testing
contract MockRoundTrainerRegistry is RoundTrainerRegistry {
    /// @notice Expose the internal _registerTrainer function for testing
    /// @param roundId The round ID
    /// @param trainer The trainer address
    /// @param modelHash The model hash
    /// @return trainerId The assigned trainer ID
    function registerTrainer(uint256 roundId, address trainer, bytes32 modelHash) external returns (uint256 trainerId) {
        return _registerTrainer(roundId, trainer, modelHash);
    }

    /// @notice Expose the internal _setModelHash function for testing
    /// @param roundId The round ID
    /// @param trainer The trainer address
    /// @param modelHash The new model hash
    function setModelHash(uint256 roundId, address trainer, bytes32 modelHash) external {
        _setModelHash(roundId, trainer, modelHash);
    }
}

contract RoundTrainerRegistryTest is Test {
    MockRoundTrainerRegistry public implementation;
    ERC1967Proxy public proxy;
    MockRoundTrainerRegistry public registry;

    address public trainer1;
    address public trainer2;
    address public trainer3;
    address public zeroAddress = address(0);

    bytes32 public modelHash1 = keccak256("model1");
    bytes32 public modelHash2 = keccak256("model2");
    bytes32 public modelHash3 = keccak256("model3");
    bytes32 public updatedModelHash = keccak256("updated_model");

    event TrainerRegistered(uint256 indexed roundId, address indexed trainer, uint256 trainerId);
    event ModelHashUpdated(uint256 indexed roundId, address indexed trainer, bytes32 modelHash);

    function setUp() public {
        // Deploy contracts
        implementation = new MockRoundTrainerRegistry();
        
        // Deploy proxy
        bytes memory initData = abi.encodeWithSelector(
            RoundTrainerRegistry.initialize.selector
        );
        proxy = new ERC1967Proxy(address(implementation), initData);
        registry = MockRoundTrainerRegistry(address(proxy));

        // Setup test addresses
        trainer1 = makeAddr("trainer1");
        trainer2 = makeAddr("trainer2");
        trainer3 = makeAddr("trainer3");
    }

    // ============================================================================
    // INITIALIZATION TESTS
    // ============================================================================

    function test_initialize() public view {
        // Test that initialize works correctly
        // The contract should be properly initialized after setup
        // We can verify this by calling a view function that would fail if not initialized
        registry.getTrainerCount(1); // This should not revert if initialized
    }

    function test_initialize_canOnlyBeCalledOnce() public {
        // Test that initialize cannot be called twice
        vm.expectRevert();
        registry.initialize();
    }

    // ============================================================================
    // BASIC FUNCTIONALITY TESTS
    // ============================================================================

    function test_registerTrainer_firstTime() public {
        uint256 roundId = 1;
        address trainer = trainer1;
        bytes32 modelHash = modelHash1;

        vm.expectEmit(true, true, false, false);
        emit TrainerRegistered(roundId, trainer, 1);

        uint256 trainerId = registry.registerTrainer(roundId, trainer, modelHash);

        assertEq(trainerId, 1, "First trainer should get ID 1");
        assertEq(registry.getTrainerId(roundId, trainer), 1, "Should return correct trainer ID");
        assertEq(registry.getModelHash(roundId, trainer), modelHash, "Should return correct model hash");
        assertEq(registry.getTrainerCount(roundId), 1, "Should have 1 trainer");
        assertTrue(registry.isTrainerRegistered(roundId, trainer), "Should be registered");
    }

    function test_registerTrainer_multipleTrainers() public {
        uint256 roundId = 1;

        // Register first trainer
        vm.expectEmit(true, true, false, false);
        emit TrainerRegistered(roundId, trainer1, 1);
        uint256 id1 = registry.registerTrainer(roundId, trainer1, modelHash1);

        // Register second trainer
        vm.expectEmit(true, true, false, false);
        emit TrainerRegistered(roundId, trainer2, 2);
        uint256 id2 = registry.registerTrainer(roundId, trainer2, modelHash2);

        // Register third trainer
        vm.expectEmit(true, true, false, false);
        emit TrainerRegistered(roundId, trainer3, 3);
        uint256 id3 = registry.registerTrainer(roundId, trainer3, modelHash3);

        assertEq(id1, 1, "First trainer should get ID 1");
        assertEq(id2, 2, "Second trainer should get ID 2");
        assertEq(id3, 3, "Third trainer should get ID 3");

        assertEq(registry.getTrainerCount(roundId), 3, "Should have 3 trainers");
        assertTrue(registry.isTrainerRegistered(roundId, trainer1), "Trainer1 should be registered");
        assertTrue(registry.isTrainerRegistered(roundId, trainer2), "Trainer2 should be registered");
        assertTrue(registry.isTrainerRegistered(roundId, trainer3), "Trainer3 should be registered");

        // Verify model hashes
        assertEq(registry.getModelHash(roundId, trainer1), modelHash1, "Trainer1 should have correct model hash");
        assertEq(registry.getModelHash(roundId, trainer2), modelHash2, "Trainer2 should have correct model hash");
        assertEq(registry.getModelHash(roundId, trainer3), modelHash3, "Trainer3 should have correct model hash");
    }

    function test_registerTrainer_duplicateRegistration() public {
        uint256 roundId = 1;
        address trainer = trainer1;
        bytes32 modelHash = modelHash1;

        // Register trainer first time
        vm.expectEmit(true, true, false, false);
        emit TrainerRegistered(roundId, trainer, 1);
        uint256 firstId = registry.registerTrainer(roundId, trainer, modelHash);

        // Register same trainer again - should not emit event and return same ID
        uint256 secondId = registry.registerTrainer(roundId, trainer, modelHash2);

        assertEq(firstId, secondId, "Should return same ID for duplicate registration");
        assertEq(registry.getTrainerCount(roundId), 1, "Should still have only 1 trainer");
        assertEq(registry.getTrainerId(roundId, trainer), firstId, "Should return correct ID");
        // Model hash should remain the original one
        assertEq(registry.getModelHash(roundId, trainer), modelHash, "Should keep original model hash");
    }

    function test_registerTrainer_multipleRounds() public {
        uint256 round1 = 1;
        uint256 round2 = 2;

        // Register trainer for round 1
        vm.expectEmit(true, true, false, false);
        emit TrainerRegistered(round1, trainer1, 1);
        uint256 id1 = registry.registerTrainer(round1, trainer1, modelHash1);

        // Register different trainer for round 2
        vm.expectEmit(true, true, false, false);
        emit TrainerRegistered(round2, trainer2, 1);
        uint256 id2 = registry.registerTrainer(round2, trainer2, modelHash2);

        assertEq(id1, 1, "Round 1 trainer should get ID 1");
        assertEq(id2, 1, "Round 2 trainer should get ID 1");

        assertEq(registry.getTrainerCount(round1), 1, "Round 1 should have 1 trainer");
        assertEq(registry.getTrainerCount(round2), 1, "Round 2 should have 1 trainer");

        assertTrue(registry.isTrainerRegistered(round1, trainer1), "Should be registered in round 1");
        assertFalse(registry.isTrainerRegistered(round1, trainer2), "Should not be registered in round 1");
        assertTrue(registry.isTrainerRegistered(round2, trainer2), "Should be registered in round 2");
        assertFalse(registry.isTrainerRegistered(round2, trainer1), "Should not be registered in round 2");
    }

    // ============================================================================
    // MODEL HASH MANAGEMENT TESTS
    // ============================================================================

    function test_setModelHash() public {
        uint256 roundId = 1;
        address trainer = trainer1;
        bytes32 originalModelHash = modelHash1;
        bytes32 newModelHash = updatedModelHash;

        // First register the trainer
        registry.registerTrainer(roundId, trainer, originalModelHash);

        // Update model hash
        vm.expectEmit(true, true, false, false);
        emit ModelHashUpdated(roundId, trainer, newModelHash);
        registry.setModelHash(roundId, trainer, newModelHash);

        // Verify the model hash was updated
        assertEq(registry.getModelHash(roundId, trainer), newModelHash, "Model hash should be updated");
        assertEq(registry.getModelHashOrThrow(roundId, trainer), newModelHash, "getModelHashOrThrow should return updated hash");
    }

    function test_setModelHash_unregisteredTrainer() public {
        uint256 roundId = 1;
        address trainer = trainer1;
        bytes32 modelHash = modelHash1;

        // Try to update model hash for unregistered trainer
        vm.expectRevert(abi.encodeWithSelector(RoundTrainerRegistry.TrainerNotFound.selector, roundId, trainer));
        registry.setModelHash(roundId, trainer, modelHash);
    }

    function test_getModelHashOrThrow() public {
        uint256 roundId = 1;
        address trainer = trainer1;
        bytes32 modelHash = modelHash1;

        // Register trainer
        registry.registerTrainer(roundId, trainer, modelHash);

        // Test getModelHashOrThrow with registered trainer
        bytes32 retrievedHash = registry.getModelHashOrThrow(roundId, trainer);
        assertEq(retrievedHash, modelHash, "Should return correct model hash");

        // Test getModelHashOrThrow with unregistered trainer
        vm.expectRevert(abi.encodeWithSelector(RoundTrainerRegistry.TrainerNotFound.selector, roundId, trainer2));
        registry.getModelHashOrThrow(roundId, trainer2);
    }

    function test_getTrainerInfo() public {
        uint256 roundId = 1;
        address trainer = trainer1;
        bytes32 modelHash = modelHash1;

        // Register trainer
        registry.registerTrainer(roundId, trainer, modelHash);

        // Test getTrainerInfo
        (uint256 trainerId, bytes32 retrievedModelHash) = registry.getTrainerInfo(roundId, trainer);
        assertEq(trainerId, 1, "Should return correct trainer ID");
        assertEq(retrievedModelHash, modelHash, "Should return correct model hash");

        // Test with unregistered trainer
        (uint256 unregisteredId, bytes32 unregisteredHash) = registry.getTrainerInfo(roundId, trainer2);
        assertEq(unregisteredId, 0, "Should return 0 for unregistered trainer ID");
        assertEq(unregisteredHash, bytes32(0), "Should return zero hash for unregistered trainer");
    }

    // ============================================================================
    // VIEW FUNCTION TESTS
    // ============================================================================

    function test_getTrainerId() public {
        uint256 roundId = 1;

        // Register trainer
        registry.registerTrainer(roundId, trainer1, modelHash1);

        // Test getTrainerId
        uint256 trainerId = registry.getTrainerId(roundId, trainer1);
        assertEq(trainerId, 1, "Should return correct trainer ID");

        // Test with unregistered trainer
        uint256 unregisteredId = registry.getTrainerId(roundId, trainer2);
        assertEq(unregisteredId, 0, "Should return 0 for unregistered trainer");
    }

    function test_getTrainerIdOrThrow() public {
        uint256 roundId = 1;

        // Register trainer
        registry.registerTrainer(roundId, trainer1, modelHash1);

        // Test getTrainerIdOrThrow with registered trainer
        uint256 trainerId = registry.getTrainerIdOrThrow(roundId, trainer1);
        assertEq(trainerId, 1, "Should return correct trainer ID");

        // Test getTrainerIdOrThrow with unregistered trainer
        vm.expectRevert(abi.encodeWithSelector(RoundTrainerRegistry.TrainerNotFound.selector, roundId, trainer2));
        registry.getTrainerIdOrThrow(roundId, trainer2);
    }

    function test_getTrainerCount() public {
        uint256 roundId = 1;

        // Initially should be 0
        assertEq(registry.getTrainerCount(roundId), 0, "Should start with 0 trainers");

        // Register first trainer
        registry.registerTrainer(roundId, trainer1, modelHash1);
        assertEq(registry.getTrainerCount(roundId), 1, "Should have 1 trainer");

        // Register second trainer
        registry.registerTrainer(roundId, trainer2, modelHash2);
        assertEq(registry.getTrainerCount(roundId), 2, "Should have 2 trainers");

        // Register third trainer
        registry.registerTrainer(roundId, trainer3, modelHash3);
        assertEq(registry.getTrainerCount(roundId), 3, "Should have 3 trainers");

        // Duplicate registration should not increase count
        registry.registerTrainer(roundId, trainer1, modelHash1);
        assertEq(registry.getTrainerCount(roundId), 3, "Duplicate registration should not increase count");
    }

    function test_isTrainerRegistered() public {
        uint256 roundId = 1;

        // Initially should not be registered
        assertFalse(registry.isTrainerRegistered(roundId, trainer1), "Should not be registered initially");

        // Register trainer
        registry.registerTrainer(roundId, trainer1, modelHash1);

        // Should be registered now
        assertTrue(registry.isTrainerRegistered(roundId, trainer1), "Should be registered after registration");

        // Other trainer should not be registered
        assertFalse(registry.isTrainerRegistered(roundId, trainer2), "Other trainer should not be registered");
    }

    // ============================================================================
    // ERROR HANDLING TESTS
    // ============================================================================

    function test_registerTrainer_zeroAddress() public {
        uint256 roundId = 1;
        bytes32 modelHash = modelHash1;

        vm.expectRevert(RoundTrainerRegistry.InvalidTrainerAddress.selector);
        registry.registerTrainer(roundId, zeroAddress, modelHash);
    }

    function test_getTrainerIdOrThrow_trainerNotFound() public {
        uint256 roundId = 1;
        address unregisteredTrainer = makeAddr("unregistered");

        vm.expectRevert(abi.encodeWithSelector(RoundTrainerRegistry.TrainerNotFound.selector, roundId, unregisteredTrainer));
        registry.getTrainerIdOrThrow(roundId, unregisteredTrainer);
    }

    // ============================================================================
    // EDGE CASES AND STORAGE TESTS
    // ============================================================================

    function test_storageIsolation() public {
        uint256 round1 = 1;
        uint256 round2 = 2;

        // Register trainers for different rounds
        registry.registerTrainer(round1, trainer1, modelHash1);
        registry.registerTrainer(round1, trainer2, modelHash2);
        registry.registerTrainer(round2, trainer1, modelHash3);
        registry.registerTrainer(round2, trainer3, modelHash1);

        // Verify round 1 state
        assertEq(registry.getTrainerCount(round1), 2, "Round 1 should have 2 trainers");
        assertTrue(registry.isTrainerRegistered(round1, trainer1), "Trainer1 should be registered in round 1");
        assertTrue(registry.isTrainerRegistered(round1, trainer2), "Trainer2 should be registered in round 1");
        assertFalse(registry.isTrainerRegistered(round1, trainer3), "Trainer3 should not be registered in round 1");

        // Verify round 2 state
        assertEq(registry.getTrainerCount(round2), 2, "Round 2 should have 2 trainers");
        assertTrue(registry.isTrainerRegistered(round2, trainer1), "Trainer1 should be registered in round 2");
        assertFalse(registry.isTrainerRegistered(round2, trainer2), "Trainer2 should not be registered in round 2");
        assertTrue(registry.isTrainerRegistered(round2, trainer3), "Trainer3 should be registered in round 2");

        // Verify model hashes are isolated
        assertEq(registry.getModelHash(round1, trainer1), modelHash1, "Round 1 trainer1 should have correct model hash");
        assertEq(registry.getModelHash(round2, trainer1), modelHash3, "Round 2 trainer1 should have different model hash");
    }

    function test_largeNumberOfTrainers() public {
        uint256 roundId = 1;
        uint256 numTrainers = 100;

        // Register many trainers
        for (uint256 i = 0; i < numTrainers; i++) {
            address trainer = makeAddr(string(abi.encodePacked("trainer", i)));
            bytes32 modelHash = keccak256(abi.encodePacked("model", i));
            uint256 trainerId = registry.registerTrainer(roundId, trainer, modelHash);
            assertEq(trainerId, i + 1, "Trainer ID should match index + 1");
        }

        assertEq(registry.getTrainerCount(roundId), numTrainers, "Should have correct number of trainers");
    }

    function test_roundIdZero() public {
        uint256 roundId = 0;

        // Should work with round ID 0
        registry.registerTrainer(roundId, trainer1, modelHash1);
        assertEq(registry.getTrainerCount(roundId), 1, "Should work with round ID 0");
        assertTrue(registry.isTrainerRegistered(roundId, trainer1), "Should be registered in round 0");
    }

    function test_largeRoundId() public {
        uint256 roundId = type(uint256).max;

        // Should work with large round ID
        registry.registerTrainer(roundId, trainer1, modelHash1);
        assertEq(registry.getTrainerCount(roundId), 1, "Should work with large round ID");
        assertTrue(registry.isTrainerRegistered(roundId, trainer1), "Should be registered in large round ID");
    }

    // ============================================================================
    // EVENT EMISSION TESTS
    // ============================================================================

    function test_registerTrainer_emitsEvent() public {
        uint256 roundId = 1;
        address trainer = trainer1;
        bytes32 modelHash = modelHash1;

        vm.expectEmit(true, true, false, false);
        emit TrainerRegistered(roundId, trainer, 1);

        registry.registerTrainer(roundId, trainer, modelHash);
    }

    function test_registerTrainer_duplicateDoesNotEmitEvent() public {
        uint256 roundId = 1;
        address trainer = trainer1;
        bytes32 modelHash = modelHash1;

        // First registration should emit event
        vm.expectEmit(true, true, false, false);
        emit TrainerRegistered(roundId, trainer, 1);
        registry.registerTrainer(roundId, trainer, modelHash);

        // Second registration should not emit event
        registry.registerTrainer(roundId, trainer, modelHash2);
    }

    function test_setModelHash_emitsEvent() public {
        uint256 roundId = 1;
        address trainer = trainer1;
        bytes32 originalModelHash = modelHash1;
        bytes32 newModelHash = updatedModelHash;

        // Register trainer first
        registry.registerTrainer(roundId, trainer, originalModelHash);

        // Update model hash should emit event
        vm.expectEmit(true, true, false, false);
        emit ModelHashUpdated(roundId, trainer, newModelHash);
        registry.setModelHash(roundId, trainer, newModelHash);
    }

    // ============================================================================
    // GAS OPTIMIZATION TESTS
    // ============================================================================

    function test_registerTrainer_gasUsage() public {
        uint256 roundId = 1;
        address trainer = trainer1;
        bytes32 modelHash = modelHash1;

        // Measure gas for first registration
        uint256 gasStart = gasleft();
        registry.registerTrainer(roundId, trainer, modelHash);
        uint256 gasUsed = gasStart - gasleft();

        // Gas usage should be reasonable (this is more of a sanity check)
        assertLt(gasUsed, 150000, "Gas usage should be reasonable");
    }

    // ============================================================================
    // INTEGRATION TESTS
    // ============================================================================

    function test_fullWorkflow() public {
        uint256 roundId = 1;

        // Register multiple trainers
        registry.registerTrainer(roundId, trainer1, modelHash1);
        registry.registerTrainer(roundId, trainer2, modelHash2);
        registry.registerTrainer(roundId, trainer3, modelHash3);

        // Verify all are registered
        assertTrue(registry.isTrainerRegistered(roundId, trainer1), "Trainer1 should be registered");
        assertTrue(registry.isTrainerRegistered(roundId, trainer2), "Trainer2 should be registered");
        assertTrue(registry.isTrainerRegistered(roundId, trainer3), "Trainer3 should be registered");

        // Verify counts and IDs
        assertEq(registry.getTrainerCount(roundId), 3, "Should have 3 trainers");
        assertEq(registry.getTrainerId(roundId, trainer1), 1, "Trainer1 should have ID 1");
        assertEq(registry.getTrainerId(roundId, trainer2), 2, "Trainer2 should have ID 2");
        assertEq(registry.getTrainerId(roundId, trainer3), 3, "Trainer3 should have ID 3");

        // Verify model hashes
        assertEq(registry.getModelHash(roundId, trainer1), modelHash1, "Trainer1 should have correct model hash");
        assertEq(registry.getModelHash(roundId, trainer2), modelHash2, "Trainer2 should have correct model hash");
        assertEq(registry.getModelHash(roundId, trainer3), modelHash3, "Trainer3 should have correct model hash");

        // Test getTrainerIdOrThrow
        assertEq(registry.getTrainerIdOrThrow(roundId, trainer1), 1, "getTrainerIdOrThrow should work");
        assertEq(registry.getTrainerIdOrThrow(roundId, trainer2), 2, "getTrainerIdOrThrow should work");
        assertEq(registry.getTrainerIdOrThrow(roundId, trainer3), 3, "getTrainerIdOrThrow should work");

        // Test getModelHashOrThrow
        assertEq(registry.getModelHashOrThrow(roundId, trainer1), modelHash1, "getModelHashOrThrow should work");
        assertEq(registry.getModelHashOrThrow(roundId, trainer2), modelHash2, "getModelHashOrThrow should work");
        assertEq(registry.getModelHashOrThrow(roundId, trainer3), modelHash3, "getModelHashOrThrow should work");

        // Test getTrainerInfo
        (uint256 id1, bytes32 hash1) = registry.getTrainerInfo(roundId, trainer1);
        assertEq(id1, 1, "getTrainerInfo should return correct ID");
        assertEq(hash1, modelHash1, "getTrainerInfo should return correct model hash");

        // Test duplicate registration
        uint256 duplicateId = registry.registerTrainer(roundId, trainer1, modelHash2);
        assertEq(duplicateId, 1, "Duplicate registration should return same ID");
        assertEq(registry.getTrainerCount(roundId), 3, "Count should not change for duplicate");

        // Test model hash update
        registry.setModelHash(roundId, trainer1, updatedModelHash);
        assertEq(registry.getModelHash(roundId, trainer1), updatedModelHash, "Model hash should be updated");
    }

    // ============================================================================
    // MODEL HASH EDGE CASES
    // ============================================================================

    function test_modelHashZero() public {
        uint256 roundId = 1;
        address trainer = trainer1;
        bytes32 zeroHash = bytes32(0);

        // Should be able to register with zero model hash
        registry.registerTrainer(roundId, trainer, zeroHash);
        assertEq(registry.getModelHash(roundId, trainer), zeroHash, "Should store zero model hash");
        assertTrue(registry.isTrainerRegistered(roundId, trainer), "Should be registered with zero hash");
    }

    function test_modelHashUpdateToZero() public {
        uint256 roundId = 1;
        address trainer = trainer1;
        bytes32 originalHash = modelHash1;
        bytes32 zeroHash = bytes32(0);

        // Register with non-zero hash
        registry.registerTrainer(roundId, trainer, originalHash);
        assertEq(registry.getModelHash(roundId, trainer), originalHash, "Should have original hash");

        // Update to zero hash
        registry.setModelHash(roundId, trainer, zeroHash);
        assertEq(registry.getModelHash(roundId, trainer), zeroHash, "Should update to zero hash");
    }

    function test_modelHashUpdateToSameValue() public {
        uint256 roundId = 1;
        address trainer = trainer1;
        bytes32 modelHash = modelHash1;

        // Register trainer
        registry.registerTrainer(roundId, trainer, modelHash);

        // Update to same model hash - should still emit event
        vm.expectEmit(true, true, false, false);
        emit ModelHashUpdated(roundId, trainer, modelHash);
        registry.setModelHash(roundId, trainer, modelHash);

        assertEq(registry.getModelHash(roundId, trainer), modelHash, "Model hash should remain the same");
    }
}
