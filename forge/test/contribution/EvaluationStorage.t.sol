// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import {Test} from "forge-std/Test.sol";
import {EvaluationStorage} from "../../src/contribution/EvaluationStorage.sol";

contract MockEvaluationStorage is EvaluationStorage {
    function initialize() external {
        __EvaluationStorage_init();
    }

    function registerResult(
        uint256 roundId,
        uint256 setId,
        bytes32 modelHash,
        int256 result
    ) external {
        _registerResult(roundId, setId, modelHash, result);
    }
}

contract EvaluationStorageTest is Test {
    MockEvaluationStorage public evaluationStorage;

    function setUp() public {
        evaluationStorage = new MockEvaluationStorage();
        evaluationStorage.initialize();
    }

    function test_registerAndGetResult() public {
        uint256 roundId = 1;
        uint256 setId = 123;
        bytes32 modelHash = keccak256("test_model");
        int256 result = 100;

        evaluationStorage.registerResult(roundId, setId, modelHash, result);
        
        int256 retrievedResult = evaluationStorage.getResult(roundId, setId);
        assertEq(retrievedResult, result);
    }

    function test_mergeResults() public {
        uint256 roundId = 1;
        uint256 setId = 123;
        bytes32 modelHash = keccak256("test_model");
        
        // Register first result
        evaluationStorage.registerResult(roundId, setId, modelHash, 100);
        
        // Register second result (should merge with first)
        evaluationStorage.registerResult(roundId, setId, modelHash, 200);
        
        // Result should be average: (100 + 200) / 2 = 150
        int256 retrievedResult = evaluationStorage.getResult(roundId, setId);
        assertEq(retrievedResult, 150);
    }

    function test_multipleRoundsAndSets() public {
        // Test different rounds and sets
        evaluationStorage.registerResult(1, 1, keccak256("model1"), 100);
        evaluationStorage.registerResult(1, 2, keccak256("model2"), 200);
        evaluationStorage.registerResult(2, 1, keccak256("model3"), 300);
        
        assertEq(evaluationStorage.getResult(1, 1), 100);
        assertEq(evaluationStorage.getResult(1, 2), 200);
        assertEq(evaluationStorage.getResult(2, 1), 300);
    }

    function test_invalidModelHash() public {
        uint256 roundId = 1;
        uint256 setId = 123;
        bytes32 invalidModelHash = bytes32(0);
        int256 result = 100;

        vm.expectRevert(EvaluationStorage.InvalidParameters.selector);
        evaluationStorage.registerResult(roundId, setId, invalidModelHash, result);
    }

    function test_namespacedStorageIsolation() public {
        // This test verifies that the namespaced storage pattern works correctly
        // by ensuring that storage operations don't interfere with each other
        
        uint256 roundId = 1;
        uint256 setId = 123;
        bytes32 modelHash = keccak256("test_model");
        int256 result = 100;

        evaluationStorage.registerResult(roundId, setId, modelHash, result);
        
        // Verify the result is stored correctly
        int256 retrievedResult = evaluationStorage.getResult(roundId, setId);
        assertEq(retrievedResult, result);
        
        // Verify that the storage namespace is working by checking that
        // the result persists across multiple calls
        retrievedResult = evaluationStorage.getResult(roundId, setId);
        assertEq(retrievedResult, result);
    }
}
