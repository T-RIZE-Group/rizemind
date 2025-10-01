// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Initializable} from "@openzeppelin-contracts-upgradeable-5.2.0/proxy/utils/Initializable.sol";

/// @title RoundEvaluatorRegistry
/// @notice Registry contract for managing evaluators per round
contract RoundEvaluatorRegistry is Initializable {
    /// @dev Structure to store evaluators for a specific round
    struct RoundEvaluators {
        mapping(address => uint256) evaluators; // Maps evaluator address to their assigned ID
        uint256 count; // Total number of evaluators registered
        uint256 nextId; // Next ID to assign to a new evaluator
    }

    /// @dev Storage namespace for RoundEvaluatorRegistry
    struct RoundEvaluatorRegistryStorage {
        mapping(uint256 => RoundEvaluators) roundEvaluators;
    }

    // Storage slot for RoundEvaluatorRegistry namespace
    bytes32 private constant ROUND_EVALUATOR_REGISTRY_STORAGE = keccak256("RoundEvaluatorRegistry.storage");

    /// @notice Emitted when an evaluator is registered for a round
    /// @param roundId The round ID
    /// @param evaluator The evaluator address
    /// @param evaluatorId The assigned evaluator ID
    event EvaluatorRegistered(uint256 indexed roundId, address indexed evaluator, uint256 evaluatorId);

    /// @notice Error thrown when trying to register a zero address evaluator
    error InvalidEvaluatorAddress();

    /// @notice Error thrown when trying to access a non-existent evaluator
    error EvaluatorNotFound(uint256 roundId, address evaluator);

    /// @notice Initializes the contract
    /// @dev This function can only be called once during proxy deployment
    function initialize() external virtual initializer {
        __RoundEvaluatorRegistry_init();
    }

    function __RoundEvaluatorRegistry_init() internal onlyInitializing {
        // No initialization required for this contract
    }

    /// @notice Register an evaluator for a specific round
    /// @dev Internal function to register evaluators and assign sequential IDs
    /// @param roundId The round ID
    /// @param evaluator The evaluator address
    /// @return evaluatorId The assigned evaluator ID
    function _registerEvaluator(uint256 roundId, address evaluator) internal returns (uint256 evaluatorId) {
        if (evaluator == address(0)) {
            revert InvalidEvaluatorAddress();
        }

        RoundEvaluatorRegistryStorage storage $ = _getRoundEvaluatorRegistryStorage();
        RoundEvaluators storage roundEvaluators = $.roundEvaluators[roundId];
        
        // If this is a new evaluator for this round, assign a new ID
        if (roundEvaluators.evaluators[evaluator] == 0) {
            evaluatorId = ++roundEvaluators.count;
            roundEvaluators.evaluators[evaluator] = evaluatorId;
            emit EvaluatorRegistered(roundId, evaluator, evaluatorId);
        } else {
            evaluatorId = roundEvaluators.evaluators[evaluator];
        }
    }

    /// @notice Get the ID of a specific evaluator for a round
    /// @param roundId The round ID
    /// @param evaluator The evaluator address
    /// @return The evaluator's ID
    function getEvaluatorId(uint256 roundId, address evaluator) public view returns (uint256) {
        RoundEvaluatorRegistryStorage storage $ = _getRoundEvaluatorRegistryStorage();
        return $.roundEvaluators[roundId].evaluators[evaluator];
    }

    /// @notice Get the ID of a specific evaluator for a round, throwing if not found
    /// @param roundId The round ID
    /// @param evaluator The evaluator address
    /// @return The evaluator's ID
    function getEvaluatorIdOrThrow(uint256 roundId, address evaluator) public view returns (uint256) {
        RoundEvaluatorRegistryStorage storage $ = _getRoundEvaluatorRegistryStorage();
        uint256 evaluatorId = $.roundEvaluators[roundId].evaluators[evaluator];
        if (evaluatorId == 0) {
            revert EvaluatorNotFound(roundId, evaluator);
        }
        return evaluatorId;
    }

    /// @notice Get the total number of evaluators for a round
    /// @param roundId The round ID
    /// @return The number of evaluators registered for the round
    function getEvaluatorCount(uint256 roundId) public view returns (uint256) {
        RoundEvaluatorRegistryStorage storage $ = _getRoundEvaluatorRegistryStorage();
        return $.roundEvaluators[roundId].count;
    }

    /// @notice Check if an evaluator is registered for a round
    /// @param roundId The round ID
    /// @param evaluator The evaluator address
    /// @return True if the evaluator is registered for the round
    function isEvaluatorRegistered(uint256 roundId, address evaluator) public view returns (bool) {
        RoundEvaluatorRegistryStorage storage $ = _getRoundEvaluatorRegistryStorage();
        return $.roundEvaluators[roundId].evaluators[evaluator] > 0;
    }

    /// @notice Returns a pointer to the storage namespace
    /// @dev This function provides access to the namespaced storage
    function _getRoundEvaluatorRegistryStorage() private pure returns (RoundEvaluatorRegistryStorage storage $) {
        bytes32 slot = ROUND_EVALUATOR_REGISTRY_STORAGE;
        assembly {
            $.slot := slot
        }
    }
}
