// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC165} from "@openzeppelin-contracts-5.2.0/utils/introspection/IERC165.sol";
import {ISelector} from "../../sampling/ISelector.sol";
import {IContributionCalculator} from "../../contribution/types.sol";
import {Initializable} from "@openzeppelin-contracts-upgradeable-5.2.0/proxy/utils/Initializable.sol";

/// @title SwarmCore
/// @notice Core contract for managing swarm configuration
contract SwarmCore is Initializable {
    /// @dev Storage namespace for SwarmCore
    struct SwarmCoreStorage {
        address trainerSelector;
        address evaluatorSelector;
        address contributionCalculator;
    }

    // Storage slot for SwarmCore namespace
    bytes32 private constant SWARM_CORE_STORAGE = keccak256("SwarmCore.storage");

    /// @notice Emitted when the trainer selector is updated
    /// @param previousSelector The previous trainer selector address
    /// @param newSelector The new trainer selector address
    event TrainerSelectorUpdated(address indexed previousSelector, address indexed newSelector);

    /// @notice Emitted when the evaluator selector is updated
    /// @param previousSelector The previous evaluator selector address
    /// @param newSelector The new evaluator selector address
    event EvaluatorSelectorUpdated(address indexed previousSelector, address indexed newSelector);

    /// @notice Emitted when the contribution calculator is updated
    /// @param previousCalculator The previous contribution calculator address
    /// @param newCalculator The new contribution calculator address
    event ContributionCalculatorUpdated(address indexed previousCalculator, address indexed newCalculator);

    /// @notice Error thrown when the provided address doesn't support ISelector interface
    error InvalidTrainerSelector();

    /// @notice Error thrown when the provided address doesn't support IEvaluatorSelector interface
    error InvalidEvaluatorSelector();

    /// @notice Error thrown when the provided address doesn't support IContributionCalculator interface
    error InvalidContributionCalculator();

    /// @notice Error thrown when trying to call a function on a non-initialized contract
    error NotInitialized();

    /// @notice Initializes the contract with the initial trainer selector, evaluator selector, and contribution calculator
    /// @dev This function can only be called once during proxy deployment
    /// @param initialTrainerSelector The initial trainer selector contract address
    /// @param initialEvaluatorSelector The initial evaluator selector contract address
    /// @param initialContributionCalculator The initial contribution calculator contract address
    function initialize(address initialTrainerSelector, address initialEvaluatorSelector, address initialContributionCalculator) external initializer {
        __SwarmCore_init(initialTrainerSelector, initialEvaluatorSelector, initialContributionCalculator);
    }

    function __SwarmCore_init(address initialTrainerSelector, address initialEvaluatorSelector, address initialContributionCalculator) internal onlyInitializing {
        _updateTrainerSelector(initialTrainerSelector);
        _updateEvaluatorSelector(initialEvaluatorSelector);
        _updateContributionCalculator(initialContributionCalculator);
    }

    /// @notice Get the current trainer selector contract address
    /// @return The address of the current trainer selector contract
    function getTrainerSelector() public view returns (address) {
        return _getSwarmCoreStorage().trainerSelector;
    }

    /// @notice Get the current evaluator selector contract address
    /// @return The address of the current evaluator selector contract
    function getEvaluatorSelector() public view returns (address) {
        return _getSwarmCoreStorage().evaluatorSelector;
    }

    /// @notice Get the current contribution calculator contract address
    /// @return The address of the current contribution calculator contract
    function getContributionCalculator() public view returns (address) {
        return _getSwarmCoreStorage().contributionCalculator;
    }

    /// @notice Internal function to update the trainer selector
    /// @dev Validates interface support before updating storage
    /// @param newTrainerSelector The new trainer selector contract address
    function _updateTrainerSelector(address newTrainerSelector) internal {
        // Check if the new address supports ISelector interface
        if (!_supportsISelector(newTrainerSelector)) {
            revert InvalidTrainerSelector();
        }

        SwarmCoreStorage storage $ = _getSwarmCoreStorage();
        address previousSelector = $.trainerSelector;
        $.trainerSelector = newTrainerSelector;

        emit TrainerSelectorUpdated(previousSelector, newTrainerSelector);
    }

    /// @notice Internal function to update the evaluator selector
    /// @dev Validates interface support before updating storage
    /// @param newEvaluatorSelector The new evaluator selector contract address
    function _updateEvaluatorSelector(address newEvaluatorSelector) internal {
        // Check if the new address supports ISelector interface
        if (!_supportsISelector(newEvaluatorSelector)) {
            revert InvalidEvaluatorSelector();
        }

        SwarmCoreStorage storage $ = _getSwarmCoreStorage();
        address previousSelector = $.evaluatorSelector;
        $.evaluatorSelector = newEvaluatorSelector;

        emit EvaluatorSelectorUpdated(previousSelector, newEvaluatorSelector);
    }

    /// @notice Internal function to update the contribution calculator
    /// @dev Validates interface support before updating storage
    /// @param newContributionCalculator The new contribution calculator contract address
    function _updateContributionCalculator(address newContributionCalculator) internal {
        // Check if the new address supports IContributionCalculator interface
        if (!_supportsIContributionCalculator(newContributionCalculator)) {
            revert InvalidContributionCalculator();
        }

        SwarmCoreStorage storage $ = _getSwarmCoreStorage();
        address previousCalculator = $.contributionCalculator;
        $.contributionCalculator = newContributionCalculator;

        emit ContributionCalculatorUpdated(previousCalculator, newContributionCalculator);
    }

    /// @notice Check if an address supports the ISelector interface
    /// @param addr The address to check
    /// @return True if the address supports ISelector interface, false otherwise
    function _supportsISelector(address addr) internal view returns (bool) {
        if (addr == address(0)) {
            return false;
        }

        try IERC165(addr).supportsInterface(type(ISelector).interfaceId) returns (bool supported) {
            return supported;
        } catch {
            return false;
        }
    }

    /// @notice Check if an address supports the IContributionCalculator interface
    /// @param addr The address to check
    /// @return True if the address supports IContributionCalculator interface, false otherwise
    function _supportsIContributionCalculator(address addr) internal view returns (bool) {
        if (addr == address(0)) {
            return false;
        }

        try IERC165(addr).supportsInterface(type(IContributionCalculator).interfaceId) returns (bool supported) {
            return supported;
        } catch {
            return false;
        }
    }


    /// @notice Returns a pointer to the storage namespace
    /// @dev This function provides access to the namespaced storage
    function _getSwarmCoreStorage() private pure returns (SwarmCoreStorage storage $) {
        bytes32 slot = SWARM_CORE_STORAGE;
        assembly {
            $.slot := slot
        }
    }
}
