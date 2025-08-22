// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ISelector} from "./ISelector.sol";
import {RNG} from "../randomness/RNG.sol";
import {ISeedProvider} from "../randomness/ISeedProvider.sol";
import {IERC165} from "@openzeppelin-contracts-5.2.0/utils/introspection/IERC165.sol";
import {EIP712Upgradeable} from "@openzeppelin-contracts-upgradeable-5.2.0/utils/cryptography/EIP712Upgradeable.sol";

/// @title Random Trainer Sampling Contract
/// @notice Implements probabilistic trainer sampling using a fixed target ratio and RNG
/// @dev Stores only a target ratio and uses RNG to determine selection based on that ratio
contract RandomSampling is ISelector, ISeedProvider, EIP712Upgradeable {

    string private constant _VERSION = "random-sampling-v1.0.0";

    /// @dev The target ratio of trainers to select (as a percentage, 1 ether = 100%)
    uint256 private _targetRatio;

    uint256 constant RATIO_DECIMALS = 10**18;
    
    /// @dev Emitted when the target ratio is updated
    event TargetRatioUpdated(uint256 oldRatio, uint256 newRatio);
    
    /// @dev Error thrown when trying to set an invalid ratio
    error InvalidTargetRatio(uint256 ratio);
    
    /// @dev Initializer sets the initial target ratio
    /// @param targetRatio The initial target ratio (as a percentage, 1 ether = 100%)
    function initialize(uint256 targetRatio) external initializer {
        if (targetRatio > RATIO_DECIMALS) {
            revert InvalidTargetRatio(targetRatio);
        }
        
        __EIP712_init("RandomSampling", _VERSION);
        _targetRatio = targetRatio;
        
        emit TargetRatioUpdated(0, targetRatio);
    }
    
    /// @notice Checks if a trainer is selected based on the target ratio and RNG
    /// @param addr The address of the trainer to check
    /// @param roundId The round ID (used for RNG seed)
    /// @return True if the trainer is selected, false otherwise
    function isSelected(
        address addr,
        uint256 roundId
    ) external view override returns (bool) {
        return _isAddressSelected(addr, roundId);
    }
    
    /// @notice Updates the target ratio for trainer selection
    /// @param newRatio The new target ratio (0-10000, where 10000 = 100%)
    function _setTargetRatio(uint256 newRatio) private {
        if (newRatio > RATIO_DECIMALS) {
            revert InvalidTargetRatio(newRatio);
        }
        
        uint256 oldRatio = _targetRatio;
        _targetRatio = newRatio;
        
        emit TargetRatioUpdated(oldRatio, newRatio);
    }
    
    
    /// @notice Gets the current target ratio
    /// @return The target ratio as a percentage (as a percentage, 1 ether = 100%)
    function getTargetRatio() external view returns (uint256) {
        return _targetRatio;
    }
    
    /// @dev Internal function to check if an address would be selected
    /// @param addr The address to check
    /// @param roundId The round ID
    /// @return True if the address would be selected
    function _isAddressSelected(address addr, uint256 roundId) internal view returns (bool) {
        // Generate a deterministic random number for this trainer and round
        bytes32 seed = _getSeed(roundId);

        uint256 trainerIndex = uint256(keccak256(abi.encodePacked(addr)));
        // Use the first 16 bytes of the seed to generate a random number
        (uint256 randomValue,) = RNG.rand(seed, trainerIndex, RATIO_DECIMALS);
        
        // If the random value is below the target ratio, the trainer is selected
        return randomValue < _targetRatio;
    }

    /// @dev See {IERC165-supportsInterface}
    function supportsInterface(bytes4 interfaceId) external pure returns (bool) {
        return interfaceId == type(ISelector).interfaceId || 
               interfaceId == type(IERC165).interfaceId;
    }

    function getSeed(uint256 roundId) external view override returns (bytes32) {
        return _getSeed(roundId);
    }

    /// @dev The version parameter for the EIP712 domain.
    function _EIP712Version()
        internal
        pure
        override(EIP712Upgradeable)
        returns (string memory)
    {
        return _VERSION;
    }
    
    function _getSeed(uint256 roundId) internal view returns (bytes32) {
        // TODO: Use VRF to generate a seed
        return keccak256(abi.encodePacked(address(this), roundId)); 
    }
}
