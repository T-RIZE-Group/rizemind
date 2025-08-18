// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ITrainerSelection} from "./ITrainerSampling.sol";
import {RNG} from "../randomness/RNG.sol";
import {ISeedProvider} from "../randomness/ISeedProvider.sol";

/// @title Random Trainer Sampling Contract
/// @notice Implements probabilistic trainer sampling using a fixed target ratio and RNG
/// @dev Stores only a target ratio and uses RNG to determine selection based on that ratio
abstract contract RandomTrainerSampling is ITrainerSelection, ISeedProvider {
    
    /// @dev The target ratio of trainers to select (as a percentage, 0-10000 where 10000 = 100%)
    uint256 public targetRatio;

    uint256 constant RATIO_DECIMALS = 10**18;
    
    /// @dev Emitted when the target ratio is updated
    event TargetRatioUpdated(uint256 oldRatio, uint256 newRatio);
    
    /// @dev Error thrown when trying to set an invalid ratio
    error InvalidTargetRatio(uint256 ratio);
    
    /// @dev Constructor sets the initial target ratio and owner
    /// @param _targetRatio The initial target ratio (0-10000, where 10000 = 100%)
    constructor(uint256 _targetRatio) {
        if (_targetRatio > RATIO_DECIMALS) {
            revert InvalidTargetRatio(_targetRatio);
        }
        
        targetRatio = _targetRatio;
        
        emit TargetRatioUpdated(0, _targetRatio);
    }
    
    /// @notice Checks if a trainer is selected based on the target ratio and RNG
    /// @param addr The address of the trainer to check
    /// @param roundId The round ID (used for RNG seed)
    /// @return True if the trainer is selected, false otherwise
    function isTrainerSelected(
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
        
        uint256 oldRatio = targetRatio;
        targetRatio = newRatio;
        
        emit TargetRatioUpdated(oldRatio, newRatio);
    }
    
    
    /// @notice Gets the current target ratio
    /// @return The target ratio as a percentage (0-10000, where 10000 = 100%)
    function getTargetRatio() external view returns (uint256) {
        return targetRatio;
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
        return randomValue <= targetRatio;
    }

    function _getSeed(uint256 roundId) internal view virtual returns (bytes32);
}
