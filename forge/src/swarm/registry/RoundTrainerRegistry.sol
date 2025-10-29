// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Initializable} from "@openzeppelin-contracts-upgradeable-5.2.0/proxy/utils/Initializable.sol";

/// @title RoundTrainerRegistry
/// @notice Registry contract for managing trainers per round
contract RoundTrainerRegistry is Initializable {
    /// @dev Structure to store trainer information
    struct TrainerInfo {
        uint256 id; // Trainer's assigned ID
        bytes32 modelHash; // Hash of the trainer's model
        bool rewardsClaimed; // Whether the trainer has claimed their rewards
    }

    /// @dev Structure to store trainers for a specific round
    struct RoundTrainers {
        mapping(address => TrainerInfo) trainers; // Maps trainer address to their info
        uint256 count; // Total number of trainers registered
    }

    /// @dev Storage namespace for RoundTrainerRegistry
    struct RoundTrainerRegistryStorage {
        mapping(uint256 => RoundTrainers) roundTrainers;
    }

    // Storage slot for RoundTrainerRegistry namespace
    bytes32 private constant ROUND_TRAINER_REGISTRY_STORAGE = keccak256("RoundTrainerRegistry.storage");

    /// @notice Emitted when a trainer is registered for a round
    /// @param roundId The round ID
    /// @param trainer The trainer address
    /// @param trainerId The assigned trainer ID
    event TrainerRegistered(uint256 indexed roundId, address indexed trainer, uint256 indexed trainerId);

    /// @notice Emitted when a trainer's model hash is updated
    /// @param roundId The round ID
    /// @param trainer The trainer address
    /// @param modelHash The new model hash
    event ModelHashUpdated(uint256 indexed roundId, address indexed trainer, bytes32 modelHash);

    /// @notice Error thrown when trying to register a zero address trainer
    error InvalidTrainerAddress();

    /// @notice Error thrown when trying to access a non-existent trainer
    error TrainerNotFound(uint256 roundId, address trainer);

    /// @notice Initializes the contract
    /// @dev This function can only be called once during proxy deployment
    function initialize() external virtual initializer {
        __RoundTrainerRegistry_init();
    }

    function __RoundTrainerRegistry_init() internal onlyInitializing {
        // No initialization required for this contract
    }

    /// @notice Register a trainer for a specific round
    /// @dev Internal function to register trainers and assign sequential IDs
    /// @param roundId The round ID
    /// @param trainer The trainer address
    /// @param modelHash The model hash
    /// @return trainerId The assigned trainer ID
    function _registerTrainer(uint256 roundId, address trainer, bytes32 modelHash) internal returns (uint256 trainerId) {
        if (trainer == address(0)) {
            revert InvalidTrainerAddress();
        }

        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        RoundTrainers storage roundTrainers = $.roundTrainers[roundId];
        
        // If this is a new trainer for this round, assign a new ID
        if (roundTrainers.trainers[trainer].id == 0) {
            trainerId = ++roundTrainers.count;
            roundTrainers.trainers[trainer] = TrainerInfo({
                id: trainerId,
                modelHash: modelHash,
                rewardsClaimed: false
            });
            emit TrainerRegistered(roundId, trainer, trainerId);
        } else {
            trainerId = roundTrainers.trainers[trainer].id;
        }
    }

    /// @notice Update a trainer's model hash for a specific round
    /// @dev Internal function to update the model hash of a registered trainer
    /// @param roundId The round ID
    /// @param trainer The trainer address
    /// @param modelHash The new model hash
    function _setModelHash(uint256 roundId, address trainer, bytes32 modelHash) internal {
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        RoundTrainers storage roundTrainers = $.roundTrainers[roundId];
        
        // Check if trainer is registered
        if (roundTrainers.trainers[trainer].id == 0) {
            revert TrainerNotFound(roundId, trainer);
        }
        
        roundTrainers.trainers[trainer].modelHash = modelHash;
        emit ModelHashUpdated(roundId, trainer, modelHash);
    }

    /// @notice Get the ID of a specific trainer for a round
    /// @param roundId The round ID
    /// @param trainer The trainer address
    /// @return The trainer's ID
    function getTrainerId(uint256 roundId, address trainer) public view returns (uint256) {
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        return $.roundTrainers[roundId].trainers[trainer].id;
    }

    /// @notice Get the ID of a specific trainer for a round, throwing if not found
    /// @param roundId The round ID
    /// @param trainer The trainer address
    /// @return The trainer's ID
    function getTrainerIdOrThrow(uint256 roundId, address trainer) public view returns (uint256) {
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        uint256 trainerId = $.roundTrainers[roundId].trainers[trainer].id;
        if (trainerId == 0) {
            revert TrainerNotFound(roundId, trainer);
        }
        return trainerId;
    }

    /// @notice Get the model hash of a specific trainer for a round
    /// @param roundId The round ID
    /// @param trainer The trainer address
    /// @return The trainer's model hash
    function getModelHash(uint256 roundId, address trainer) public view returns (bytes32) {
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        return $.roundTrainers[roundId].trainers[trainer].modelHash;
    }

    /// @notice Get the model hash of a specific trainer for a round, throwing if not found
    /// @param roundId The round ID
    /// @param trainer The trainer address
    /// @return The trainer's model hash
    function getModelHashOrThrow(uint256 roundId, address trainer) public view returns (bytes32) {
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        TrainerInfo storage trainerInfo = $.roundTrainers[roundId].trainers[trainer];
        if (trainerInfo.id == 0) {
            revert TrainerNotFound(roundId, trainer);
        }
        return trainerInfo.modelHash;
    }

    /// @notice Get both ID and model hash of a specific trainer for a round
    /// @param roundId The round ID
    /// @param trainer The trainer address
    /// @return trainerId The trainer's ID
    /// @return modelHash The trainer's model hash
    function getTrainerInfo(uint256 roundId, address trainer) public view returns (uint256 trainerId, bytes32 modelHash) {
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        TrainerInfo storage trainerInfo = $.roundTrainers[roundId].trainers[trainer];
        return (trainerInfo.id, trainerInfo.modelHash);
    }

    /// @notice Get the total number of trainers for a round
    /// @param roundId The round ID
    /// @return The number of trainers registered for the round
    function getTrainerCount(uint256 roundId) public view returns (uint256) {
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        return $.roundTrainers[roundId].count;
    }

    /// @notice Check if a trainer is registered for a round
    /// @param roundId The round ID
    /// @param trainer The trainer address
    /// @return True if the trainer is registered for the round
    function isTrainerRegistered(uint256 roundId, address trainer) public view returns (bool) {
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        return $.roundTrainers[roundId].trainers[trainer].id > 0;
    }

    /// @notice Set the rewards claimed status for a trainer
    /// @param roundId The round ID
    /// @param trainer The trainer address
    function _setClaimedRewards(uint256 roundId, address trainer) internal {
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        $.roundTrainers[roundId].trainers[trainer].rewardsClaimed = true;
    }

    /// @notice Check if a trainer has claimed their rewards for a round
    /// @param roundId The round ID
    /// @param trainer The trainer address
    /// @return True if the trainer has claimed their rewards
    function hasClaimedRewards(uint256 roundId, address trainer) public view returns (bool) {
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        return $.roundTrainers[roundId].trainers[trainer].rewardsClaimed;
    }

    /// @notice Returns a pointer to the storage namespace
    /// @dev This function provides access to the namespaced storage
    function _getRoundTrainerRegistryStorage() private pure returns (RoundTrainerRegistryStorage storage $) {
        bytes32 slot = ROUND_TRAINER_REGISTRY_STORAGE;
        assembly {
            $.slot := slot
        }
    }
}
