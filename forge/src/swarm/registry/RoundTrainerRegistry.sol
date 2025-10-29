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

    /// @dev Configuration for trainer privacy enforcement
    struct TrainerPrivacyConfig {
        uint256 penalty; // Penalty reserved per commitment (in wei)
        uint16 finderRewardBps; // Finder reward expressed in basis points
    }

    /// @dev Aggregator bond accounting
    struct AggregatorBondState {
        uint256 balance; // Total bond deposited by the aggregator
        uint256 reserved; // Portion reserved to cover outstanding commitments
    }

    /// @dev Commitment tracking for trainer privacy
    struct CommitmentInfo {
        bytes32 modelHash;
        uint64 revealDeadline;
        uint256 penalty;
        uint16 finderRewardBps;
        bool revealed;
        bool slashed;
        bool exists;
        address trainer;
        bool pendingCounted;
    }

    /// @dev Structure to store trainers for a specific round
    struct RoundTrainers {
        mapping(address => TrainerInfo) trainers; // Maps trainer address to their info
        uint256 count; // Total number of trainers registered
    }

    /// @dev Storage namespace for RoundTrainerRegistry
    struct RoundTrainerRegistryStorage {
        mapping(uint256 => RoundTrainers) roundTrainers;
        TrainerPrivacyConfig privacyConfig;
        AggregatorBondState aggregatorBond;
        mapping(uint256 => mapping(bytes32 => CommitmentInfo)) commitments;
        mapping(uint256 => uint256) pendingCommitments;
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

    /// @notice Error thrown when the reserved bond is insufficient for the requested amount
    error InsufficientAggregatorBond(uint256 available, uint256 required);

    /// @notice Error thrown when finder reward basis points exceed 100%
    error InvalidFinderReward(uint16 finderRewardBps);

    /// @notice Error thrown when attempting to reuse an existing commitment
    error CommitmentAlreadyRegistered(uint256 roundId, bytes32 commitment);

    /// @notice Error thrown when a commitment cannot be located
    error CommitmentNotFound(uint256 roundId, bytes32 commitment);

    /// @notice Error thrown when attempting to slash a commitment before its reveal deadline
    error CommitmentRevealPending(uint256 roundId, bytes32 commitment, uint64 deadline);

    /// @notice Error thrown when a commitment has already been revealed
    error CommitmentAlreadyRevealed(uint256 roundId, bytes32 commitment);

    /// @notice Error thrown when a commitment has already been slashed
    error CommitmentAlreadySlashed(uint256 roundId, bytes32 commitment);

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

    /// @notice Configure trainer privacy enforcement parameters
    /// @param penalty Penalty amount reserved per commitment (in wei)
    /// @param finderRewardBps Finder reward in basis points (max 10_000)
    function _setTrainerPrivacyConfig(uint256 penalty, uint16 finderRewardBps) internal {
        if (finderRewardBps > 10_000) {
            revert InvalidFinderReward(finderRewardBps);
        }

        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        $.privacyConfig = TrainerPrivacyConfig({
            penalty: penalty,
            finderRewardBps: finderRewardBps
        });
    }

    /// @notice Increase the aggregator bond balance
    /// @param amount Amount of ETH deposited (in wei)
    function _increaseAggregatorBond(uint256 amount) internal {
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        $.aggregatorBond.balance += amount;
    }

    /// @notice Decrease the aggregator bond balance (withdrawal)
    /// @param amount Amount of ETH to withdraw (in wei)
    function _decreaseAggregatorBond(uint256 amount) internal {
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        AggregatorBondState storage bond = $.aggregatorBond;

        if (amount > bond.balance - bond.reserved) {
            revert InsufficientAggregatorBond(bond.balance - bond.reserved, amount);
        }

        bond.balance -= amount;
    }

    /// @notice Register a trainer commitment while preserving privacy
    /// @param roundId The training round identifier
    /// @param commitment The commitment hash (typically hash(trainer, nonce))
    /// @param modelHash The trainer's model hash associated with the commitment
    /// @param revealDeadline Deadline after which the commitment can be slashed
    /// @return trainerId Placeholder trainer identifier (always 0 prior to reveal)
    function _commitTrainerPrivacy(
        uint256 roundId,
        bytes32 commitment,
        bytes32 modelHash,
        uint64 revealDeadline
    ) internal returns (uint256 trainerId) {
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        CommitmentInfo storage info = $.commitments[roundId][commitment];
        if (info.exists) {
            revert CommitmentAlreadyRegistered(roundId, commitment);
        }

        TrainerPrivacyConfig memory config = $.privacyConfig;
        AggregatorBondState storage bond = $.aggregatorBond;

        if (config.penalty > 0) {
            uint256 available = bond.balance - bond.reserved;
            if (available < config.penalty) {
                revert InsufficientAggregatorBond(available, config.penalty);
            }
            bond.reserved += config.penalty;
        }

        info.modelHash = modelHash;
        info.revealDeadline = revealDeadline;
        info.penalty = config.penalty;
        info.finderRewardBps = config.finderRewardBps;
        info.revealed = false;
        info.slashed = false;
        info.exists = true;
        info.trainer = address(0);
        info.pendingCounted = true;

        $.pendingCommitments[roundId] += 1;

        // Trainer ID is unknown until reveal; return 0 as a placeholder to preserve interface compatibility
        return 0;
    }

    /// @notice Reveal a previously committed trainer and register them for the round
    /// @param roundId The training round identifier
    /// @param trainer The trainer address being revealed
    /// @param nonce Nonce that, combined with trainer address, reproduces the commitment hash
    /// @return trainerId The trainer identifier assigned upon registration
    /// @return commitment The derived commitment hash that was revealed
    function _revealTrainerPrivacy(
        uint256 roundId,
        address trainer,
        bytes calldata nonce
    ) internal returns (uint256 trainerId, bytes32 commitment) {
        if (trainer == address(0)) {
            revert InvalidTrainerAddress();
        }

        commitment = keccak256(abi.encodePacked(trainer, nonce));
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        CommitmentInfo storage info = $.commitments[roundId][commitment];

        if (!info.exists) {
            revert CommitmentNotFound(roundId, commitment);
        }
        if (info.revealed) {
            revert CommitmentAlreadyRevealed(roundId, commitment);
        }

        info.revealed = true;
        info.trainer = trainer;

        uint256 penalty = info.penalty;
        if (penalty > 0 && !info.slashed) {
            AggregatorBondState storage bond = $.aggregatorBond;
            if (bond.reserved < penalty) {
                revert InsufficientAggregatorBond(bond.reserved, penalty);
            }
            bond.reserved -= penalty;
        }

        if (info.pendingCounted) {
            info.pendingCounted = false;
            uint256 pending = $.pendingCommitments[roundId];
            if (pending > 0) {
                $.pendingCommitments[roundId] = pending - 1;
            }
        }

        trainerId = _registerTrainer(roundId, trainer, info.modelHash);
        return (trainerId, commitment);
    }

    /// @notice Slash a commitment that missed its reveal deadline
    /// @param roundId The training round identifier
    /// @param commitment The commitment hash to slash
    /// @param finder Address initiating the slash (eligible for finder reward)
    /// @return penalty The penalty amount deducted from the aggregator bond
    /// @return finderReward The portion of the penalty awarded to the finder
    function _slashTrainerCommitment(
        uint256 roundId,
        bytes32 commitment,
        address finder
    ) internal returns (uint256 penalty, uint256 finderReward) {
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        CommitmentInfo storage info = $.commitments[roundId][commitment];

        if (!info.exists) {
            revert CommitmentNotFound(roundId, commitment);
        }
        if (info.revealed) {
            revert CommitmentAlreadyRevealed(roundId, commitment);
        }
        if (info.slashed) {
            revert CommitmentAlreadySlashed(roundId, commitment);
        }
        if (block.timestamp <= info.revealDeadline) {
            revert CommitmentRevealPending(roundId, commitment, info.revealDeadline);
        }

        info.slashed = true;

        penalty = info.penalty;
        finderReward = (penalty * info.finderRewardBps) / 10_000;
        if (finder == address(0)) {
            finderReward = 0;
        }

        if (penalty > 0) {
            AggregatorBondState storage bond = $.aggregatorBond;
            if (bond.reserved < penalty) {
                revert InsufficientAggregatorBond(bond.reserved, penalty);
            }
            bond.reserved -= penalty;
            if (bond.balance < penalty) {
                revert InsufficientAggregatorBond(bond.balance, penalty);
            }
            bond.balance -= penalty;
        }

        if (info.pendingCounted) {
            info.pendingCounted = false;
            uint256 pending = $.pendingCommitments[roundId];
            if (pending > 0) {
                $.pendingCommitments[roundId] = pending - 1;
            }
        }

        return (penalty, finderReward);
    }

    /// @notice Return the current aggregator bond state
    /// @return balance Total bond posted by the aggregator
    /// @return reserved Portion of the bond reserved against active commitments
    function getAggregatorBondState() public view returns (uint256 balance, uint256 reserved) {
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        AggregatorBondState storage bond = $.aggregatorBond;
        return (bond.balance, bond.reserved);
    }

    /// @notice Return the number of pending trainer commitments awaiting reveal or slash
    /// @param roundId The round identifier
    /// @return The count of outstanding commitments
    function getPendingCommitmentCount(uint256 roundId) public view returns (uint256) {
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        return $.pendingCommitments[roundId];
    }

    /// @notice Return total trainer slots including pending commitments
    /// @param roundId The round identifier
    /// @return The combined count of registered trainers and outstanding commitments
    function getTrainerSlotCount(uint256 roundId) public view returns (uint256) {
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        return $.roundTrainers[roundId].count + $.pendingCommitments[roundId];
    }
}
