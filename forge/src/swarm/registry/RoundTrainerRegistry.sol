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
        uint256 count; // Total number of trainers registered (revealed or pending)
    }

    /// @dev Structure storing commitment data for privacy-aware registrations
    struct TrainerCommitment {
        uint256 trainerId; // Reserved trainer ID
        bytes32 modelHash; // Hash of the trainer's model
        uint64 revealDeadline; // Deadline for aggregator reveal
        uint64 committedAt; // Timestamp when commitment was registered
        uint256 penalty; // Penalty amount reserved against aggregator bond
        bool revealed; // Whether the commitment has been revealed
        bool slashed; // Whether the commitment incurred a penalty
        bytes32 nonceHash; // Hash of the nonce revealed on-chain
    }

    /// @dev Storage namespace for RoundTrainerRegistry
    struct RoundTrainerRegistryStorage {
        mapping(uint256 => RoundTrainers) roundTrainers;
        mapping(uint256 => mapping(bytes32 => TrainerCommitment)) commitments; // roundId => commitment => data
        mapping(uint256 => mapping(uint256 => bytes32)) commitmentByTrainerId; // roundId => trainerId => commitment
        uint256 aggregatorBondBalance; // Total aggregator bond held by the contract
        uint256 aggregatorBondReserved; // Portion of the bond reserved for active commitments
        uint256 defaultRevealPenalty; // Penalty applied per commitment when reveal deadline missed
        uint16 finderRewardBps; // Finder reward share (basis points)
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

    event TrainerCommitted(uint256 indexed roundId, bytes32 indexed commitment, uint256 indexed trainerId, bytes32 modelHash, uint64 revealDeadline, uint256 penalty);
    event TrainerRevealed(uint256 indexed roundId, address indexed trainer, bytes32 indexed commitment, uint256 trainerId, bytes32 nonceHash, bool slashed);
    event TrainerCommitmentSlashed(uint256 indexed roundId, bytes32 indexed commitment, uint256 penalty, uint256 finderReward, address finder);
    event PrivacyConfigUpdated(uint256 penalty, uint16 finderRewardBps);
    event AggregatorBondChanged(uint256 balance);
    event AggregatorBondReserved(uint256 reserved);

    /// @notice Error thrown when trying to register a zero address trainer
    error InvalidTrainerAddress();

    /// @notice Error thrown when trying to access a non-existent trainer
    error TrainerNotFound(uint256 roundId, address trainer);

    /// @notice Error thrown when a trainer is already registered for the round
    error TrainerAlreadyRegistered(uint256 roundId, address trainer);

    /// @notice Error thrown when a commitment hash is invalid
    error InvalidCommitment();

    /// @notice Error thrown when attempting to reuse an existing commitment
    error CommitmentAlreadyExists(uint256 roundId, bytes32 commitment);

    /// @notice Error thrown when a commitment does not exist
    error CommitmentNotFound(uint256 roundId, bytes32 commitment);

    /// @notice Error thrown when attempting to reveal an already revealed commitment
    error CommitmentAlreadyRevealed(uint256 roundId, bytes32 commitment);

    /// @notice Error thrown when attempting to slash an already slashed commitment
    error CommitmentAlreadySlashed(uint256 roundId, bytes32 commitment);

    /// @notice Error thrown when attempting to slash when no penalty is configured
    error NoPenaltyToSlash(uint256 roundId, bytes32 commitment);

    /// @notice Error thrown when attempting to reveal before the deadline has passed (for slashing)
    error RevealDeadlineNotElapsed(uint256 roundId, bytes32 commitment, uint64 deadline);

    /// @notice Error thrown when aggregator bond is insufficient to reserve or slash
    error AggregatorBondInsufficient(uint256 required, uint256 available);

    /// @notice Error thrown when attempting to withdraw more bond than is free
    error WithdrawExceedsFreeBond(uint256 requested, uint256 freeAmount);

    /// @notice Error thrown when the reveal deadline provided is invalid
    error InvalidRevealDeadline(uint64 deadline);

    /// @notice Error thrown when finder reward configuration exceeds 100%
    error InvalidFinderRewardBps(uint16 value);

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

    function getCommitmentInfo(uint256 roundId, bytes32 commitment) public view returns (TrainerCommitment memory info) {
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        info = $.commitments[roundId][commitment];
    }

    function getCommitmentInfoOrThrow(uint256 roundId, bytes32 commitment) public view returns (TrainerCommitment memory info) {
        info = getCommitmentInfo(roundId, commitment);
        if (info.trainerId == 0) {
            revert CommitmentNotFound(roundId, commitment);
        }
    }

    function getCommitmentByTrainerId(uint256 roundId, uint256 trainerId) public view returns (bytes32 commitment) {
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        commitment = $.commitmentByTrainerId[roundId][trainerId];
    }

    function getAggregatorBondState() public view returns (uint256 balance, uint256 reserved) {
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        balance = $.aggregatorBondBalance;
        reserved = $.aggregatorBondReserved;
    }

    function getPrivacyConfig() public view returns (uint256 penalty, uint16 finderRewardBps) {
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        penalty = $.defaultRevealPenalty;
        finderRewardBps = $.finderRewardBps;
    }

    function _setTrainerPrivacyConfig(uint256 penalty, uint16 finderRewardBps) internal {
        if (finderRewardBps > 10_000) {
            revert InvalidFinderRewardBps(finderRewardBps);
        }
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        $.defaultRevealPenalty = penalty;
        $.finderRewardBps = finderRewardBps;
        emit PrivacyConfigUpdated(penalty, finderRewardBps);
    }

    function _increaseAggregatorBond(uint256 amount) internal {
        if (amount == 0) {
            return;
        }
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        $.aggregatorBondBalance += amount;
        emit AggregatorBondChanged($.aggregatorBondBalance);
    }

    function _decreaseAggregatorBond(uint256 amount) internal {
        if (amount == 0) {
            return;
        }
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        uint256 freeBond = _freeAggregatorBond($);
        if (amount > freeBond) {
            revert WithdrawExceedsFreeBond(amount, freeBond);
        }
        $.aggregatorBondBalance -= amount;
        emit AggregatorBondChanged($.aggregatorBondBalance);
    }

    function _commitTrainerPrivacy(uint256 roundId, bytes32 commitment, bytes32 modelHash, uint64 revealDeadline) internal returns (uint256 trainerId) {
        if (commitment == bytes32(0)) {
            revert InvalidCommitment();
        }
        if (revealDeadline <= block.timestamp) {
            revert InvalidRevealDeadline(revealDeadline);
        }
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        TrainerCommitment storage info = $.commitments[roundId][commitment];
        if (info.trainerId != 0) {
            revert CommitmentAlreadyExists(roundId, commitment);
        }
        RoundTrainers storage roundTrainers = $.roundTrainers[roundId];
        trainerId = ++roundTrainers.count;
        uint256 penalty = $.defaultRevealPenalty;
        if (penalty > 0) {
            uint256 freeBond = _freeAggregatorBond($);
            if (penalty > freeBond) {
                revert AggregatorBondInsufficient(penalty, freeBond);
            }
            $.aggregatorBondReserved += penalty;
            emit AggregatorBondReserved($.aggregatorBondReserved);
        }
        info.trainerId = trainerId;
        info.modelHash = modelHash;
        info.revealDeadline = revealDeadline;
        info.committedAt = uint64(block.timestamp);
        info.penalty = penalty;
        info.revealed = false;
        info.slashed = false;
        info.nonceHash = bytes32(0);
        $.commitmentByTrainerId[roundId][trainerId] = commitment;
        emit TrainerCommitted(roundId, commitment, trainerId, modelHash, revealDeadline, penalty);
    }

    function _revealTrainerPrivacy(uint256 roundId, address trainer, bytes calldata nonce) internal returns (uint256 trainerId, bytes32 commitment) {
        if (trainer == address(0)) {
            revert InvalidTrainerAddress();
        }
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        RoundTrainers storage roundTrainers = $.roundTrainers[roundId];
        if (roundTrainers.trainers[trainer].id != 0) {
            revert TrainerAlreadyRegistered(roundId, trainer);
        }
        commitment = keccak256(abi.encodePacked(trainer, nonce));
        TrainerCommitment storage info = $.commitments[roundId][commitment];
        if (info.trainerId == 0) {
            revert CommitmentNotFound(roundId, commitment);
        }
        if (info.revealed) {
            revert CommitmentAlreadyRevealed(roundId, commitment);
        }
        trainerId = info.trainerId;
        roundTrainers.trainers[trainer] = TrainerInfo({
            id: trainerId,
            modelHash: info.modelHash,
            rewardsClaimed: false
        });
        emit TrainerRegistered(roundId, trainer, trainerId);
        if (info.penalty > 0) {
            $.aggregatorBondReserved -= info.penalty;
            emit AggregatorBondReserved($.aggregatorBondReserved);
            info.penalty = 0;
        }
        info.revealed = true;
        info.nonceHash = keccak256(nonce);
        emit TrainerRevealed(roundId, trainer, commitment, trainerId, info.nonceHash, info.slashed);
    }

    function _slashTrainerCommitment(uint256 roundId, bytes32 commitment, address finder) internal returns (uint256 penalty, uint256 finderReward) {
        RoundTrainerRegistryStorage storage $ = _getRoundTrainerRegistryStorage();
        TrainerCommitment storage info = $.commitments[roundId][commitment];
        if (info.trainerId == 0) {
            revert CommitmentNotFound(roundId, commitment);
        }
        if (info.revealed) {
            revert CommitmentAlreadyRevealed(roundId, commitment);
        }
        if (info.slashed) {
            revert CommitmentAlreadySlashed(roundId, commitment);
        }
        if (info.revealDeadline == 0 || block.timestamp <= info.revealDeadline) {
            revert RevealDeadlineNotElapsed(roundId, commitment, info.revealDeadline);
        }
        penalty = info.penalty;
        if (penalty == 0) {
            revert NoPenaltyToSlash(roundId, commitment);
        }
        if (penalty > $.aggregatorBondReserved) {
            revert AggregatorBondInsufficient(penalty, $.aggregatorBondReserved);
        }
        $.aggregatorBondReserved -= penalty;
        emit AggregatorBondReserved($.aggregatorBondReserved);
        if (penalty > $.aggregatorBondBalance) {
            revert AggregatorBondInsufficient(penalty, $.aggregatorBondBalance);
        }
        $.aggregatorBondBalance -= penalty;
        emit AggregatorBondChanged($.aggregatorBondBalance);
        uint16 finderRewardBps = $.finderRewardBps;
        finderReward = finderRewardBps == 0 ? 0 : (penalty * finderRewardBps) / 10_000;
        info.penalty = 0;
        info.slashed = true;
        emit TrainerCommitmentSlashed(roundId, commitment, penalty, finderReward, finder);
    }

    function _freeAggregatorBond(RoundTrainerRegistryStorage storage $) private view returns (uint256) {
        uint256 balance = $.aggregatorBondBalance;
        uint256 reserved = $.aggregatorBondReserved;
        if (reserved >= balance) {
            return 0;
        }
        return balance - reserved;
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
