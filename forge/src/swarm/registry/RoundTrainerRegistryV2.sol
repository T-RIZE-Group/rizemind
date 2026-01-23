// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Initializable} from "@openzeppelin-contracts-upgradeable-5.2.0/proxy/utils/Initializable.sol";

/// @title RoundTrainerRegistry
/// @notice Registry contract for managing trainers per round
contract RoundTrainerRegistryV2 is Initializable {
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

    struct AggregatorBondState {
        uint256 balance;
        uint256 reserved;
    }

    struct CommitmentState {
        bytes32 modelHash;
        uint64 revealDeadline;
        address trainer;
        uint256 penalty;
        bool active;
        bool revealed;
        bool slashed;
        bool exists;
    }

    struct RoundPrivacyState {
        mapping(bytes32 => CommitmentState) commitments;
        uint256 pendingCount;
    }

    struct TrainerPrivacyStorage {
        AggregatorBondState bond;
        uint256 penalty;
        uint16 finderRewardBps;
        mapping(uint256 => RoundPrivacyState) rounds;
    }

    // Maximum allowed time window (in seconds) for aggregator reveals.
    uint64 private constant MAX_REVEAL_DEADLINE = 2 days;

    // Storage slots for namespaced storage
    bytes32 private constant ROUND_TRAINER_REGISTRY_STORAGE =
        keccak256("RoundTrainerRegistry.storage");
    bytes32 private constant TRAINER_PRIVACY_STORAGE =
        keccak256("RoundTrainerRegistry.privacy.storage");

    /// @notice Emitted when a trainer is registered for a round
    /// @param roundId The round ID
    /// @param trainer The trainer address
    /// @param trainerId The assigned trainer ID
    event TrainerRegistered(
        uint256 indexed roundId,
        address indexed trainer,
        uint256 indexed trainerId
    );

    /// @notice Emitted when a trainer's model hash is updated
    /// @param roundId The round ID
    /// @param trainer The trainer address
    /// @param modelHash The new model hash
    event ModelHashUpdated(
        uint256 indexed roundId,
        address indexed trainer,
        bytes32 modelHash
    );

    /// @notice Error thrown when trying to register a zero address trainer
    error InvalidTrainerAddress();

    /// @notice Error thrown when trying to access a non-existent trainer
    error TrainerNotFound(uint256 roundId, address trainer);

    /// @notice Error thrown when finder reward basis points exceed 100%
    error InvalidFinderRewardBps(uint16 finderRewardBps);

    /// @notice Error thrown when the reveal deadline exceeds the allowed horizon
    error RevealDeadlineTooLong(uint256 roundId, uint64 deadline, uint64 maxAllowed);

    /// @notice Error thrown when attempting to commit a duplicate privacy commitment
    error CommitmentAlreadyRegistered(uint256 roundId, bytes32 commitment);

    /// @notice Error thrown when attempting to operate on a non-existent commitment
    error CommitmentNotFound(uint256 roundId, bytes32 commitment);

    /// @notice Error thrown when attempting to reveal an already revealed commitment
    error CommitmentAlreadyRevealed(uint256 roundId, bytes32 commitment);

    /// @notice Error thrown when attempting to slash an already slashed commitment
    error CommitmentAlreadySlashed(uint256 roundId, bytes32 commitment);

    /// @notice Error thrown when attempting to slash before the reveal deadline has elapsed
    error RevealDeadlineNotReached(
        uint256 roundId,
        bytes32 commitment,
        uint64 deadline
    );

    /// @notice Error thrown when the aggregator bond lacks sufficient free balance
    error AggregatorBondInsufficient(uint256 requested, uint256 available);

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
    function _registerTrainer(
        uint256 roundId,
        address trainer,
        bytes32 modelHash
    ) internal returns (uint256 trainerId) {
        if (trainer == address(0)) {
            revert InvalidTrainerAddress();
        }

        RoundTrainerRegistryStorage
            storage $ = _getRoundTrainerRegistryStorage();
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
    function _setModelHash(
        uint256 roundId,
        address trainer,
        bytes32 modelHash
    ) internal {
        RoundTrainerRegistryStorage
            storage $ = _getRoundTrainerRegistryStorage();
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
    function getTrainerId(
        uint256 roundId,
        address trainer
    ) public view returns (uint256) {
        RoundTrainerRegistryStorage
            storage $ = _getRoundTrainerRegistryStorage();
        return $.roundTrainers[roundId].trainers[trainer].id;
    }

    /// @notice Get the ID of a specific trainer for a round, throwing if not found
    /// @param roundId The round ID
    /// @param trainer The trainer address
    /// @return The trainer's ID
    function getTrainerIdOrThrow(
        uint256 roundId,
        address trainer
    ) public view returns (uint256) {
        RoundTrainerRegistryStorage
            storage $ = _getRoundTrainerRegistryStorage();
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
    function getModelHash(
        uint256 roundId,
        address trainer
    ) public view returns (bytes32) {
        RoundTrainerRegistryStorage
            storage $ = _getRoundTrainerRegistryStorage();
        return $.roundTrainers[roundId].trainers[trainer].modelHash;
    }

    /// @notice Get the model hash of a specific trainer for a round, throwing if not found
    /// @param roundId The round ID
    /// @param trainer The trainer address
    /// @return The trainer's model hash
    function getModelHashOrThrow(
        uint256 roundId,
        address trainer
    ) public view returns (bytes32) {
        RoundTrainerRegistryStorage
            storage $ = _getRoundTrainerRegistryStorage();
        TrainerInfo storage trainerInfo = $.roundTrainers[roundId].trainers[
            trainer
        ];
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
    function getTrainerInfo(
        uint256 roundId,
        address trainer
    ) public view returns (uint256 trainerId, bytes32 modelHash) {
        RoundTrainerRegistryStorage
            storage $ = _getRoundTrainerRegistryStorage();
        TrainerInfo storage trainerInfo = $.roundTrainers[roundId].trainers[
            trainer
        ];
        return (trainerInfo.id, trainerInfo.modelHash);
    }

    /// @notice Get the total number of trainers for a round
    /// @param roundId The round ID
    /// @return The number of trainers registered for the round
    function getTrainerCount(uint256 roundId) public view returns (uint256) {
        RoundTrainerRegistryStorage
            storage $ = _getRoundTrainerRegistryStorage();
        return $.roundTrainers[roundId].count;
    }

    /// @notice Check if a trainer is registered for a round
    /// @param roundId The round ID
    /// @param trainer The trainer address
    /// @return True if the trainer is registered for the round
    function isTrainerRegistered(
        uint256 roundId,
        address trainer
    ) public view returns (bool) {
        RoundTrainerRegistryStorage
            storage $ = _getRoundTrainerRegistryStorage();
        return $.roundTrainers[roundId].trainers[trainer].id > 0;
    }

    /// @notice Set the rewards claimed status for a trainer
    /// @param roundId The round ID
    /// @param trainer The trainer address
    function _setClaimedRewards(uint256 roundId, address trainer) internal {
        RoundTrainerRegistryStorage
            storage $ = _getRoundTrainerRegistryStorage();
        $.roundTrainers[roundId].trainers[trainer].rewardsClaimed = true;
    }

    /// @notice Check if a trainer has claimed their rewards for a round
    /// @param roundId The round ID
    /// @param trainer The trainer address
    /// @return True if the trainer has claimed their rewards
    function hasClaimedRewards(
        uint256 roundId,
        address trainer
    ) public view returns (bool) {
        RoundTrainerRegistryStorage
            storage $ = _getRoundTrainerRegistryStorage();
        return $.roundTrainers[roundId].trainers[trainer].rewardsClaimed;
    }

    /// @notice Configure the trainer privacy penalty and finder reward share
    /// @param penalty The amount of bond to reserve per commitment
    /// @param finderRewardBps Finder reward share expressed in basis points (max 10_000)
    function _setTrainerPrivacyConfig(
        uint256 penalty,
        uint16 finderRewardBps
    ) internal {
        if (finderRewardBps > 10_000) {
            revert InvalidFinderRewardBps(finderRewardBps);
        }

        TrainerPrivacyStorage storage $ = _getTrainerPrivacyStorage();
        $.penalty = penalty;
        $.finderRewardBps = finderRewardBps;
    }

    /// @notice Increase the aggregator bond balance by an amount
    /// @param amount Amount of wei added to the bond balance
    function _increaseAggregatorBond(uint256 amount) internal {
        TrainerPrivacyStorage storage $ = _getTrainerPrivacyStorage();
        $.bond.balance += amount;
    }

    /// @notice Decrease the aggregator bond balance by an amount
    /// @param amount Amount of wei to remove from the bond balance
    function _decreaseAggregatorBond(uint256 amount) internal {
        TrainerPrivacyStorage storage $ = _getTrainerPrivacyStorage();
        AggregatorBondState storage bond = $.bond;

        uint256 available = bond.balance - bond.reserved;
        if (amount > available) {
            revert AggregatorBondInsufficient(amount, available);
        }

        bond.balance -= amount;
    }

    /// @notice Return the current aggregator bond balance and reserved amount
    function getAggregatorBondState()
        public
        view
        returns (uint256 balance, uint256 reserved)
    {
        TrainerPrivacyStorage storage $ = _getTrainerPrivacyStorage();
        AggregatorBondState storage bond = $.bond;
        balance = bond.balance;
        reserved = bond.reserved;
    }

    /// @notice Commit to a trainer using privacy mode
    /// @dev Aggregator bond is reserved until reveal or slash; long deadlines therefore lock capital.
    /// @param roundId The round identifier
    /// @param commitment The commitment hash binding trainer and nonce
    /// @param modelHash The trainer's model hash recorded on reveal
    /// @param revealDeadline Deadline timestamp after which the commitment can be slashed
    /// @return pendingCommitments The updated number of pending commitments for the round
    function _commitTrainerPrivacy(
        uint256 roundId,
        bytes32 commitment,
        bytes32 modelHash,
        uint64 revealDeadline
    ) internal returns (uint256 pendingCommitments) {
        uint64 maxDeadline = uint64(block.timestamp + MAX_REVEAL_DEADLINE);
        if (revealDeadline > maxDeadline) {
            revert RevealDeadlineTooLong(roundId, revealDeadline, maxDeadline);
        }

        TrainerPrivacyStorage storage $ = _getTrainerPrivacyStorage();
        RoundPrivacyState storage roundPrivacy = $.rounds[roundId];
        CommitmentState storage state = roundPrivacy.commitments[commitment];

        if (state.exists) {
            revert CommitmentAlreadyRegistered(roundId, commitment);
        }

        uint256 penalty = $.penalty;
        if (penalty > 0) {
            AggregatorBondState storage bond = $.bond;
            uint256 available = bond.balance - bond.reserved;
            if (penalty > available) {
                revert AggregatorBondInsufficient(penalty, available);
            }
            bond.reserved += penalty;
        }

        state.modelHash = modelHash;
        state.revealDeadline = revealDeadline;
        state.trainer = address(0);
        state.penalty = penalty;
        state.active = true;
        state.revealed = false;
        state.slashed = false;
        state.exists = true;

        pendingCommitments = ++roundPrivacy.pendingCount;
    }

    /// @notice Reveal a trainer's identity for a previously committed contribution
    /// @param roundId The training round identifier
    /// @param trainer The trainer being revealed
    /// @param nonce The nonce used within the commitment
    /// @return trainerId The trainer identifier assigned within the round
    /// @return commitment The resolved commitment hash
    function _revealTrainerPrivacy(
        uint256 roundId,
        address trainer,
        bytes calldata nonce
    ) internal returns (uint256 trainerId, bytes32 commitment) {
        commitment = keccak256(abi.encodePacked(trainer, nonce));

        TrainerPrivacyStorage storage $ = _getTrainerPrivacyStorage();
        RoundPrivacyState storage roundPrivacy = $.rounds[roundId];
        CommitmentState storage state = roundPrivacy.commitments[commitment];

        if (!state.exists) {
            revert CommitmentNotFound(roundId, commitment);
        }
        if (state.revealed) {
            revert CommitmentAlreadyRevealed(roundId, commitment);
        }

        if (state.active) {
            if (state.penalty > 0) {
                AggregatorBondState storage bond = $.bond;
                if (bond.reserved < state.penalty) {
                    revert AggregatorBondInsufficient(
                        state.penalty,
                        bond.reserved
                    );
                }
                bond.reserved -= state.penalty;
            }
            if (roundPrivacy.pendingCount > 0) {
                roundPrivacy.pendingCount -= 1;
            }
            state.active = false;
        }

        state.revealed = true;
        state.trainer = trainer;

        trainerId = _registerTrainer(roundId, trainer, state.modelHash);
    }

    /// @notice Slash a commitment whose reveal deadline has elapsed
    /// @param roundId The training round identifier
    /// @param commitment The commitment hash to slash
    /// @param finder Address that triggered the slashing action
    /// @return penalty The penalty deducted from the aggregator bond
    /// @return finderReward The finder reward computed from the penalty
    function _slashAggregatorBond(
        uint256 roundId,
        bytes32 commitment,
        address finder
    ) internal returns (uint256 penalty, uint256 finderReward) {
        finder; // silence unused parameter warning until utilized
        TrainerPrivacyStorage storage $ = _getTrainerPrivacyStorage();
        RoundPrivacyState storage roundPrivacy = $.rounds[roundId];
        CommitmentState storage state = roundPrivacy.commitments[commitment];

        if (!state.exists) {
            revert CommitmentNotFound(roundId, commitment);
        }
        if (state.slashed) {
            revert CommitmentAlreadySlashed(roundId, commitment);
        }
        if (state.revealed) {
            revert CommitmentAlreadyRevealed(roundId, commitment);
        }
        if (block.timestamp <= state.revealDeadline) {
            revert RevealDeadlineNotReached(
                roundId,
                commitment,
                state.revealDeadline
            );
        }

        penalty = state.penalty;

        if (state.active) {
            if (penalty > 0) {
                AggregatorBondState storage bond = $.bond;
                if (bond.reserved < penalty) {
                    revert AggregatorBondInsufficient(penalty, bond.reserved);
                }
                bond.reserved -= penalty;
                if (bond.balance < penalty) {
                    revert AggregatorBondInsufficient(penalty, bond.balance);
                }
                bond.balance -= penalty;
            }
            if (roundPrivacy.pendingCount > 0) {
                roundPrivacy.pendingCount -= 1;
            }
            state.active = false;
        }

        state.slashed = true;

        finderReward = (penalty * $.finderRewardBps) / 10_000;
    }

    /// @notice Return the number of pending commitments awaiting reveal for a round
    /// @param roundId The round identifier
    function getPendingCommitmentCount(
        uint256 roundId
    ) public view returns (uint256) {
        TrainerPrivacyStorage storage $ = _getTrainerPrivacyStorage();
        return $.rounds[roundId].pendingCount;
    }

    /// @notice Return the current trainer privacy configuration
    function getTrainerPrivacyConfig()
        public
        view
        returns (uint256 penalty, uint16 finderRewardBps)
    {
        TrainerPrivacyStorage storage $ = _getTrainerPrivacyStorage();
        penalty = $.penalty;
        finderRewardBps = $.finderRewardBps;
    }

    /// @notice Returns a pointer to the storage namespace
    /// @dev This function provides access to the namespaced storage
    function _getRoundTrainerRegistryStorage()
        private
        pure
        returns (RoundTrainerRegistryStorage storage $)
    {
        bytes32 slot = ROUND_TRAINER_REGISTRY_STORAGE;
        assembly {
            $.slot := slot
        }
    }

    /// @notice Returns a pointer to the trainer privacy storage namespace
    function _getTrainerPrivacyStorage()
        private
        pure
        returns (TrainerPrivacyStorage storage $)
    {
        bytes32 slot = TRAINER_PRIVACY_STORAGE;
        assembly {
            $.slot := slot
        }
    }
}
