// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {EIP712Upgradeable} from "@openzeppelin-contracts-upgradeable-5.2.0/utils/cryptography/EIP712Upgradeable.sol";
import {IERC165} from "@openzeppelin-contracts-5.2.0/utils/introspection/IERC165.sol";
import {IAccessControl} from "../access/IAccessControl.sol";
import {RoundTraining} from "../training/RoundTraining.sol";
import {CertificateRegistry} from "./registry/CertificateRegistry.sol";
import {SwarmCore} from "./registry/SwarmCore.sol";
import {ISelector} from "../sampling/ISelector.sol";
import {TaskAssignment} from "../scheduling/TaskAssignment.sol";
import {BaseTrainingPhases} from "../training/BaseTrainingPhases.sol";
import {RoundTrainerRegistryV2} from "./registry/RoundTrainerRegistryV2.sol";
import {RoundEvaluatorRegistry} from "./registry/RoundEvaluatorRegistry.sol";
import {ContributionCalculator} from "../contribution/ContributionCalculator.sol";
import {ICompensation} from "../compensation/types.sol";
import {TrainerContributed} from "../contribution/types.sol";

/**
 * @title SwarmV2
 * @notice SwarmV2 coordinates federated training rounds with optional privacy-preserving
 *     trainer commitments and bond-backed enforcement for aggregator-managed reveals.
 *
 * SwarmV2 keeps the Swarm lifecycle (round management, access control integration,
 * contribution tracking, and payout distribution) while extending the public API with
 * privacy tooling:
 * - Aggregators can toggle privacy mode to submit trainer commitments on behalf of
 *   participants and reveal their identities after the training phase.
 * - A configurable aggregator bond backs each commitment, enabling finder rewards and
 *   slashing when reveals miss their deadlines.
 * - Public helpers expose bond balances, privacy configuration, and commitment status so
 *   off-chain automation can monitor late reveals and trigger slashing.
 */
contract SwarmV2 is
    EIP712Upgradeable,
    RoundTraining,
    BaseTrainingPhases,
    CertificateRegistry,
    RoundTrainerRegistryV2,
    RoundEvaluatorRegistry,
    TaskAssignment,
    SwarmCore
{
    string private constant _VERSION = "swarm-v2.0.0";

    error ForbiddenRound(uint256 roundId);
    error NotIdle();
    error NotTrainingPhase();
    error NotEvaluatorRegistrationPhase();
    error NotAssignedTo(uint256 roundId, uint256 evalId, address evaluator);
    error NotEvaluationPhase();
    error WrongInitialization();
    error NotAggregator();
    error NotTrainer();
    error NotEvaluator();
    error RewardsAlreadyClaimed(uint256 roundId, address trainer);
    error PrivacyModeDisabled();
    error PrivacyModeEnabled();
    error ZeroBondAmount();
    error InvalidRecipient();
    error RevealNotAvailable();
    error TransferFailed(address to, uint256 amount);

    event AggregatorBondDeposited(
        address indexed aggregator,
        uint256 amount,
        uint256 newBalance
    );
    event AggregatorBondWithdrawn(
        address indexed aggregator,
        address indexed recipient,
        uint256 amount,
        uint256 newBalance
    );
    event TrainerPrivacyModeUpdated(bool enabled);

    struct SwarmV2InitializeParams {
        string name;
        address initialTrainerSelector;
        address initialEvaluatorSelector;
        address initialContributionCalculator;
        address initialAccessControl;
        address initialCompensation;
        BaseTrainingPhases.TrainingPhaseConfiguration trainingPhaseConfiguration;
        BaseTrainingPhases.EvaluationPhaseConfiguration evaluationPhaseConfiguration;
    }

    modifier onlyAggregator(address aggregator) {
        if (!IAccessControl(getAccessControl()).isAggregator(aggregator)) {
            revert NotAggregator();
        }
        _;
    }

    modifier onlyTrainer(address trainer) {
        if (!IAccessControl(getAccessControl()).isTrainer(trainer)) {
            revert NotTrainer();
        }
        _;
    }

    modifier onlyEvaluator(address evaluator) {
        if (!IAccessControl(getAccessControl()).isEvaluator(evaluator)) {
            revert NotEvaluator();
        }
        _;
    }

    function initialize(
        SwarmV2InitializeParams memory params
    ) external virtual initializer {
        __EIP712_init(params.name, _VERSION);
        __RoundTraining_init();
        __BaseTrainingPhases_init(
            params.trainingPhaseConfiguration,
            params.evaluationPhaseConfiguration
        );
        __CertificateRegistry_init();
        __RoundTrainerRegistry_init();
        __RoundEvaluatorRegistry_init();
        __TaskAssignment_init();
        __SwarmCore_init(
            params.initialTrainerSelector,
            params.initialEvaluatorSelector,
            params.initialContributionCalculator,
            params.initialAccessControl,
            params.initialCompensation
        );
    }

    function initialize()
        external
        virtual
        override(RoundTrainerRegistryV2, RoundEvaluatorRegistry, TaskAssignment)
    {
        revert WrongInitialization();
    }

    function canTrain(
        address trainer,
        uint256 roundId
    ) public view returns (bool) {
        ISelector selector = ISelector(getTrainerSelector());
        IAccessControl accessControl = IAccessControl(getAccessControl());
        return
            accessControl.isTrainer(trainer) &&
            selector.isSelected(trainer, roundId);
    }

    function canEvaluate(
        address evaluator,
        uint256 roundId
    ) public view returns (bool) {
        ISelector selector = ISelector(getEvaluatorSelector());
        IAccessControl accessControl = IAccessControl(getAccessControl());
        return
            accessControl.isEvaluator(evaluator) &&
            selector.isSelected(evaluator, roundId);
    }

    function updateTrainerSelector(
        address newTrainerSelector
    ) external onlyAggregator(msg.sender) {
        _updateTrainerSelector(newTrainerSelector);
    }

    function updateEvaluatorSelector(
        address newEvaluatorSelector
    ) external onlyAggregator(msg.sender) {
        _updateEvaluatorSelector(newEvaluatorSelector);
    }

    bool private _trainerPrivacyEnabled;

    function distribute(
        uint256 roundId,
        address[] calldata trainers,
        uint64[] calldata contributions
    ) external onlyAggregator(msg.sender) {
        _distribute(roundId, trainers, contributions);
    }

    function registerRoundContributionPrivacy(
        uint256 roundId,
        bytes32 commitment,
        bytes32 modelHash,
        uint64 revealDeadline
    ) external onlyAggregator(msg.sender) {
        // _commitTrainerPrivacy enforces a hard cap (2 days) on reveal deadlines to keep bond lockups bounded.
        if (!_trainerPrivacyEnabled) {
            revert PrivacyModeDisabled();
        }
        if (updatePhase() != TRAINING_PHASE) {
            revert NotTrainingPhase();
        }
        if (roundId != currentRound()) {
            revert ForbiddenRound(roundId);
        }
        _commitTrainerPrivacy(roundId, commitment, modelHash, revealDeadline);
    }

    function revealTrainerCommitment(
        uint256 roundId,
        address trainer,
        bytes calldata nonce
    ) external onlyAggregator(msg.sender) {
        updatePhase();
        _revealTrainerPrivacy(roundId, trainer, nonce);
    }

    function slashAggregatorBond(
        uint256 roundId,
        bytes32 commitment
    ) external returns (uint256 penalty, uint256 finderReward) {
        (penalty, finderReward) = _slashAggregatorBond(
            roundId,
            commitment,
            msg.sender
        );
        if (finderReward > 0) {
            (bool success, ) = payable(msg.sender).call{value: finderReward}(
                ""
            );
            if (!success) {
                revert TransferFailed(msg.sender, finderReward);
            }
        }
    }

    function configureTrainerPrivacy(
        uint256 penalty,
        uint16 finderRewardBps
    ) external onlyAggregator(msg.sender) {
        _setTrainerPrivacyConfig(penalty, finderRewardBps);
    }

    function setTrainerPrivacyMode(
        bool enabled
    ) external onlyAggregator(msg.sender) {
        if (_trainerPrivacyEnabled == enabled) {
            return;
        }
        _trainerPrivacyEnabled = enabled;
        emit TrainerPrivacyModeUpdated(enabled);
    }

    function depositAggregatorBond()
        external
        payable
        onlyAggregator(msg.sender)
    {
        if (msg.value == 0) {
            revert ZeroBondAmount();
        }
        _increaseAggregatorBond(msg.value);
        (uint256 balance, ) = getAggregatorBondState();
        emit AggregatorBondDeposited(msg.sender, msg.value, balance);
    }

    function withdrawAggregatorBond(
        uint256 amount,
        address payable recipient
    ) external onlyAggregator(msg.sender) {
        if (amount == 0) {
            revert ZeroBondAmount();
        }
        if (recipient == address(0)) {
            revert InvalidRecipient();
        }
        _decreaseAggregatorBond(amount);
        (bool success, ) = recipient.call{value: amount}("");
        if (!success) {
            revert TransferFailed(recipient, amount);
        }
        (uint256 balance, ) = getAggregatorBondState();
        emit AggregatorBondWithdrawn(msg.sender, recipient, amount, balance);
    }

    function getAggregatorFreeBond() external view returns (uint256) {
        (uint256 balance, uint256 reserved) = getAggregatorBondState();
        return reserved >= balance ? 0 : balance - reserved;
    }

    function isTrainerPrivacyEnabled() external view returns (bool) {
        return _trainerPrivacyEnabled;
    }

    function startTrainingRound() external onlyAggregator(msg.sender) {
        if (updatePhase() != IDLE_PHASE) {
            revert NotIdle();
        }
        _nextRound();
        _forceStartTrainingPhase();
    }

    function _endTrainingPhase() internal override returns (bytes32) {
        uint256 numberOfTrainers = getTrainerCount(currentRound());
        uint256 pendingCommitments = getPendingCommitmentCount(currentRound());
        if (pendingCommitments > 0) {
            numberOfTrainers += pendingCommitments;
        }
        if (numberOfTrainers <= 0) {
            return TRAINING_PHASE;
        }
        return super._endTrainingPhase();
    }

    function registerRoundContribution(
        uint256 roundId,
        bytes32 modelHash
    ) external {
        if (_trainerPrivacyEnabled) {
            revert PrivacyModeEnabled();
        }
        if (!canTrain(msg.sender, roundId)) {
            revert NotTrainer();
        }
        _registerRoundContribution(roundId, msg.sender, modelHash);
    }

    function _registerRoundContribution(
        uint256 roundId,
        address trainer,
        bytes32 modelHash
    ) internal {
        if (updatePhase() != TRAINING_PHASE) {
            revert NotTrainingPhase();
        }
        if (roundId != currentRound()) {
            revert ForbiddenRound(roundId);
        }
        _registerTrainer(roundId, trainer, modelHash);
    }

    function registerForRoundEvaluation(uint256 roundId) external {
        if (!canEvaluate(msg.sender, roundId)) {
            revert NotEvaluator();
        }
        _registerForRoundEvaluations(roundId, msg.sender);
    }

    function _registerForRoundEvaluations(
        uint256 roundId,
        address evaluator
    ) internal {
        bytes32 phase = updatePhase();
        if (phase == EVALUATION_PHASE || phase == IDLE_PHASE) {
            revert NotEvaluatorRegistrationPhase();
        }
        _registerEvaluator(roundId, evaluator);
    }

    function _endEvaluatorRegistrationPhase()
        internal
        override
        returns (bytes32)
    {
        uint256 roundId = currentRound();
        uint256 nNodes = getEvaluatorCount(roundId);
        uint256 nTrainers = getTrainerCount(roundId);
        uint256 pendingCommitments = getPendingCommitmentCount(roundId);
        if (pendingCommitments > 0) {
            nTrainers += pendingCommitments;
        }
        if (nNodes == 0) {
            // we're going to trigger a TaskAssigment#InvalidConfig error
            return EVALUATOR_REGISTRATION_PHASE;
        }
        ContributionCalculator calc = ContributionCalculator(
            getContributionCalculator()
        );
        uint256 nTasks = calc.getEvaluationsRequired(roundId, uint8(nTrainers));
        if (nTasks == 0) {
            nTasks = calc.getEvaluationsRequired(roundId - 1, uint8(nTrainers));
            calc.setEvaluationsRequired(roundId, nTasks);
        }

        uint256 nTasksPerNode = 1;
        if (nTasks > nNodes) {
            // TODO: handle potential rounding error
            nTasksPerNode = nTasks / nNodes;
        }
        _setConfig(roundId, Config({T: nTasks, N: nNodes, R: nTasksPerNode}));
        return super._endEvaluatorRegistrationPhase();
    }

    function registerEvaluation(
        uint256 roundId,
        uint256 evalId,
        uint256 setId,
        bytes32 modelHash,
        int256 result
    ) external {
        _registerEvaluation(
            roundId,
            evalId,
            setId,
            modelHash,
            result,
            msg.sender
        );
    }

    /**
     * @param roundId   The round ID
     * @param taskId    The task ID
     * @param modelHash  The model hash
     * @param result     The evaluation result
     * @param evaluator  The evaluator address
     */
    function _registerEvaluation(
        uint256 roundId,
        uint256 taskId,
        uint256 setId,
        bytes32 modelHash,
        int256 result,
        address evaluator
    ) internal {
        if (updatePhase() != EVALUATION_PHASE) {
            revert NotEvaluationPhase();
        }
        ContributionCalculator calc = ContributionCalculator(
            getContributionCalculator()
        );
        uint256 evaluatorId = getEvaluatorIdOrThrow(roundId, evaluator);
        // evaluator id starts at 1,but TaskAssigment starts at 0
        if (!isAssigned(roundId, evaluatorId - 1, taskId)) {
            revert NotAssignedTo(roundId, taskId, evaluator);
        }
        uint256 nTrainers = getTrainerCount(roundId);
        uint256 pendingCommitments = getPendingCommitmentCount(roundId);
        if (pendingCommitments > 0) {
            nTrainers += pendingCommitments;
        }
        calc.registerResult(
            roundId,
            taskId,
            setId,
            modelHash,
            result,
            uint8(nTrainers)
        );
    }

    function claimReward(uint256 roundId, address trainer) external {
        if (hasClaimedRewards(roundId, trainer)) {
            revert RewardsAlreadyClaimed(roundId, trainer);
        }
        uint256 currentRound = currentRound();
        if (
            roundId > currentRound ||
            (roundId == currentRound && updatePhase() != IDLE_PHASE)
        ) {
            revert ForbiddenRound(roundId);
        }

        ContributionCalculator calc = ContributionCalculator(
            getContributionCalculator()
        );
        uint256 trainerId = getTrainerIdOrThrow(roundId, trainer);
        // trainer id starts at 1,but ContributionCalculator starts at 0
        int256 contribution = calc.calculateContribution(
            roundId,
            trainerId - 1,
            uint8(getTrainerCount(roundId))
        );
        emit TrainerContributed(trainer, contribution);
        _setClaimedRewards(roundId, trainer);
        address[] memory trainers = new address[](1);
        trainers[0] = trainer;
        uint64[] memory contributions = new uint64[](1);
        contributions[0] = uint64(uint256(contribution));
        _distribute(roundId, trainers, contributions);
    }

    function _distribute(
        uint256 roundId,
        address[] memory trainers,
        uint64[] memory contributions
    ) internal {
        ICompensation compensation = ICompensation(getCompensation());
        compensation.distribute(roundId, trainers, contributions);
    }

    /**
     * @dev The version parameter for the EIP712 domain.
     */
    // solhint-disable-next-line func-name-mixedcase
    function _EIP712Version()
        internal
        pure
        override(EIP712Upgradeable)
        returns (string memory)
    {
        return _VERSION;
    }

    function supportsInterface(
        bytes4 interfaceId
    )
        public
        view
        virtual
        override(RoundTraining, CertificateRegistry)
        returns (bool)
    {
        return
            interfaceId == type(IERC165).interfaceId ||
            RoundTraining.supportsInterface(interfaceId) ||
            CertificateRegistry.supportsInterface(interfaceId) ||
            interfaceId == this.canTrain.selector ||
            interfaceId == this.distribute.selector;
    }

    function setCertificate(
        bytes32 id,
        bytes calldata value
    ) external override onlyAggregator(msg.sender) {
        _setCertificate(id, value);
    }
}
