// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {EIP712Upgradeable} from "@openzeppelin-contracts-upgradeable-5.2.0/utils/cryptography/EIP712Upgradeable.sol";
import {ContextUpgradeable} from "@openzeppelin-contracts-upgradeable-5.2.0/utils/ContextUpgradeable.sol";

import {IAccessControl} from "../access/IAccessControl.sol";
import {SimpleMintCompensation} from "../compensation/SimpleMintCompensation.sol";
import {RoundTraining} from "../training/RoundTraining.sol";
import {CertificateRegistry} from "./registry/CertificateRegistry.sol";
import {SwarmCore} from "./registry/SwarmCore.sol";
import {ISelector} from "../sampling/ISelector.sol";
import {TaskAssignment} from "../scheduling/TaskAssignment.sol";
import {BaseTrainingPhases} from "../training/BaseTrainingPhases.sol";
import {RoundTrainerRegistry} from "./registry/RoundTrainerRegistry.sol";
import {RoundEvaluatorRegistry} from "./registry/RoundEvaluatorRegistry.sol";
import {ContributionCalculator} from "../contribution/ContributionCalculator.sol";
import {ICompensation} from "../compensation/types.sol";

/**
 * @title SwarmV1
 * @author 
 * @notice SwarmV1 is the entrypoint for the Swarm Coordination.
 * 
 * It encapsulates the training liefecycle, access control, contribution calculation and metadata storage.
 * 
 * Overview of a round:
 * 1. Aggregator calls startTrainingRound() to start the training round
 * 2. Trainers call registerRoundContribution() to register their contributions
 * 3. Evaluators call registerForRoundEvaluations() to register for round evaluations
 * 4. Evaluators call registerEvaluation() to register their evaluations
 * 5. Aggregator calls nextRound() to finish the round
 * 
 * The swarm has a set of whitelisted trainers and evaluators based on the FLAccessControl contract.
 * Each round a subset of those nodes are selected by the SamplerSelector contract.
 * 
 * For contribution calculation, the IContributionCalculator defines the number of evaluation tasks required.
 * These tasks are assigned an incremental task ID.
 * 
 * Before the evaluation starts, the evaluators registers so the TaskAssigment module distributes tasks
 * uniformly to the evaluators.
 * 
 * After the evaluation is completed, the trainers can claim their rewards by calling claimReward().
 */
contract SwarmV1 is
    EIP712Upgradeable,
    RoundTraining,
    BaseTrainingPhases,
    CertificateRegistry,
    RoundTrainerRegistry,
    RoundEvaluatorRegistry,
    TaskAssignment,
    SwarmCore
{
    string private constant _VERSION = "swarm-v1.0.0";

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
        string memory name,
        string memory symbol,
        address aggregator,
        address[] memory initialTrainers,
        address initialTrainerSelector,
        address initialEvaluatorSelector,
        address initialContributionCalculator
    ) external virtual initializer {
        __EIP712_init(name, _VERSION);
        __RoundTraining_init();
        // TODO: make these configurable
        __BaseTrainingPhases_init(BaseTrainingPhases.TrainingPhaseConfiguration({ttl: 1000}), BaseTrainingPhases.EvaluationPhaseConfiguration({ttl: 1000, registrationTtl: 1000}));
        __CertificateRegistry_init();
        __RoundTrainerRegistry_init();
        __RoundEvaluatorRegistry_init();
        __TaskAssignment_init();
        __SwarmCore_init(initialTrainerSelector, initialEvaluatorSelector, initialContributionCalculator, address(0), address(0));
    }

    function initialize() external virtual override(RoundTrainerRegistry, RoundEvaluatorRegistry, TaskAssignment) {
        revert WrongInitialization();
    }

    function canTrain(
        address trainer,
        uint256 roundId
    ) public view returns (bool) {
        ISelector selector = ISelector(getTrainerSelector());
        IAccessControl accessControl = IAccessControl(getAccessControl());
        return accessControl.isTrainer(trainer) && selector.isSelected(trainer, roundId);
    }

    function updateTrainerSelector(address newTrainerSelector) external onlyAggregator(msg.sender) {
        _updateTrainerSelector(newTrainerSelector);
    }

    function updateEvaluatorSelector(address newEvaluatorSelector) external onlyAggregator(msg.sender) {
        _updateEvaluatorSelector(newEvaluatorSelector);
    }

    function distribute(
        uint256 roundId,
        address[] calldata trainers,
        uint64[] calldata contributions
    ) external onlyAggregator(msg.sender) {
        _distribute(roundId, trainers, contributions);
    }

    function startTrainingRound() external onlyAggregator(msg.sender) {
        if (updatePhase() != IDLE_PHASE) {
            revert NotIdle();
        }
        _nextRound();
        _forceStartTrainingPhase();
    }

    function registerRoundContribution(uint256 roundId, bytes32 modelHash) external onlyTrainer(msg.sender) {
        _registerRoundContribution(roundId, msg.sender, modelHash);
    }

    function _registerRoundContribution(uint256 roundId, address trainer, bytes32 modelHash) internal {
        if (updatePhase() != TRAINING_PHASE) {
            revert NotTrainingPhase();
        }
        if (roundId != currentRound()) {
            revert ForbiddenRound(roundId);
        }
        _registerTrainer(roundId, trainer, modelHash);
    }

    function registerForRoundEvaluation(uint256 roundId) external onlyEvaluator(msg.sender) {
        _registerForRoundEvaluations(roundId, msg.sender);
    }

    function _registerForRoundEvaluations(uint256 roundId, address evaluator) internal {
        if (updatePhase() != EVALUATOR_REGISTRATION_PHASE) {
            revert NotEvaluatorRegistrationPhase();
        }
        _registerEvaluator(roundId, evaluator);
    }

    function _endEvaluatorRegistrationPhase() internal override returns (bytes32) {
        uint256 roundId = currentRound();
        ContributionCalculator calc = ContributionCalculator(getContributionCalculator());
        uint256 nTasks = calc.getEvaluationsRequired(roundId, uint8(getTrainerCount(roundId)));
        if (nTasks == 0) {
            nTasks = calc.getEvaluationsRequired(roundId - 1, uint8(getTrainerCount(roundId - 1)));
            calc.setEvaluationsRequired(roundId, nTasks);
        }
        uint256 nNodes = getEvaluatorCount(roundId);
        uint256 nTasksPerNode = 1;
        if (nTasks > nNodes) {
            // TODO: handle potential rounding error
            nTasksPerNode = nTasks / nNodes;
        }
        _setConfig(roundId, Config({
            T: nTasks,
            N: nNodes,
            R: nTasksPerNode
        }));
        return super._endEvaluatorRegistrationPhase();
    }

    function registerEvaluation(
        uint256 roundId,
        uint256 evalId,
        uint256 setId,
        bytes32 modelHash,
        int256 result
    ) external {
        _registerEvaluation(roundId, evalId, setId, modelHash, result, msg.sender);
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
        ContributionCalculator calc = ContributionCalculator(getContributionCalculator());
        uint256 evaluatorId = getEvaluatorIdOrThrow(roundId, evaluator);
        // evaluator id starts at 1,but TaskAssigment starts at 0
        if (!isAssigned(roundId, evaluatorId - 1, taskId)) {
            revert NotAssignedTo(roundId, taskId, evaluator);
        }
        uint256 nTrainers = getTrainerCount(roundId);
        calc.registerResult(roundId, taskId, setId, modelHash, result, uint8(nTrainers));
    }

    function claimReward(uint256 roundId, address trainer) external {
        if(hasClaimedRewards(roundId, trainer)) {
            revert RewardsAlreadyClaimed(roundId, trainer);
        }
        uint256 currentRound = currentRound();
        if (roundId > currentRound || (roundId == currentRound && updatePhase() != IDLE_PHASE)) {
            revert ForbiddenRound(roundId);
        }

        ContributionCalculator calc = ContributionCalculator(getContributionCalculator());
        uint256 trainerId = getTrainerIdOrThrow(roundId, trainer);
        // trainer id starts at 1,but ContributionCalculator starts at 0
        int256 contribution = calc.calculateContribution(roundId, trainerId - 1, uint8(getTrainerCount(roundId)));
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
