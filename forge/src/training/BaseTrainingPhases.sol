// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import {Initializable} from "@openzeppelin-contracts-upgradeable-5.2.0/proxy/utils/Initializable.sol";
import {ITrainingPhases} from "./ITrainingPhases.sol";
import {console} from "forge-std/console.sol";

/**
 * @title BaseTrainingPhases
 * @author 
 * @notice Base contract for managing training lifecycle within a round.
 * 
 * This contract manages the flow between different phases of a training round:
 * 1. IDLE_PHASE - Initial state, waiting for training to begin
 * 2. TRAINING_PHASE - Active training period where trainers submit models. Triggered with _forceStartTrainingPhase(). Ends after Time to live.
 * 3. EVALUATOR_REGISTRATION_PHASE - Period to register for evaluation. Enters after the training phase is done. Ends after Time to live.
 * 4. EVALUATION_PHASE - Active evaluation period where evaluators assess models. Enters after the evaluator registration phase is done. Ends after Time to live and return in idle phase.
 * 
 * The phases automatically transition based on configured time-to-live (TTL) values.
 * The updatePhase() function can be called to advance through phases as time progresses.
 */
contract BaseTrainingPhases is ITrainingPhases, Initializable {
  bytes32 private _currentPhase;
  TrainingPhaseStorage private _trainingPhaseStorage;
  EvaluationPhaseStorage private _evaluationPhaseStorage;

  TrainingPhaseConfiguration private _trainingPhaseConfiguration;
  EvaluationPhaseConfiguration private _evaluationPhaseConfiguration;

  bytes32 public constant IDLE_PHASE = keccak256("IDLE");
  bytes32 public constant TRAINING_PHASE = keccak256("TRAINING");
  bytes32 public constant EVALUATOR_REGISTRATION_PHASE = keccak256("EVALUATOR_REGISTRATION");
  bytes32 public constant EVALUATION_PHASE = keccak256("EVALUATION");

  error WrongPhase(bytes32 expected, bytes32 actual);

  event PhaseTransition(bytes32 from, bytes32 to);
  event TrainingTtlUpdated(uint256 oldTtl, uint256 newTtl);
  event EvaluationTtlUpdated(uint256 oldTtl, uint256 newTtl);
  event EvaluationRegistrationTtlUpdated(uint256 oldTtl, uint256 newTtl);

  struct TrainingPhaseStorage {
    uint256 start;
  }

  struct TrainingPhaseConfiguration {
    uint256 ttl;
  }

  struct EvaluationPhaseStorage {
    uint256 start;
  }

  struct EvaluationPhaseConfiguration {
    uint256 ttl;
    uint256 registrationTtl;
  }

  function __BaseTrainingPhases_init(
    TrainingPhaseConfiguration memory trainingConfig,
    EvaluationPhaseConfiguration memory evaluationConfig
  ) internal onlyInitializing {
    _currentPhase = IDLE_PHASE;
    _trainingPhaseConfiguration = trainingConfig;
    _evaluationPhaseConfiguration = evaluationConfig;
  }

  function getCurrentPhase() public view returns (bytes32) {
    return _currentPhase;
  }

  function updatePhase() public returns (bytes32) {
    bytes32 currentPhase = _currentPhase;
    bytes32 newPhase = _run(currentPhase);
    
    // aderyn-ignore-next-line(costly-loop)
    while (currentPhase != newPhase) {
      emit PhaseTransition(currentPhase, newPhase);
      currentPhase = newPhase;
      newPhase = _run(currentPhase);
    }
    
    _currentPhase = newPhase;
    return newPhase;
  }

  function isTraining() external view returns (bool) {
    return _currentPhase == TRAINING_PHASE;
  }

  function isEvaluation() external view returns (bool) {
    return _currentPhase == EVALUATION_PHASE || _currentPhase == EVALUATOR_REGISTRATION_PHASE;
  }

  function isIdle() external view returns (bool) {
    return _currentPhase == IDLE_PHASE;
  }

  function isPhase(bytes32 phase) external view returns (bool) {
    return _currentPhase == phase;
  }

  function _requirePhase(bytes32 phase) internal view {
    if (_currentPhase != phase) {
      revert WrongPhase(phase, _currentPhase);
    }
  }

  function _run(bytes32 currentPhase) internal returns (bytes32 nextPhase) {
    if (currentPhase == IDLE_PHASE) {
      nextPhase = _runIdlePhaseTransition();
    } else if (currentPhase == TRAINING_PHASE) {
      nextPhase = _runTrainingPhaseTransition();
    } else if (currentPhase == EVALUATOR_REGISTRATION_PHASE) {
      nextPhase = _runEvaluatorRegistrationPhaseTransition();
    } else if (currentPhase == EVALUATION_PHASE) {
      nextPhase = _runEvaluationPhaseTransition();
    }
  }

  function _runIdlePhaseTransition() internal returns (bytes32) {
    return IDLE_PHASE;
  }

  function _forceStartTrainingPhase() internal returns (bytes32) {
    _requirePhase(IDLE_PHASE);
    bytes32 newPhase = _startTrainingPhase();
    _currentPhase = newPhase;
    return newPhase;
  }

  function _startTrainingPhase() internal virtual returns (bytes32) {
    _trainingPhaseStorage.start = block.timestamp;
    return TRAINING_PHASE;
  }

  function _runTrainingPhaseTransition() internal virtual returns (bytes32) {
    if (block.timestamp >= _getTrainingPhaseExpiry()) {
      return _endTrainingPhase();
    }
    return TRAINING_PHASE;
  }

  function _getTrainingPhaseExpiry() internal view returns (uint256) {
    return _trainingPhaseStorage.start + _getTrainingPhaseTtl();
  }

  function _getTrainingPhaseTtl() internal view returns (uint256) {
    return _trainingPhaseConfiguration.ttl;
  }

  function _endTrainingPhase() internal virtual returns (bytes32) {
    return _startEvaluationPhase();
  }

  function _startEvaluationPhase() internal virtual returns (bytes32) {
    _evaluationPhaseStorage.start = _getTrainingPhaseExpiry();
    return EVALUATOR_REGISTRATION_PHASE;
  }


  function _runEvaluatorRegistrationPhaseTransition() internal virtual returns (bytes32) {
    uint256 registrationExpiry = _getRegistrationExpiry();
    if (block.timestamp >= registrationExpiry) {
      return _endEvaluatorRegistrationPhase();
    }
    return EVALUATOR_REGISTRATION_PHASE;
  }

  function _endEvaluatorRegistrationPhase() internal virtual returns (bytes32) {
    return EVALUATION_PHASE;
  }

  function _getRegistrationExpiry() internal view returns (uint256) {
    return _getEvaluationStart() + _getRegistrationTtl();
  }

  function _getRegistrationTtl() internal view returns (uint256) {
    return _evaluationPhaseConfiguration.registrationTtl;
  }

  function _getEvaluationStart() internal view returns (uint256) {
    return _evaluationPhaseStorage.start;
  }

  function _runEvaluationPhaseTransition() internal virtual returns (bytes32) {
    if (block.timestamp > _getEvaluationPhaseExpiry()) {
      return _endEvaluationPhase();
    }
    return EVALUATION_PHASE;
  }

  function _getEvaluationPhaseExpiry() internal view returns (uint256) {
    return _evaluationPhaseStorage.start + _getEvaluationPhaseTtl();
  }

  function _getEvaluationPhaseTtl() internal view returns (uint256) {
    return _evaluationPhaseConfiguration.ttl;
  }

  function _endEvaluationPhase() internal virtual returns (bytes32) {
    return IDLE_PHASE;
  }

  // ============================================================================
  // CONFIGURATION UPDATE FUNCTIONS
  // ============================================================================

  /// @notice Update training phase configuration
  /// @param config The training phase configuration struct
  function _setTrainingPhaseConfiguration(TrainingPhaseConfiguration memory config) internal {
    _trainingPhaseConfiguration = config;
  }

  /// @notice Update evaluation phase configuration
  /// @param config The evaluation phase configuration struct
  function _setEvaluationPhaseConfiguration(EvaluationPhaseConfiguration memory config) internal {
    _evaluationPhaseConfiguration = config;
  }

  /// @notice Update training phase TTL only
  /// @param ttl The time-to-live for training phase
  function _setTrainingPhaseTtl(uint256 ttl) internal {
    uint256 oldTtl = _trainingPhaseConfiguration.ttl;
    _trainingPhaseConfiguration.ttl = ttl;
    emit TrainingTtlUpdated(oldTtl, ttl);
  }

  /// @notice Update evaluation phase TTL only
  /// @param ttl The time-to-live for evaluation phase
  function _setEvaluationPhaseTtl(uint256 ttl) internal {
    uint256 oldTtl = _evaluationPhaseConfiguration.ttl;
    _evaluationPhaseConfiguration.ttl = ttl;
    emit EvaluationTtlUpdated(oldTtl, ttl);
  }

  /// @notice Update registration TTL only
  /// @param registrationTtl The time-to-live for evaluator registration phase
  function _setRegistrationTtl(uint256 registrationTtl) internal {
    uint256 oldTtl = _evaluationPhaseConfiguration.registrationTtl;
    _evaluationPhaseConfiguration.registrationTtl = registrationTtl;
    emit EvaluationRegistrationTtlUpdated(oldTtl, registrationTtl);
  }

  /// @notice Get training phase configuration
  /// @return The training phase configuration struct
  function getTrainingPhaseConfiguration() public view returns (TrainingPhaseConfiguration memory) {
    return _trainingPhaseConfiguration;
  }

  /// @notice Get evaluation phase configuration
  /// @return The evaluation phase configuration struct
  function getEvaluationPhaseConfiguration() public view returns (EvaluationPhaseConfiguration memory) {
    return _evaluationPhaseConfiguration;
  }
}