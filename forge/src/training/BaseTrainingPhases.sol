// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import {Initializable} from "@openzeppelin-contracts-upgradeable-5.2.0/proxy/utils/Initializable.sol";
import {ITrainingPhases} from "./ITrainingPhases.sol";

contract BaseTrainingPhases is ITrainingPhases, Initializable {
  bytes32 private _currentPhase;
  TrainingPhaseStorage private _trainingPhaseStorage;
  EvaluationPhaseStorage private _evaluationPhaseStorage;

  TrainingPhaseConfiguration private _trainingPhaseConfiguration;
  EvaluationPhaseConfiguration private _evaluationPhaseConfiguration;

  bytes32 private constant IDLE_PHASE = keccak256("IDLE");
  bytes32 private constant TRAINING_PHASE = keccak256("TRAINING");
  bytes32 private constant EVALUATOR_REGISTRATION_PHASE = keccak256("EVALUATOR_REGISTRATION");
  bytes32 private constant EVALUATION_PHASE = keccak256("EVALUATION");

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

  function __BaseTrainingPhases_init() internal onlyInitializing {
    _currentPhase = IDLE_PHASE;
  }

  function getCurrentPhase() external returns (bytes32) {
    bytes32 currentPhase = _currentPhase;
    bytes32 newPhase = _run();
    while (currentPhase != newPhase) {
      currentPhase = newPhase;
      newPhase = _run();
    }
    _currentPhase = newPhase;
    return currentPhase;
  }

  function isTraining() external view returns (bool) {
    return _currentPhase == TRAINING_PHASE;
  }

  function isEvaluation() external view returns (bool) {
    return _currentPhase == EVALUATION_PHASE || _currentPhase == EVALUATOR_REGISTRATION_PHASE;
  }

  function _run() internal returns (bytes32 nextPhase) {
    if (_currentPhase == IDLE_PHASE) {
      nextPhase = _runIdlePhaseTransition();
    } else if (_currentPhase == TRAINING_PHASE) {
      nextPhase = _runTrainingPhaseTransition();
    } else if (_currentPhase == EVALUATOR_REGISTRATION_PHASE) {
      nextPhase = _runEvaluatorRegistrationPhaseTransition();
    } else if (_currentPhase == EVALUATION_PHASE) {
      nextPhase = _runEvaluationPhaseTransition();
    }
  }

  function _runIdlePhaseTransition() internal returns (bytes32) {
    return IDLE_PHASE;
  }

  function _startTrainingPhase() internal returns (bytes32) {
    _trainingPhaseStorage.start = block.timestamp;
    return TRAINING_PHASE;
  }

  function _runTrainingPhaseTransition() internal returns (bytes32) {
    if (block.timestamp > _getTrainingPhaseExpiry()) {
      return _startEvaluationPhase();
    }
    return TRAINING_PHASE;
  }

  function _getTrainingPhaseExpiry() internal view returns (uint256) {
    return _trainingPhaseStorage.start + _getTrainingPhaseTtl();
  }

  function _getTrainingPhaseTtl() internal view returns (uint256) {
    return _trainingPhaseConfiguration.ttl;
  }

  function _startEvaluationPhase() internal returns (bytes32) {
    
    _evaluationPhaseStorage.start = block.timestamp;
    return EVALUATOR_REGISTRATION_PHASE;
  }


  function _runEvaluatorRegistrationPhaseTransition() internal returns (bytes32) {
    uint256 registrationExpiry = _getRegistrationExpiry();
    if (registrationExpiry < block.timestamp) {
      return _endEvaluatorRegistrationPhase();
    }
    return EVALUATOR_REGISTRATION_PHASE;
  }

  function _endEvaluatorRegistrationPhase() internal returns (bytes32) {
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

  function _runEvaluationPhaseTransition() internal returns (bytes32) {
    return IDLE_PHASE;
  }
}