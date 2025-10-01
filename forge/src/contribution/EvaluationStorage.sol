// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import {Initializable} from "@openzeppelin-contracts-upgradeable-5.2.0/proxy/utils/Initializable.sol";
import {IERC165} from "@openzeppelin-contracts-5.2.0/utils/introspection/IERC165.sol";
import {IEvaluationStorage} from "./types.sol";

/**
 * @title EvaluationStorage
 * @dev Upgradeable contract for storing evaluation results
 * @notice This contract allows admin contracts to store evaluation results
 */
// aderyn-ignore-next-line(contract-locks-ether)
contract EvaluationStorage is Initializable, IERC165, IEvaluationStorage {
    /// @dev Storage namespace for EvaluationStorage
    struct EvaluationStorageStruct {
        /// @dev Mapping from result ID to evaluation result
        mapping(uint256 => mapping(uint256 => EvaluationResult)) evaluationResults;
    }

    /// @dev Structure to store evaluation results
    struct EvaluationResult {
        bytes32 modelHash; // Hash of the model being evaluated
        int256 result; // The evaluation result as a uint
    }

    // Storage slot for EvaluationStorage namespace
    bytes32 private constant EVALUATION_STORAGE_SLOT = keccak256("EvaluationStorage.storage");

    // Events
    event ResultRegistered(
        uint256 indexed setId,
        uint256 indexed roundId,
        bytes32 indexed modelHash,
        int256 result,
        address evaluator
    );

    // Errors
    error InvalidParameters();
    error NotEvaluated(uint256 roundId, uint256 setId);

    function __EvaluationStorage_init() internal {}

    /**
     * @dev Register an evaluation result
     * @param setId The set ID for the evaluation
     * @param modelHash Hash of the model being evaluated
     * @param result The evaluation result as a uint
     * @param roundId The round ID when evaluation was performed
     */
    function _registerResult(
        uint256 roundId,
        uint256 setId,
        bytes32 modelHash,
        int256 result
    ) internal virtual {
        if (modelHash == bytes32(0)) {
            revert InvalidParameters();
        }

        EvaluationStorageStruct storage $ = _getEvaluationStorage();
        EvaluationResult storage storedResult = $.evaluationResults[roundId][setId];
        
        if (storedResult.modelHash == bytes32(0)) {
            _setResults(roundId, setId, modelHash, result);
        } else {
            _mergeResults(storedResult, result);
        }

        emit ResultRegistered(
            setId,
            roundId,
            modelHash,
            result,
            msg.sender
        );
    }

    function _mergeResults(
        EvaluationResult storage evalA,
        int256 result
    ) internal virtual{
        int256 averagedResult = (evalA.result + result) / 2;
        evalA.result = averagedResult;
    }

    function _setResults(
        uint256 roundId,
        uint256 setId,
        bytes32 modelHash,
        int256 result
    ) internal virtual{
        EvaluationStorageStruct storage $ = _getEvaluationStorage();
        $.evaluationResults[roundId][setId] = EvaluationResult({
            modelHash: modelHash,
            result: result
        });
    }

    /**
     * @dev Get evaluation result by result ID
     * @param roundId The round ID
     * @param setId The evaluated set ID
     * @return The evaluation result struct
     */
    function getResult(
        uint256 roundId,
        uint256 setId
    ) public virtual view returns (int256) {
        EvaluationStorageStruct storage $ = _getEvaluationStorage();
        return $.evaluationResults[roundId][setId].result;
    }

        /**
     * @dev Get evaluation result by result ID
     * @param roundId The round ID
     * @param setId The evaluated set ID
     * @return The evaluation result struct
     */
    function getResultOrThrow(
        uint256 roundId,
        uint256 setId
    ) public virtual view returns (int256) {
        EvaluationStorageStruct storage $ = _getEvaluationStorage();
        if ($.evaluationResults[roundId][setId].modelHash == bytes32(0)) {
            revert NotEvaluated(roundId, setId);
        }
        return $.evaluationResults[roundId][setId].result;
    }

    /**
     * @notice Returns a pointer to the storage namespace
     * @dev This function provides access to the namespaced storage
     */
    function _getEvaluationStorage() private pure returns (EvaluationStorageStruct storage $) {
        bytes32 slot = EVALUATION_STORAGE_SLOT;
        assembly {
            $.slot := slot
        }
    }

    /// @dev See {IERC165-supportsInterface}
    function supportsInterface(bytes4 interfaceId) public virtual view override returns (bool) {
        return interfaceId == type(IEvaluationStorage).interfaceId || 
               interfaceId == type(IERC165).interfaceId;
    }
}
