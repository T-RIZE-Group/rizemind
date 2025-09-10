// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import { IERC5267 } from "@openzeppelin-contracts-5.2.0/interfaces/IERC5267.sol";

/**
 * @title IAccessControl
 * @notice Interface for federated learning access control functionality
 * @dev Defines the interface for managing roles and permissions in a federated learning system
 */
interface IAccessControl is IERC5267 {
    // View functions
    function isTrainer(address trainer) external view returns (bool);
    function isAggregator(address aggregator) external view returns (bool);
    function isEvaluator(address evaluator) external view returns (bool);

}
