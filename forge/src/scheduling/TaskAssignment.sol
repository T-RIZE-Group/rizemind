// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Initializable} from "@openzeppelin-contracts-upgradeable-5.2.0/proxy/utils/Initializable.sol";

/// @notice Minimal-write affine assignment of T tasks to N nodes.
/// - Zero storage per (node, task) pair.
/// - Set config every round; all lookups are computed on-the-fly.
contract TaskAssignment is Initializable {
    // Custom errors
    error InvalidConfig(uint256 N, uint256 T, uint256 R);
    error NodeIdOutOfRange(uint256 nodeId, uint256 maxNodes);
    error TaskIndexOutOfRange(uint256 taskIndex, uint256 maxTasks);
    error TaskOutOfRange(uint256 taskId, uint256 maxTasks);
    error RangeOutOfBounds(uint256 nodeId, uint256 taskId, uint256 maxNodes, uint256 maxTasks);
    /// @dev Storage namespace for TaskAssignment
    struct TaskAssignmentStorage {
        mapping(uint256 => Config) configs; // roundId => Config
    }

    struct Config {
        uint256 N;      // number of nodes (indices 0..N-1)
        uint256 T;      // number of tasks (indices 0..T-1)
        uint256 R;      // tasks per node (>=1)
    }

    // Storage slot for TaskAssignment namespace
    bytes32 private constant TASK_ASSIGNMENT_STORAGE_SLOT = keccak256("TaskAssignment.storage");

    event ConfigUpdated(Config cfg);

    /// @notice Get the configuration for a specific round
    function cfg(uint256 roundId) public view returns (Config memory) {
        TaskAssignmentStorage storage $ = _getTaskAssignmentStorage();
        return $.configs[roundId];
    }

    /// @notice Initialize the contract
    function initialize() external virtual initializer {
        __TaskAssignment_init();
    }

    function __TaskAssignment_init() internal onlyInitializing {
        // No initial config - round 0 will be empty by default
    }

    /// @notice Update config for a specific round (rare write). Ensure coprimality and ranges.
    function setConfig(uint256 roundId, Config calldata next) external {
        _setConfig(roundId, next);
    }

    function _setConfig(uint256 roundId, Config memory next) internal {
        if (next.T <= 1 || next.N == 0) {
            revert InvalidConfig(next.N, next.T, next.R);
        }
        if (next.R == 0) {
            revert InvalidConfig(next.N, next.T, next.R);
        }
        if (next.R > next.T) {
            revert InvalidConfig(next.N, next.T, next.R);
        }
        
        TaskAssignmentStorage storage $ = _getTaskAssignmentStorage();
        $.configs[roundId] = next;
        emit ConfigUpdated(next);
    }

    /// @notice Tasks assigned to node n (length = R).
    /// @dev nodeId must be < N. View is cheap;
    function tasksOfNode(uint256 roundId, uint256 nodeId) external view returns (uint256[] memory out) {
        TaskAssignmentStorage storage $ = _getTaskAssignmentStorage();
        Config memory c = $.configs[roundId];
        if (nodeId >= c.N) {
            revert NodeIdOutOfRange(nodeId, c.N);
        }

        out = new uint256[](c.R);
        // a = T - 1, so base = ((T-1) * nodeId + roundId) % T
        uint256 base = _mod((c.T - 1) * nodeId + roundId, c.T);
        for (uint256 r = 0; r < c.R; ++r) {
            // s = T - 1, so task = (base + r*(T-1)) % T
            out[r] = _mod(base + r * (c.T - 1), c.T);
        }
    }

    /// @notice Get the nth task assigned to a node (0-indexed).
    /// @dev nodeId must be < N, taskIndex must be < R.
    function nthTaskOfNode(uint256 roundId, uint256 nodeId, uint256 taskIndex) external view returns (uint256) {
        TaskAssignmentStorage storage $ = _getTaskAssignmentStorage();
        Config memory c = $.configs[roundId];
        if (nodeId >= c.N) {
            revert NodeIdOutOfRange(nodeId, c.N);
        }
        if (taskIndex >= c.R) {
            revert TaskIndexOutOfRange(taskIndex, c.R);
        }

        // a = T - 1, so base = ((T-1) * nodeId + roundId) % T
        uint256 base = _mod((c.T - 1) * nodeId + roundId, c.T);
        // s = T - 1, so task = (base + taskIndex*(T-1)) % T
        return _mod(base + taskIndex * (c.T - 1), c.T);
    }

    /// @notice Count of nodes assigned to task t.
    /// @dev Often ~ N * R / T (rounded), but computed exactly.
    function nodeCountOfTask(uint256 roundId, uint256 t) public view returns (uint256 total) {
        TaskAssignmentStorage storage $ = _getTaskAssignmentStorage();
        Config memory c = $.configs[roundId];
        if (t >= c.T) {
            revert TaskOutOfRange(t, c.T);
        }
        // Since a = T - 1, invA = T - 1 (since (T-1) * (T-1) ≡ 1 (mod T))

        for (uint256 r = 0; r < c.R; ++r) {
            // n0 = (T-1) * (t - roundId - r*(T-1)) mod T
            uint256 residue = _mod(t + c.T - roundId + c.T - r * (c.T - 1), c.T);
            uint256 n0 = _mod((c.T - 1) * residue, c.T);

            if (n0 < c.N) {
                // count for this r: n = n0 + k*T < N  -> k_max = floor((N-1 - n0)/T)
                total += 1 + (c.N - 1 - n0) / c.T;
            }
        }
    }

    /// @notice Check membership: is node n assigned to task t?
    function isAssigned(uint256 roundId, uint256 n, uint256 t) public view returns (bool) {
        TaskAssignmentStorage storage $ = _getTaskAssignmentStorage();
        Config memory c = $.configs[roundId];
        if (n >= c.N || t >= c.T) {
            revert RangeOutOfBounds(n, t, c.N, c.T);
        }

        uint256 base = _mod((c.T - 1) * n + roundId, c.T);
        uint256 delta = _mod(base + c.T - t, c.T); // (base - t) mod T
        return (delta < c.R);
    }

    function _mod(uint256 x, uint256 m) private pure returns (uint256) {
        uint256 r = x % m;
        return r;
    }

    function _trim(uint256[] memory arr, uint256 len) private pure returns (uint256[] memory out) {
        if (arr.length == len) return arr;
        out = new uint256[](len);
        for (uint256 i = 0; i < len; ++i) out[i] = arr[i];
    }

    /**
     * @notice Returns a pointer to the storage namespace
     * @dev This function provides access to the namespaced storage
     */
    function _getTaskAssignmentStorage() private pure returns (TaskAssignmentStorage storage $) {
        bytes32 slot = TASK_ASSIGNMENT_STORAGE_SLOT;
        assembly {
            $.slot := slot
        }
    }
}
