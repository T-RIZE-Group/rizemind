// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Initializable} from "@openzeppelin-contracts-upgradeable-5.2.0/proxy/utils/Initializable.sol";

/// @notice Minimal-write affine assignment of T tasks to N nodes.
/// - Zero storage per (node, task) pair.
/// - Set config every round; all lookups are computed on-the-fly.
contract TaskAssignment is Initializable {
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
        require(next.T > 1 && next.N > 0, "N,T > 0, T > 1");
        require(next.R > 0, "R > 0");
        require(next.R <= next.T, "R <= T");
        
        TaskAssignmentStorage storage $ = _getTaskAssignmentStorage();
        $.configs[roundId] = next;
        emit ConfigUpdated(next);
    }

    /// @notice Tasks assigned to node n (length = R).
    /// @dev nodeId must be < N. View is cheap;
    function tasksOfNode(uint256 roundId, uint256 nodeId) external view returns (uint256[] memory out) {
        TaskAssignmentStorage storage $ = _getTaskAssignmentStorage();
        Config memory c = $.configs[roundId];
        require(nodeId < c.N, "n out of range");

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
        require(nodeId < c.N, "nodeId out of range");
        require(taskIndex < c.R, "taskIndex out of range");

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
        require(t < c.T, "t out of range");
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

    /// @notice Enumerate nodes assigned to task t with pagination.
    /// @param t Task id
    /// @param limit Max items to return this call
    /// @param rCursor Start r (0..R), pass 0 initially
    /// @param kCursor Start k within current r, pass 0 initially
    /// @return nodes chunk of node indices
    /// @return nextRCursor next r to resume from
    /// @return nextKCursor next k within that r
    /// @return done whether enumeration finished
    function nodesOfTaskPaged(
        uint256 roundId,
        uint256 t,
        uint256 limit,
        uint256 rCursor,
        uint256 kCursor
    )
        external
        view
        returns (uint256[] memory nodes, uint256 nextRCursor, uint256 nextKCursor, bool done)
    {
        TaskAssignmentStorage storage $ = _getTaskAssignmentStorage();
        Config memory c = $.configs[roundId];
        require(t < c.T, "t out of range");
        if (limit == 0) limit = 1;

        nodes = new uint256[](limit);
        uint256 count = 0;
        // Since a = T - 1, invA = T - 1 (since (T-1) * (T-1) ≡ 1 (mod T))

        for (uint256 r = rCursor; r < c.R && count < limit; ++r) {
            uint256 residue = _mod(t + c.T - roundId + c.T - r * (c.T - 1), c.T);
            uint256 n0 = _mod((c.T - 1) * residue, c.T);

            if (n0 < c.N) {
                // nodes in this bucket: n = n0 + k*T  while n < N
                // start from kCursor if resuming this r
                for (uint256 k = (r == rCursor ? kCursor : 0); count < limit; ++k) {
                    uint256 n = n0 + k * c.T;
                    if (n >= c.N) {
                        // no more in this r
                        k = type(uint256).max; // sentinel to break outer condition
                        break;
                    }
                    nodes[count++] = n;
                }

                if (count == limit) {
                    // pause here
                    return (_trim(nodes, count), r, /* next k */ ((n0 + ( (r==rCursor?kCursor:0) + count )/ (c.T==0?1:c.T))), false);
                }
            }

            // reset k cursor when moving to next r
            kCursor = 0;
        }

        // Finished all r's
        return (_trim(nodes, count), c.R, 0, true);
    }

    /// @notice Check membership: is node n assigned to task t?
    function isAssigned(uint256 roundId, uint256 n, uint256 t) public view returns (bool) {
        TaskAssignmentStorage storage $ = _getTaskAssignmentStorage();
        Config memory c = $.configs[roundId];
        require(n < c.N && t < c.T, "range");

        // a = T - 1, so base = ((T-1) * n + roundId) % T
        uint256 base = _mod((c.T - 1) * n + roundId, c.T);
        if (base == t) return true;

        // Check the R-1 remaining positions: t ≡ base + r*(T-1) (mod T)
        // We'll just step R times (R is typically small).
        uint256 pos = base;
        for (uint256 r = 1; r < c.R; ++r) {
            pos = _mod(pos + (c.T - 1), c.T);
            if (pos == t) return true;
        }
        return false;
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
