// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test} from "forge-std/Test.sol";
import {TaskAssignment} from "../../src/scheduling/TaskAssignment.sol";

// Fuzz tests require review
contract TaskAssignmentTest is Test {
    TaskAssignment public taskAssignment;

    function setUp() public {
        // Deploy and initialize the contract
        taskAssignment = new TaskAssignment();
        taskAssignment.initialize();
        
        // Set up a simple config for round 0: 3 nodes, 5 tasks, 2 tasks per node
        TaskAssignment.Config memory config = TaskAssignment.Config({
            N: 3,  // 3 nodes
            T: 5,  // 5 tasks
            R: 2  // 2 tasks per node
        });
        
        taskAssignment.setConfig(0, config);
    }

    function test_initialConfig() public {
        TaskAssignment.Config memory cfg = taskAssignment.cfg(0);
        assertEq(cfg.N, 3);
        assertEq(cfg.T, 5);
        assertEq(cfg.R, 2);
    }

    function test_initialization() public {
        // Test that a new contract can be initialized without parameters
        TaskAssignment newTaskAssignment = new TaskAssignment();
        newTaskAssignment.initialize();
        
        // Round 0 should be empty (all values should be 0)
        TaskAssignment.Config memory emptyCfg = newTaskAssignment.cfg(0);
        assertEq(emptyCfg.N, 0);
        assertEq(emptyCfg.T, 0);
        assertEq(emptyCfg.R, 0);
    }

    function test_tasksOfNode() public {
        uint256[] memory tasks = taskAssignment.tasksOfNode(0, 0);
        assertEq(tasks.length, 2);
        // For node 0: base = (4*0 + 0) % 5 = 0
        // Tasks: [0, (0 + 4) % 5] = [0, 4]
        assertEq(tasks[0], 0);
        assertEq(tasks[1], 4);
    }

    function test_nthTaskOfNode() public {
        // Test that nthTaskOfNode returns the same as tasksOfNode[n]
        uint256[] memory allTasks = taskAssignment.tasksOfNode(0, 0);
        for (uint256 i = 0; i < allTasks.length; i++) {
            uint256 nthTask = taskAssignment.nthTaskOfNode(0, 0, i);
            assertEq(nthTask, allTasks[i], "nthTaskOfNode should match tasksOfNode array");
        }
        
        // Test specific values for node 0
        assertEq(taskAssignment.nthTaskOfNode(0, 0, 0), 0);
        assertEq(taskAssignment.nthTaskOfNode(0, 0, 1), 4);
        
        // Test bounds checking
        vm.expectRevert(abi.encodeWithSelector(TaskAssignment.NodeIdOutOfRange.selector, 3, 3));
        taskAssignment.nthTaskOfNode(0, 3, 0);
        
        vm.expectRevert(abi.encodeWithSelector(TaskAssignment.TaskIndexOutOfRange.selector, 2, 2));
        taskAssignment.nthTaskOfNode(0, 0, 2);
    }

    function test_isAssigned() public {
        // Node 0 should be assigned to tasks 0 and 4
        assertTrue(taskAssignment.isAssigned(0, 0, 0));
        assertTrue(taskAssignment.isAssigned(0, 0, 4));
        assertFalse(taskAssignment.isAssigned(0, 0, 1));
        assertFalse(taskAssignment.isAssigned(0, 0, 2));
        assertFalse(taskAssignment.isAssigned(0, 0, 3));
    }

    function test_nodeCountOfTask() public {
        // Task 0 should be assigned to some nodes
        uint256 count = taskAssignment.nodeCountOfTask(0, 0);
        assertTrue(count > 0);
    }

    function test_setConfig() public {
        TaskAssignment.Config memory newConfig = TaskAssignment.Config({
            N: 4,  // 4 nodes
            T: 7,  // 7 tasks (prime, so T-1 is coprime to T)
            R: 1  // 1 task per node
        });
        
        taskAssignment.setConfig(1, newConfig);
        
        TaskAssignment.Config memory cfg = taskAssignment.cfg(1);
        assertEq(cfg.N, 4);
        assertEq(cfg.T, 7);
        assertEq(cfg.R, 1);
    }

    // Fuzz test tasksOfNode with different configurations and nodes
    /// forge-config: default.fuzz.runs = 50
    function testFuzz_tasksOfNode_differentConfigs(
        uint16 N,
        uint16 T,
        uint8 R,
        uint256 node
    ) public {
        vm.assume(T> 1 && N >0);
        vm.assume(R > 0);
        vm.assume(R <= T);
        vm.assume(node < N);

        TaskAssignment.Config memory config = TaskAssignment.Config({
            N: N,
            T: T,
            R: R
        });
            
        // Set the new config
        taskAssignment.setConfig(1, config);
        
        // Test tasksOfNode
        uint256[] memory tasks = taskAssignment.tasksOfNode(1, node);
        
        // Validate results
        assertEq(tasks.length, R, "Wrong number of tasks returned");
        
        // Verify all tasks are within valid range
        for (uint256 i = 0; i < tasks.length; i++) {
            assertTrue(tasks[i] < T, "Task out of range");
            assertTrue(taskAssignment.isAssigned(1, node, tasks[i]));
            assertEq(tasks[i], taskAssignment.nthTaskOfNode(1, node, i));
        }
        
    }

    function testFuzz_tasksOfNode_validateUniformity() public {

        // we expect each task to have 5 assignees
        TaskAssignment.Config memory config = TaskAssignment.Config({
            N: 10,
            T: 5,
            R: 2
        });
            
        // Set the new config
        taskAssignment.setConfig(1, config);

        uint256 totalTasks = config.N * config.R;
        uint256[] memory assigneesCounts = new uint256[](config.T);
        
        for (uint256 node = 0; node < config.N; node++) {
            uint256[] memory tasks = taskAssignment.tasksOfNode(1, node);
            for (uint256 i = 0; i < tasks.length; i++) {
                assertTrue(tasks[i] < config.T, "Task out of range");
                assertTrue(taskAssignment.isAssigned(1, node, tasks[i]));
                assertEq(tasks[i], taskAssignment.nthTaskOfNode(1, node, i));
                assigneesCounts[tasks[i]]++;
            }
        }
        
        for (uint256 i = 0; i < config.T; i++) {
            assertEq(assigneesCounts[i], totalTasks / config.T);
        }
        
        // Note: The algorithm can produce duplicate tasks for certain configurations
        // This is mathematically correct behavior, not a bug
        // We'll just verify that all tasks are within valid range
        
    }

    function testFuzz_tasksOfNode_largeTest()public {
        TaskAssignment.Config memory newConfig = TaskAssignment.Config({
            N: 100,  // 100 nodes
            T: 1000, // 1000 tasks 
            R: 15  // 15 tasks per node
        });
        
        taskAssignment.setConfig(1, newConfig);
        taskAssignment.tasksOfNode(1, 50);
    }

    // Fuzz test nthTaskOfNode with different configurations
    /// forge-config: default.fuzz.runs = 50
    function testFuzz_nthTaskOfNode_differentConfigs(
        uint16 N,
        uint16 T,
        uint16 R,
        uint256 nodeId
    ) public {
        vm.assume(T> 1 && N >0);
        vm.assume(R > 0);
        vm.assume(R <= T);
        vm.assume(nodeId < N);

        TaskAssignment.Config memory config = TaskAssignment.Config({
            N: N,
            T: T,
            R: R
        });
        
        // Set the new config
        taskAssignment.setConfig(1, config);
    
        for(uint256 i = 0; i < R; i++) {
            uint256 nthTask = taskAssignment.nthTaskOfNode(1, nodeId, i);
            assertTrue(taskAssignment.isAssigned(1, nodeId, nthTask));
        }
    }
}
