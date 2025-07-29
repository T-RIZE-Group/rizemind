// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

import "forge-std/Test.sol";

import {InitializableFLAccessControl} from "@rizemind-contracts/access/FLAccessControl.sol";

contract InitializableFLAccessControlTest is Test {
    InitializableFLAccessControl public accessControl;
    address public aggregator;
    address[] public trainers;
    address public nonTrainer;
    address public evaluator;
    address public anotherAggregator;

    function setUp() public {
        aggregator = vm.addr(1);
        // Define trainers as accounts[1], accounts[2], accounts[3]
        trainers.push(vm.addr(2));
        trainers.push(vm.addr(3));
        trainers.push(vm.addr(4));
        // Define non-trainer as accounts[4]
        nonTrainer = vm.addr(5);
        evaluator = vm.addr(6);
        anotherAggregator = vm.addr(7);
        // Deploy the contract as aggregator and initialize it.
        vm.prank(aggregator);
        accessControl = new InitializableFLAccessControl();
        vm.prank(aggregator);
        accessControl.initialize(aggregator, trainers);
    }

    function testIsTrainer() public view {
        // aggregator should not have the trainer role.
        bool isAggregatorTrainer = accessControl.isTrainer(aggregator);
        assertFalse(isAggregatorTrainer, "aggregator should not be trainer");

        // The first trainer should have the trainer role.
        bool isTrainer = accessControl.isTrainer(trainers[0]);
        assertTrue(isTrainer, "trainer role not assigned to trainers[0]");

        // nonTrainer should not have the trainer role.
        bool isNonTrainer = accessControl.isTrainer(nonTrainer);
        assertFalse(
            isNonTrainer,
            "non-trainer incorrectly assigned trainer role"
        );
    }

    function testAddTrainerByAggregator() public {
        vm.prank(aggregator);
        accessControl.addTrainer(nonTrainer);
        assertTrue(accessControl.isTrainer(nonTrainer), "addTrainer failed");
    }

    function testAddTrainerByNonAggregatorReverts() public {
        vm.prank(nonTrainer);
        vm.expectRevert();
        accessControl.addTrainer(nonTrainer);
    }

    function testAddAggregatorByAggregator() public {
        vm.prank(aggregator);
        accessControl.addAggregator(anotherAggregator);
        assertTrue(
            accessControl.isAggregator(anotherAggregator),
            "addAggregator failed"
        );
    }

    function testAddAggregatorByNonAggregatorReverts() public {
        vm.prank(nonTrainer);
        vm.expectRevert();
        accessControl.addAggregator(nonTrainer);
    }

    function testIsAggregator() public view {
        // aggregator should have the aggregator role.
        bool isAggregator = accessControl.isAggregator(aggregator);
        assertTrue(isAggregator, "aggregator role not assigned");

        // A trainer should not have the aggregator role.
        bool trainerIsAggregator = accessControl.isAggregator(trainers[0]);
        assertFalse(
            trainerIsAggregator,
            "trainer incorrectly assigned aggregator role"
        );

        // nonTrainer should not have the aggregator role.
        bool nonTrainerIsAggregator = accessControl.isAggregator(nonTrainer);
        assertFalse(
            nonTrainerIsAggregator,
            "non-trainer incorrectly assigned aggregator role"
        );
    }

    function testAddEvaluatorByAggregator() public {
        vm.prank(aggregator);
        accessControl.addEvaluator(evaluator);
        assertTrue(accessControl.isEvaluator(evaluator), "addEvaluator failed");
    }

    function testAddEvaluatorByNonAggregatorReverts() public {
        vm.prank(nonTrainer);
        vm.expectRevert();
        accessControl.addEvaluator(evaluator);
    }

    function testIsEvaluator() public {
        vm.prank(aggregator);
        accessControl.addEvaluator(evaluator);
        assertTrue(
            accessControl.isEvaluator(evaluator),
            "evaluator role not assigned"
        );
        assertFalse(
            accessControl.isEvaluator(nonTrainer),
            "non-evaluator incorrectly assigned evaluator role"
        );
    }

    function testSupportsInterface() public view {
        // Should not support a random selector (e.g., bytes4(0xdeadbeef))
        assertFalse(
            accessControl.supportsInterface(0xdeadbeef),
            "should not support random selector"
        );
        // Custom selectors
        assertTrue(
            accessControl.supportsInterface(accessControl.addTrainer.selector),
            "addTrainer selector"
        );
        assertTrue(
            accessControl.supportsInterface(accessControl.isTrainer.selector),
            "isTrainer selector"
        );
        assertTrue(
            accessControl.supportsInterface(
                accessControl.addAggregator.selector
            ),
            "addAggregator selector"
        );
        assertTrue(
            accessControl.supportsInterface(
                accessControl.isAggregator.selector
            ),
            "isAggregator selector"
        );
        assertTrue(
            accessControl.supportsInterface(
                accessControl.addEvaluator.selector
            ),
            "addEvaluator selector"
        );
        assertTrue(
            accessControl.supportsInterface(accessControl.isEvaluator.selector),
            "isEvaluator selector"
        );
    }

    function testDoubleAddTrainer() public {
        vm.prank(aggregator);
        accessControl.addTrainer(trainers[0]);
        assertTrue(
            accessControl.isTrainer(trainers[0]),
            "trainer should still have role after double add"
        );
    }

    function testDoubleAddAggregator() public {
        vm.prank(aggregator);
        accessControl.addAggregator(aggregator);
        assertTrue(
            accessControl.isAggregator(aggregator),
            "aggregator should still have role after double add"
        );
    }

    function testDoubleAddEvaluator() public {
        vm.prank(aggregator);
        accessControl.addEvaluator(evaluator);
        vm.prank(aggregator);
        accessControl.addEvaluator(evaluator);
        assertTrue(
            accessControl.isEvaluator(evaluator),
            "evaluator should still have role after double add"
        );
    }

    function testInitializeRevertsIfCalledTwice() public {
        vm.prank(aggregator);
        vm.expectRevert();
        accessControl.initialize(aggregator, trainers);
    }
}
