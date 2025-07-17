// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

import "forge-std/Test.sol";
// Adjust the import path below to where your contract is located.
import {InitializableFLAccessControl} from "@rizemind-contracts/access/FLAccessControl.sol";

contract InitializableFLAccessControlTest is Test {
    InitializableFLAccessControl public accessControl;
    address public aggregator;
    address[] public trainers;
    address public nonTrainer;

    function setUp() public {
        // Define our test accounts.
        aggregator = vm.addr(1);
        // Define trainers as accounts[1], accounts[2], accounts[3]
        trainers.push(vm.addr(2));
        trainers.push(vm.addr(3));
        trainers.push(vm.addr(4));
        // Define non-trainer as accounts[4]
        nonTrainer = vm.addr(5);

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
}
