// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import {SwarmV1} from "@rizemind-contracts/swarm/SwarmV1.sol";
import {SwarmV1Factory} from "@rizemind-contracts/swarm/SwarmV1Factory.sol";

contract SwarmV1FactoryTest is Test {
    SwarmV1 public swarmImpl;
    SwarmV1Factory public swarmFactory;
    address public aggregator;
    address[] public trainers;
    address public nonTrainer;

    function setUp() public {
        aggregator = vm.addr(1);
        trainers.push(vm.addr(2));
        trainers.push(vm.addr(3));
        trainers.push(vm.addr(4));
        nonTrainer = vm.addr(5);
        // Deploy the implementation contract (SwarmV1) as aggregator.
        vm.prank(aggregator);
        swarmImpl = new SwarmV1();
        // Deploy the factory with the implementation address.
        vm.prank(aggregator);
        swarmFactory = new SwarmV1Factory(address(swarmImpl));
    }

    function testFactoryDeploy() public {
        vm.expectEmit(false, false, true, false);
        emit SwarmV1Factory.ContractCreated(address(0), address(0), "hello");
        vm.prank(aggregator);
        swarmFactory.createSwarm("hello", "world", aggregator, trainers);
    }

    function testUpdateImplementation() public {
        vm.prank(aggregator);
        SwarmV1 newImpl = new SwarmV1();
        vm.prank(aggregator);
        swarmFactory.updateImplementation(address(newImpl));
        vm.prank(aggregator);
        swarmFactory.createSwarm("hello", "world", aggregator, trainers);
    }

    function testUpdateImplementationProtected() public {
        vm.prank(trainers[0]);
        vm.expectRevert();
        swarmFactory.updateImplementation(trainers[0]);
    }

    function testGetImplementation() public view {
        assertEq(
            swarmFactory.getImplementation(),
            address(swarmImpl),
            "getImplementation should return the correct address"
        );
    }

    function testUpdateImplementationZeroAddressReverts() public {
        vm.prank(aggregator);
        vm.expectRevert(bytes("implementation cannot be null"));
        swarmFactory.updateImplementation(address(0));
    }
}
