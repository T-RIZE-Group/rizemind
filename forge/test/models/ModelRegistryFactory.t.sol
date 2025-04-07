// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import {ModelRegistryV1} from "@rizemind-contracts/models/ModelRegistryV1.sol";
import {ModelRegistryFactory} from "@rizemind-contracts/models/ModelV1Factory.sol";

contract ModelRegistryFactoryTest is Test {
    ModelRegistryV1 public modelContract;
    ModelRegistryFactory public modelFactory;
    address public aggregator;
    address[] public trainers;
    address public nonTrainer;

    function setUp() public {
        // Set up our test accounts.
        aggregator = vm.addr(1);
        trainers.push(vm.addr(2)); // trainers[0]
        trainers.push(vm.addr(3)); // trainers[1]
        trainers.push(vm.addr(4)); // trainers[2]
        nonTrainer = vm.addr(5);

        // Deploy the implementation contract (ModelRegistryV1) as aggregator.
        vm.prank(aggregator);
        modelContract = new ModelRegistryV1();
        // Deploy the factory with the implementation address.
        vm.prank(aggregator);
        modelFactory = new ModelRegistryFactory(address(modelContract));
    }

    function testFactoryDeploy() public {
        // Expect the ContractCreated event.
        // We use vm.expectEmit to tell Forge to check that the next transaction emits an event
        // matching our expected parameters. Here we only check that an event of type ContractCreated
        // is emitted (we ignore its data by setting false for the other topics).
        vm.expectEmit(false, false, true, false);
        emit ModelRegistryFactory.ContractCreated(
            address(0),
            address(0),
            "hello"
        ); // placeholder: we don't check the actual address

        // Call createModel as aggregator.
        vm.prank(aggregator);
        modelFactory.createModel("hello", "world", aggregator, trainers);
    }

    function testUpdateImplementation() public {
        // Deploy a new implementation.
        vm.prank(aggregator);
        ModelRegistryV1 newModel = new ModelRegistryV1();
        vm.prank(aggregator);
        newModel.initialize("Test2", "tst", aggregator, trainers);

        // Update the factory implementation.
        vm.prank(aggregator);
        modelFactory.updateImplementation(address(newModel));

        // Now call createModel. If the new implementation is used, the call should succeed.
        vm.prank(aggregator);
        modelFactory.createModel("hello", "world", aggregator, trainers);
    }

    function testUpdateImplementationProtected() public {
        // Only the aggregator should be allowed to update the implementation.
        // Here we simulate a call from trainers[0] which should revert.
        vm.prank(trainers[0]);
        vm.expectRevert(); // Optionally, add an error message if your contract reverts with one.
        modelFactory.updateImplementation(trainers[0]);
    }
}
