// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import {Deployments} from "./Deployments.sol";
import {SwarmV1} from "@rizemind-contracts/swarm/SwarmV1.sol";
import {SwarmV1Factory} from "@rizemind-contracts/swarm/SwarmV1Factory.sol";

contract DeployModelFactoryScript is Script, Deployments {
    function run() external {
        vm.startBroadcast();

        // Deploy the implementation contract.
        SwarmV1 modelImpl = new SwarmV1();
        // Deploy the factory using the implementation address.
        SwarmV1Factory modelFactory = new SwarmV1Factory(address(modelImpl));

        vm.stopBroadcast();

        console.log("ModelRegistryV1 deployed at:", address(modelImpl));
        console.log("ModelRegistryFactory deployed at:", address(modelFactory));
        save("ModelRegistryFactory", address(modelFactory));
    }
}
