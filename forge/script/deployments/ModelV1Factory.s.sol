// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import {Deployments} from "./Deployments.sol";
import {ModelRegistryV1} from "@rizemind-contracts/models/ModelRegistryV1.sol";
import {ModelRegistryFactory} from "@rizemind-contracts/models/ModelV1Factory.sol";

contract DeployModelFactoryScript is Script, Deployments {
    function run() external {
        vm.startBroadcast();

        // Deploy the implementation contract.
        ModelRegistryV1 modelImpl = new ModelRegistryV1();
        // Deploy the factory using the implementation address.
        ModelRegistryFactory modelFactory = new ModelRegistryFactory(
            address(modelImpl)
        );

        vm.stopBroadcast();

        console.log("ModelRegistryV1 deployed at:", address(modelImpl));
        console.log("ModelRegistryFactory deployed at:", address(modelFactory));
        save("ModelRegistryFactory", address(modelFactory));
    }
}
