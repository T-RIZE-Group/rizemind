// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import {Deployments} from "./Deployments.sol";
import {ModelMetaV1} from "@rizemind-contracts/models/ModelMetaV1.sol";
import {ModelFactory} from "@rizemind-contracts/models/ModelFactory.sol";

contract DeployModelFactoryScript is Script, Deployments {
    function run() external {
        vm.startBroadcast();

        // Deploy the implementation contract.
        ModelMetaV1 modelImpl = new ModelMetaV1();
        // Deploy the factory using the implementation address.
        ModelFactory modelFactory = new ModelFactory(address(modelImpl));

        vm.stopBroadcast();

        console.log("ModelRegistryV1 deployed at:", address(modelImpl));
        console.log("ModelRegistryFactory deployed at:", address(modelFactory));
        save("ModelRegistryFactory", address(modelFactory));
    }
}
