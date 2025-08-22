// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import {RandomSampling} from "@rizemind-contracts/sampling/RandomSampling.sol";
import {SelectorFactory} from "@rizemind-contracts/sampling/SelectorFactory.sol";
import {DeploySelectorFactory} from "./SelectorFactory.s.sol";

contract DeployRandomSampling is Script {
    // Selector ID for RandomSampling
    
    function run() external {
        DeploySelectorFactory deploySelectorFactory = new DeploySelectorFactory();
        address selectorFactoryAddress = deploySelectorFactory.getDeployedAddress();
        
        SelectorFactory selectorFactory = SelectorFactory(selectorFactoryAddress);

        RandomSampling randomSamplingImpl = new RandomSampling();
        (,, string memory version,,,,) = randomSamplingImpl.eip712Domain();
        bytes32 id = selectorFactory.getID(version);
        // Check if RandomSampling is already registered
        if (selectorFactory.isSelectorRegistered(id)) {
            console.log("=== RandomSampling Already Registered ===");
            address existingImpl = selectorFactory.getSelectorImplementation(id);
            console.log("RandomSampling is already registered at:", existingImpl);
            console.log("Version:", version);
            console.log("Selector ID:", vm.toString(id));
            return;
        }
        
        vm.startBroadcast();
        console.log("Deploying RandomSampling...");
        // Deploy RandomSampling implementation
        RandomSampling deployment = new RandomSampling();
        console.log("RandomSampling deployed at:", address(deployment));
        // Register RandomSampling with the factory
        selectorFactory.registerSelectorImplementation(address(deployment));

        vm.stopBroadcast();

        // Log deployment and registration
        console.log("=== RandomSampling Deployment & Registration ===");
        console.log("RandomSampling implementation deployed at:", address(deployment));
        console.log("Registered with SelectorFactory at:", selectorFactoryAddress);
        console.log("Version:", version);
        console.log("Selector ID:", vm.toString(id));

    }
}
