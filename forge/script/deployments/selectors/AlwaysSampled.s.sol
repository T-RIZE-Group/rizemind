// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import {AlwaysSampled} from "@rizemind-contracts/sampling/AlwaysSampled.sol";
import {SelectorFactory} from "@rizemind-contracts/sampling/SelectorFactory.sol";
import {DeploySelectorFactory} from "./SelectorFactory.s.sol";

contract DeployAlwaysSampled is Script {
    
    function run() external {
        DeploySelectorFactory deploySelectorFactory = new DeploySelectorFactory();
        address selectorFactoryAddress = deploySelectorFactory.getDeployedAddress();
        
        SelectorFactory selectorFactory = SelectorFactory(selectorFactoryAddress);

        AlwaysSampled alwaysSampledImpl = new AlwaysSampled();
        (,, string memory version,,,,) = alwaysSampledImpl.eip712Domain();
        bytes32 id = selectorFactory.getID(version);
        // Check if AlwaysSampled is already registered
        if (selectorFactory.isSelectorRegistered(id)) {
            console.log("=== AlwaysSampled Already Registered ===");
            address existingImpl = selectorFactory.getSelectorImplementation(id);
            console.log("AlwaysSampled is already registered at:", existingImpl);
            console.log("Version:", version);
            console.log("Selector ID:", vm.toString(id));
            return;
        }
        
        vm.startBroadcast();
        console.log("Deploying AlwaysSampled...");
        // Deploy AlwaysSampled implementation
        AlwaysSampled deployment = new AlwaysSampled();
        console.log("AlwaysSampled deployed at:", address(deployment));
        // Register AlwaysSampled with the factory
        selectorFactory.registerSelectorImplementation(address(deployment));

        vm.stopBroadcast();

        // Log deployment and registration
        console.log("=== AlwaysSampled Deployment & Registration ===");
        console.log("AlwaysSampled implementation deployed at:", address(deployment));
        console.log("Registered with SelectorFactory at:", selectorFactoryAddress);
        console.log("Version:", version);
        console.log("Selector ID:", vm.toString(id));

    }
}
