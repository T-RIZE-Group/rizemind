// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import {SimpleMintCompensation} from "@rizemind-contracts/compensation/SimpleMintCompensation.sol";
import {CompensationFactory} from "@rizemind-contracts/compensation/CompensationFactory.sol";
import {DeployCompensationFactory} from "./CompensationFactory.s.sol";

contract DeploySimpleMintCompensation is Script {
    
    function run() external {
        DeployCompensationFactory deployCompensationFactory = new DeployCompensationFactory();
        address compensationFactoryAddress = deployCompensationFactory.getDeployedAddress();
        
        CompensationFactory compensationFactory = CompensationFactory(compensationFactoryAddress);

        SimpleMintCompensation simpleMintCompensationImpl = new SimpleMintCompensation();
        (,, string memory version,,,,) = simpleMintCompensationImpl.eip712Domain();

        bytes32 id = compensationFactory.getID(version);
        // Check if SimpleMintCompensation is already registered
        if (compensationFactory.isCompensationRegistered(id)) {
            console.log("=== SimpleMintCompensation Already Registered ===");
            address existingImpl = compensationFactory.getCompensationImplementation(id);
            console.log("SimpleMintCompensation is already registered at:", existingImpl);
            console.log("Version:", version);
            console.log("Compensation ID:", vm.toString(id));
            return;
        }
        
        vm.startBroadcast();
        console.log("Deploying SimpleMintCompensation...");
        // Deploy SimpleMintCompensation implementation
        SimpleMintCompensation deployment = new SimpleMintCompensation();
        console.log("SimpleMintCompensation deployed at:", address(deployment));
        // Register SimpleMintCompensation with the factory
        compensationFactory.registerCompensationImplementation(address(deployment));

        vm.stopBroadcast();

        // Log deployment and registration
        console.log("=== SimpleMintCompensation Deployment & Registration ===");
        console.log("SimpleMintCompensation implementation deployed at:", address(deployment));
        console.log("Registered with CompensationFactory at:", compensationFactoryAddress);
        console.log("Version:", version);
        console.log("Compensation ID:", vm.toString(id));

    }
}
