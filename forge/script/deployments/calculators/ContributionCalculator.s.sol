// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import {ContributionCalculator} from "@rizemind-contracts/contribution/ContributionCalculator.sol";
import {CalculatorFactory} from "@rizemind-contracts/contribution/CalculatorFactory.sol";
import {DeployCalculatorFactory} from "./CalculatorFactory.s.sol";

contract DeployContributionCalculator is Script {
    
    function run() external {
        DeployCalculatorFactory deployCalculatorFactory = new DeployCalculatorFactory();
        address calculatorFactoryAddress = deployCalculatorFactory.getDeployedAddress();
        
        CalculatorFactory calculatorFactory = CalculatorFactory(calculatorFactoryAddress);

        ContributionCalculator contributionCalculatorImpl = new ContributionCalculator();
        (,, string memory version,,,,) = contributionCalculatorImpl.eip712Domain();

        bytes32 id = calculatorFactory.getID(version);
        // Check if ContributionCalculator is already registered
        if (calculatorFactory.isCalculatorRegistered(id)) {
            console.log("=== ContributionCalculator Already Registered ===");
            address existingImpl = calculatorFactory.getCalculatorImplementation(id);
            console.log("ContributionCalculator is already registered at:", existingImpl);
            console.log("Version:", version);
            console.log("Calculator ID:", vm.toString(id));
            return;
        }
        
        vm.startBroadcast();
        console.log("Deploying ContributionCalculator...");
        // Deploy ContributionCalculator implementation
        ContributionCalculator deployment = new ContributionCalculator();
        console.log("ContributionCalculator deployed at:", address(deployment));
        // Register ContributionCalculator with the factory
        calculatorFactory.registerCalculatorImplementation(address(deployment));

        vm.stopBroadcast();

        // Log deployment and registration
        console.log("=== ContributionCalculator Deployment & Registration ===");
        console.log("ContributionCalculator implementation deployed at:", address(deployment));
        console.log("Registered with CalculatorFactory at:", calculatorFactoryAddress);
        console.log("Version:", version);
        console.log("Calculator ID:", vm.toString(id));

    }
}
