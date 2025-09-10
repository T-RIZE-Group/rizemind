// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import {SwarmV1} from "@rizemind-contracts/swarm/SwarmV1.sol";
import {SwarmV1Factory} from "@rizemind-contracts/swarm/SwarmV1Factory.sol";
import {DeploySelectorFactory} from "./selectors/SelectorFactory.s.sol";
import {DeployCalculatorFactory} from "./calculators/CalculatorFactory.s.sol";
import {DeployAccessControlFactory} from "./access_control/AccessControlFactory.s.sol";
import {DeployCompensationFactory} from "./compensation/CompensationFactory.s.sol";

contract DeploySwarmV1Factory is Script {
    function run() external {
        // Deploy all required factories using their deployment scripts
        DeploySelectorFactory deploySelectorFactory = new DeploySelectorFactory();
        address selectorFactoryAddress = deploySelectorFactory.getDeployedAddress();
        
        DeployCalculatorFactory deployCalculatorFactory = new DeployCalculatorFactory();
        address calculatorFactoryAddress = deployCalculatorFactory.getDeployedAddress();
        
        DeployAccessControlFactory deployAccessControlFactory = new DeployAccessControlFactory();
        address accessControlFactoryAddress = deployAccessControlFactory.getDeployedAddress();
        
        DeployCompensationFactory deployCompensationFactory = new DeployCompensationFactory();
        address compensationFactoryAddress = deployCompensationFactory.getDeployedAddress();

        vm.startBroadcast();

        // Deploy the implementation contract.
        SwarmV1 swarmImpl = new SwarmV1();
        
        // Deploy the factory using the implementation address and all factory addresses.
        SwarmV1Factory swarmFactory = new SwarmV1Factory(
            address(swarmImpl), 
            selectorFactoryAddress,
            calculatorFactoryAddress,
            accessControlFactoryAddress,
            compensationFactoryAddress
        );

        vm.stopBroadcast();

        console.log("=== SwarmV1Factory Deployment ===");
        console.log("Swarm Implementation deployed at:", address(swarmImpl));
        console.log("SwarmV1Factory deployed at:", address(swarmFactory));
        console.log("SelectorFactory address:", selectorFactoryAddress);
        console.log("CalculatorFactory address:", calculatorFactoryAddress);
        console.log("AccessControlFactory address:", accessControlFactoryAddress);
        console.log("CompensationFactory address:", compensationFactoryAddress);
    }
}
