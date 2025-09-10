// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import {CalculatorFactory} from "@rizemind-contracts/contribution/CalculatorFactory.sol";
import {DevOpsTools} from "../../Devops.sol";

contract DeployCalculatorFactory is Script {
    
    function run() external {
        address owner = vm.envAddress("CALCULATOR_FACTORY_OWNER");
        if (isDeployed()) {
            address deployedAddress = getDeployedAddress();
            console.log("CalculatorFactory already deployed at:", deployedAddress);
            CalculatorFactory calculatorImpl = new CalculatorFactory(owner);
            require(deployedAddress.codehash == address(calculatorImpl).codehash, "CalculatorFactory codehash mismatch");
            return;
        }
        
        vm.startBroadcast();

        CalculatorFactory calculatorFactory = new CalculatorFactory(owner);

        vm.stopBroadcast();

        console.log("=== CalculatorFactory Deployment ===");
        console.log("CalculatorFactory deployed at:", address(calculatorFactory));
    }
    
    function isDeployed() public view returns (bool) {
        try vm.envAddress("CALCULATOR_FACTORY") returns (address envAddress) {
            return true;
        } catch {
            try DevOpsTools.get_most_recent_deployment("CalculatorFactory", block.chainid) returns (address factoryAddress) {
                return true;
            } catch {
                return false;
            }
        }
    }

    
    function getDeployedAddress() public view returns (address calculatorFactoryAddress) {
        try vm.envAddress("CALCULATOR_FACTORY") returns (address envAddress) {
            calculatorFactoryAddress = envAddress;
        } catch {
            try DevOpsTools.get_most_recent_deployment("CalculatorFactory", block.chainid) returns (address factoryAddress) {
                calculatorFactoryAddress = factoryAddress;
            } catch {
                revert("CalculatorFactory not found. Deploy CalculatorFactory first or set CALCULATOR_FACTORY environment variable.");
            }
        }
    }
}
