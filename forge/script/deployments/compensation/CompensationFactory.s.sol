// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import {CompensationFactory} from "@rizemind-contracts/compensation/CompensationFactory.sol";
import {DevOpsTools} from "../../Devops.sol";

contract DeployCompensationFactory is Script {
    
    function run() external {
        address owner = vm.envAddress("COMPENSATION_FACTORY_OWNER");
        if (isDeployed()) {
            address deployedAddress = getDeployedAddress();
            console.log("CompensationFactory already deployed at:", deployedAddress);
            CompensationFactory compensationImpl = new CompensationFactory(owner);
            require(deployedAddress.codehash == address(compensationImpl).codehash, "CompensationFactory codehash mismatch");
            return;
        }
        
        vm.startBroadcast();

        CompensationFactory compensationFactory = new CompensationFactory(owner);

        vm.stopBroadcast();

        console.log("=== CompensationFactory Deployment ===");
        console.log("CompensationFactory deployed at:", address(compensationFactory));
    }
    
    function isDeployed() public view returns (bool) {
        try vm.envAddress("COMPENSATION_FACTORY") returns (address envAddress) {
            return true;
        } catch {
            try DevOpsTools.get_most_recent_deployment("CompensationFactory", block.chainid) returns (address factoryAddress) {
                return true;
            } catch {
                return false;
            }
        }
    }

    
    function getDeployedAddress() public view returns (address compensationFactoryAddress) {
        try vm.envAddress("COMPENSATION_FACTORY") returns (address envAddress) {
            compensationFactoryAddress = envAddress;
        } catch {
            try DevOpsTools.get_most_recent_deployment("CompensationFactory", block.chainid) returns (address factoryAddress) {
                compensationFactoryAddress = factoryAddress;
            } catch {
                revert("CompensationFactory not found. Deploy CompensationFactory first or set COMPENSATION_FACTORY environment variable.");
            }
        }
    }
}
