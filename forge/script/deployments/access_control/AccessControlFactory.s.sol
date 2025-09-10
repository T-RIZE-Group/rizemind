// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import {AccessControlFactory} from "@rizemind-contracts/access/AccessControlFactory.sol";
import {DevOpsTools} from "../../Devops.sol";

contract DeployAccessControlFactory is Script {
    
    function run() external {
        address owner = vm.envAddress("ACCESS_CONTROL_FACTORY_OWNER");
        if (isDeployed()) {
            address deployedAddress = getDeployedAddress();
            console.log("AccessControlFactory already deployed at:", deployedAddress);
            AccessControlFactory accessControlImpl = new AccessControlFactory(owner);
            require(deployedAddress.codehash == address(accessControlImpl).codehash, "AccessControlFactory codehash mismatch");
            return;
        }
        
        vm.startBroadcast();

        AccessControlFactory accessControlFactory = new AccessControlFactory(owner);

        vm.stopBroadcast();

        console.log("=== AccessControlFactory Deployment ===");
        console.log("AccessControlFactory deployed at:", address(accessControlFactory));
    }
    
    function isDeployed() public view returns (bool) {
        try vm.envAddress("ACCESS_CONTROL_FACTORY") returns (address envAddress) {
            return true;
        } catch {
            try DevOpsTools.get_most_recent_deployment("AccessControlFactory", block.chainid) returns (address factoryAddress) {
                return true;
            } catch {
                return false;
            }
        }
    }

    
    function getDeployedAddress() public view returns (address accessControlFactoryAddress) {
        try vm.envAddress("ACCESS_CONTROL_FACTORY") returns (address envAddress) {
            accessControlFactoryAddress = envAddress;
        } catch {
            try DevOpsTools.get_most_recent_deployment("AccessControlFactory", block.chainid) returns (address factoryAddress) {
                accessControlFactoryAddress = factoryAddress;
            } catch {
                revert("AccessControlFactory not found. Deploy AccessControlFactory first or set ACCESS_CONTROL_FACTORY environment variable.");
            }
        }
    }
}
