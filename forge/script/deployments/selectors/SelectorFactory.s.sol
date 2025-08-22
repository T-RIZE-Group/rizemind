// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import {SelectorFactory} from "@rizemind-contracts/sampling/SelectorFactory.sol";
import {DevOpsTools} from "../../Devops.sol";

contract DeploySelectorFactory is Script {
    
    function run() external {
        if (isDeployed()) {
            address deployedAddress = getDeployedAddress();
            console.log("SelectorFactory already deployed at:", deployedAddress);
            SelectorFactory selectorImpl = new SelectorFactory(address(1));
            require(deployedAddress.codehash == address(selectorImpl).codehash, "SelectorFactory codehash mismatch");
            return;
        }
        
        vm.startBroadcast();

        address owner = vm.envAddress("SELECTOR_FACTORY_OWNER");
        SelectorFactory selectorFactory = new SelectorFactory(owner);

        vm.stopBroadcast();

        console.log("=== SelectorFactory Deployment ===");
        console.log("SelectorFactory deployed at:", address(selectorFactory));
    }
    
    function isDeployed() public view returns (bool) {
        try vm.envAddress("SELECTOR_FACTORY") returns (address envAddress) {
            return true;
        } catch {
            try DevOpsTools.get_most_recent_deployment("SelectorFactory", block.chainid) returns (address factoryAddress) {
                return true;
            } catch {
                return false;
            }
        }
    }

    
    function getDeployedAddress() public view returns (address selectorFactoryAddress) {
        try vm.envAddress("SELECTOR_FACTORY") returns (address envAddress) {
            selectorFactoryAddress = envAddress;
        } catch {
            try DevOpsTools.get_most_recent_deployment("SelectorFactory", block.chainid) returns (address factoryAddress) {
                selectorFactoryAddress = factoryAddress;
            } catch {
                revert("SelectorFactory not found. Deploy SelectorFactory first or set SELECTOR_FACTORY environment variable.");
            }
        }
    }
}
