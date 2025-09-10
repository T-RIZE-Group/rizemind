// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import {BaseAccessControl} from "@rizemind-contracts/access/BaseAccessControl.sol";
import {AccessControlFactory} from "@rizemind-contracts/access/AccessControlFactory.sol";
import {DeployAccessControlFactory} from "./AccessControlFactory.s.sol";

contract DeployBaseAccessControl is Script {
    
    function run() external {
        DeployAccessControlFactory deployAccessControlFactory = new DeployAccessControlFactory();
        address accessControlFactoryAddress = deployAccessControlFactory.getDeployedAddress();
        
        AccessControlFactory accessControlFactory = AccessControlFactory(accessControlFactoryAddress);

        BaseAccessControl baseAccessControlImpl = new BaseAccessControl();
        (,, string memory version,,,,) = baseAccessControlImpl.eip712Domain();

        bytes32 id = accessControlFactory.getID(version);
        // Check if BaseAccessControl is already registered
        if (accessControlFactory.isAccessControlRegistered(id)) {
            console.log("=== BaseAccessControl Already Registered ===");
            address existingImpl = accessControlFactory.getAccessControlImplementation(id);
            console.log("BaseAccessControl is already registered at:", existingImpl);
            console.log("Version:", version);
            console.log("AccessControl ID:", vm.toString(id));
            return;
        }
        
        vm.startBroadcast();
        console.log("Deploying BaseAccessControl...");
        // Deploy BaseAccessControl implementation
        BaseAccessControl deployment = new BaseAccessControl();
        console.log("BaseAccessControl deployed at:", address(deployment));
        // Register BaseAccessControl with the factory
        accessControlFactory.registerAccessControlImplementation(address(deployment));

        vm.stopBroadcast();

        // Log deployment and registration
        console.log("=== BaseAccessControl Deployment & Registration ===");
        console.log("BaseAccessControl implementation deployed at:", address(deployment));
        console.log("Registered with AccessControlFactory at:", accessControlFactoryAddress);
        console.log("Version:", version);
        console.log("AccessControl ID:", vm.toString(id));

    }
}
