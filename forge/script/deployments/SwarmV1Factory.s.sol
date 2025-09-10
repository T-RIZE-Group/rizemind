// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import {SwarmV1} from "@rizemind-contracts/swarm/SwarmV1.sol";
import {SwarmV1Factory} from "@rizemind-contracts/swarm/SwarmV1Factory.sol";
import {SelectorFactory} from "@rizemind-contracts/sampling/SelectorFactory.sol";
import {DeploySelectorFactory} from "./selectors/SelectorFactory.s.sol";

contract DeploySwarmV1Factory is Script {
    function run() external {
        DeploySelectorFactory deploySelectorFactory = new DeploySelectorFactory();
        address selectorFactoryAddress = deploySelectorFactory.getDeployedAddress();

        vm.startBroadcast();

        // Deploy the implementation contract.
        SwarmV1 swarmImpl = new SwarmV1();
    
        
        // Deploy the factory using the implementation address and selector factory.
        SwarmV1Factory swarmFactory = new SwarmV1Factory(address(swarmImpl), address(selectorFactoryAddress), address(0), address(0), address(0));

        vm.stopBroadcast();

        console.log("Swarm Implementation deployed at:", address(swarmImpl));
        console.log("Factory deployed at:", address(swarmFactory));
    }
}
