// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import {SwarmV1} from "@rizemind-contracts/swarm/SwarmV1.sol";
import {SwarmV1Factory} from "@rizemind-contracts/swarm/SwarmV1Factory.sol";

contract DeploySwarmV1Factory is Script {
    function run() external {
        vm.startBroadcast();

        // Deploy the implementation contract.
        SwarmV1 swarmImpl = new SwarmV1();
        // Deploy the factory using the implementation address.
        SwarmV1Factory swarmFactory = new SwarmV1Factory(address(swarmImpl));

        vm.stopBroadcast();

        console.log("Swarm Implementation deployed at:", address(swarmImpl));
        console.log("Factory deployed at:", address(swarmFactory));
    }
}
