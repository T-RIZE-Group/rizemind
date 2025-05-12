// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import {Deployments} from "./Deployments.sol";
import {PrivateShapley} from "../../../examples/private_shapley/private_shapley/contracts/src/privateShapley.sol";

contract DeployPrivateShapleyScript is Script, Deployments {
    function run() external {
        vm.startBroadcast();

        // Deploy the PrivateShapley contract
        PrivateShapley privateShapley = new PrivateShapley();

        vm.stopBroadcast();

        console.log("PrivateShapley deployed at:", address(privateShapley));
        save("PrivateShapley", address(privateShapley));
    }
}
