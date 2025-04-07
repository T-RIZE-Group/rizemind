// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Script.sol";

contract Deployments is Script {
    function save(string memory name, address contractAddress) public {
        // Build the file path using simplified string.concat syntax.
        string memory projectRoot = vm.projectRoot();
        string memory chainDir = string.concat(vm.toString(block.chainid), "/");
        // Build the directory path.
        string memory dirPath = string.concat(
            projectRoot,
            "/output/",
            chainDir
        );
        // Ensure the directory exists.
        vm.createDir(dirPath, true);
        // Build the full file path.
        string memory filePath = string.concat(
            dirPath,
            string.concat(name, ".json")
        );

        // Convert the factory address to a hexadecimal string.
        string memory contractAddressStr = vm.toString(contractAddress);

        // Build the JSON object using vm.serializeString.
        // Here, we start with an empty base (i.e. "") and serialize the key "address" with its value.
        string memory finalJson = vm.serializeString(
            "",
            "address",
            contractAddressStr
        );

        // Write the JSON content to the file.
        vm.writeJson(finalJson, filePath);
    }
}
