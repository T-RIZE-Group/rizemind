// SPDX-License-Identifier: MIT

//from https://github.com/Cyfrin/foundry-devops/blob/main/src/DevOpsTools.sol

pragma solidity >=0.8.13 <0.9.0;

import {Vm} from "forge-std/Vm.sol";
import {stdJson} from "forge-std/StdJson.sol";
import {StdCheatsSafe} from "forge-std/StdCheats.sol";
import {console} from "forge-std/console.sol";
import {StringUtils} from "./StringUtils.sol";

library DevOpsTools {
    using stdJson for string;
    using StringUtils for string;

    Vm public constant vm = Vm(address(uint160(uint256(keccak256("hevm cheat code")))));

    string public constant RELATIVE_BROADCAST_PATH = "./broadcast";

    function get_most_recent_deployment(string memory contractName, uint256 chainId) external view returns (address) {
        return get_most_recent_deployment(contractName, chainId, RELATIVE_BROADCAST_PATH);
    }

    function get_most_recent_deployment(
        string memory contractName,
        uint256 chainId,
        string memory relativeBroadcastPath
    ) public view returns (address) {
        address latestAddress = address(0);
        uint256 lastTimestamp;

        bool runProcessed;
        Vm.DirEntry[] memory entries = vm.readDir(relativeBroadcastPath, 3);
        for (uint256 i = 0; i < entries.length; i++) {
            string memory normalizedPath = normalizePath(entries[i].path);
            if (
                normalizedPath.contains(string.concat("/", vm.toString(chainId), "/"))
                    && normalizedPath.contains(".json") && !normalizedPath.contains("dry-run")
            ) {
                string memory json = vm.readFile(normalizedPath);
                latestAddress = processRun(json, contractName, latestAddress);
            }
        }
        for (uint256 i = 0; i < entries.length; i++) {
            Vm.DirEntry memory entry = entries[i];
            if (
                entry.path.contains(string.concat("/", vm.toString(chainId), "/")) && entry.path.contains(".json")
                    && !entry.path.contains("dry-run")
            ) {
                runProcessed = true;
                string memory json = vm.readFile(entry.path);

                uint256 timestamp = vm.parseJsonUint(json, ".timestamp");

                if (timestamp > lastTimestamp) {
                    address newLatestAddress = processRun(json, contractName, latestAddress);
                    if(newLatestAddress != address(0) && _isContract(newLatestAddress)) {
                        lastTimestamp = timestamp;
                        latestAddress = newLatestAddress;
                    }
                }
            }
        }

        if (!runProcessed) {
            revert("No deployment artifacts were found for specified chain");
        }

        if (latestAddress != address(0)) {
            return latestAddress;
        } else {
            revert(
                string.concat(
                    "No contract named ", "'", contractName, "'", " has been deployed on chain ", vm.toString(chainId)
                )
            );
        }
    }

    function processRun(string memory json, string memory contractName, address latestAddress)
        internal
        view
        returns (address)
    {
        for (uint256 i = 0; vm.keyExistsJson(json, string.concat("$.transactions[", vm.toString(i), "]")); i++) {
            string memory contractNamePath = string.concat("$.transactions[", vm.toString(i), "].contractName");
            if (vm.keyExistsJson(json, contractNamePath)) {
                string memory deployedContractName = json.readString(contractNamePath);
                if (deployedContractName.isEqualTo(contractName)) {
                    latestAddress =
                        json.readAddress(string.concat("$.transactions[", vm.toString(i), "].contractAddress"));
                }
            }
        }

        return latestAddress;
    }

    function normalizePath(string memory path) internal pure returns (string memory) {
        // Replace backslashes with forward slashes
        bytes memory b = bytes(path);
        for (uint256 i = 0; i < b.length; i++) {
            if (b[i] == bytes1("\\")) {
                b[i] = "/";
            }
        }
        return string(b);
    }
    
    /// @dev Checks if an address is a smart contract with bytecode
    /// @param addr The address to check
    /// @return True if the address is a smart contract with bytecode
    function _isContract(address addr) internal view returns (bool) {
        return addr.code.length > 0;
    }
}