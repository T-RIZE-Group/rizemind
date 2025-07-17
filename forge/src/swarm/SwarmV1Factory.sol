// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ERC1967Proxy} from "@openzeppelin-contracts-5.2.0/proxy/ERC1967/ERC1967Proxy.sol";
import {AccessControl} from "@openzeppelin-contracts-5.2.0/access/AccessControl.sol";

import {SwarmV1} from "./SwarmV1.sol";

contract SwarmV1Factory is AccessControl {
    address private _logicContract;

    event ContractCreated(
        address indexed proxyAddress,
        address indexed owner,
        string name
    );
    event ProxyUpgraded(address indexed proxyAddress, address indexed newLogic);

    constructor(address logicContract) {
        _logicContract = logicContract;
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
    }

    function createSwarm(
        string memory name,
        string memory symbol,
        address aggregator,
        address[] memory initialTrainers
    ) external returns (address) {
        bytes memory data = abi.encodeWithSelector(
            SwarmV1.initialize.selector,
            name,
            symbol,
            aggregator,
            initialTrainers
        );

        ERC1967Proxy proxy = new ERC1967Proxy(_logicContract, data);

        emit ContractCreated(address(proxy), msg.sender, name);
        return address(proxy);
    }

    function getImplementation() external view returns (address) {
        return _logicContract;
    }

    function updateImplementation(
        address implementation
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(implementation != address(0), "implementation cannot be null");
        _logicContract = implementation;
    }
}
