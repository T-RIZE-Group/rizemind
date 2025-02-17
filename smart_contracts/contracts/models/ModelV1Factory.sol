// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";

import {ModelRegistryV1} from "./ModelRegistryV1.sol";

contract ModelRegistryFactory {
    address private _logicContract;

    event ContractCreated(
        address indexed proxyAddress,
        address indexed owner,
        string name
    );
    event ProxyUpgraded(address indexed proxyAddress, address indexed newLogic);

    constructor(address logicContract) {
        _logicContract = logicContract;
    }

    function createModel(
        string memory name,
        string memory symbol,
        address aggregator,
        address[] memory initialTrainers
    ) external {
        bytes memory data = abi.encodeWithSelector(
            ModelRegistryV1.initialize.selector,
            name,
            symbol,
            aggregator,
            initialTrainers
        );

        ERC1967Proxy proxy = new ERC1967Proxy(_logicContract, data);

        emit ContractCreated(address(proxy), msg.sender, name);
    }

    function implementation() external returns (address) {
        return _logicContract;
    }
}
