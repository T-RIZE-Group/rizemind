// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ERC1967Proxy} from "@openzeppelin-contracts-5.2.0/proxy/ERC1967/ERC1967Proxy.sol";
import {AccessControl} from "@openzeppelin-contracts-5.2.0/access/AccessControl.sol";

import {SwarmV1} from "./SwarmV1.sol";
import {SelectorFactory} from "../sampling/SelectorFactory.sol";

// aderyn-ignore-next-line(centralization-risk)
contract SwarmV1Factory is AccessControl {
    address private _logicContract;
    address private _selectorFactory;

    event ContractCreated(
        address indexed proxyAddress,
        address indexed owner,
        string name
    );
    event ProxyUpgraded(address indexed proxyAddress, address indexed newLogic);

    struct SwarmV1Params {
        string name;
        string symbol;
        address aggregator;
        address[] initialTrainers;
        address initialTrainerSelector;
    }

    struct SelectorParams {
        bytes32 id;
        bytes initData;
    }

    struct SwarmParams {
        SwarmV1Params swarm;
        SelectorParams trainerSelector;
        SelectorParams evaluatorSelector;
    }

    constructor(address logicContract, address selectorFactory) {
        _logicContract = logicContract;
        _selectorFactory = selectorFactory;
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);

    }

    function createSwarm(
        SwarmParams memory params
    ) external returns (address) {
        SelectorFactory selectorFactory = SelectorFactory(_selectorFactory);
        bytes32 salt = keccak256(abi.encodePacked(params.swarm.name, params.swarm.symbol));
        address trainerSelector = selectorFactory.createSelector(
            params.trainerSelector.id,
            salt,
            params.trainerSelector.initData
        );

        address evaluatorSelector = selectorFactory.createSelector(
            params.evaluatorSelector.id,
            salt,
            params.evaluatorSelector.initData
        );

        bytes memory data = abi.encodeWithSelector(
            SwarmV1.initialize.selector,
            params.swarm.name,
            params.swarm.symbol,
            params.swarm.aggregator,
            params.swarm.initialTrainers,
            trainerSelector,
            evaluatorSelector
        );
        ERC1967Proxy proxy = new ERC1967Proxy(_logicContract, data);

        emit ContractCreated(address(proxy), msg.sender, params.swarm.name);
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
