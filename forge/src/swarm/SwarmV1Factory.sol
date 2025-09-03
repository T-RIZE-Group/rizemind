// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ERC1967Proxy} from "@openzeppelin-contracts-5.2.0/proxy/ERC1967/ERC1967Proxy.sol";
import {AccessControl} from "@openzeppelin-contracts-5.2.0/access/AccessControl.sol";
import {Create2} from "@openzeppelin-contracts-5.2.0/utils/Create2.sol";
import {SwarmV1} from "./SwarmV1.sol";
import {SelectorFactory} from "../sampling/SelectorFactory.sol";
import {CalculatorFactory} from "../contribution/CalculatorFactory.sol";

// aderyn-ignore-next-line(centralization-risk)
contract SwarmV1Factory is AccessControl {
    address private _logicContract;
    address private _selectorFactory;
    address private _calculatorFactory;

    event ContractCreated(
        address indexed proxyAddress,
        address indexed owner,
        string name
    );
    event ProxyUpgraded(address indexed proxyAddress, address indexed newLogic);
    event CalculatorFactoryUpdated(address indexed oldFactory, address indexed newFactory);

    struct SwarmV1Params {
        string name;
        string symbol;
        address aggregator;
        address[] trainers;
    }

    struct SelectorParams {
        bytes32 id;
        bytes initData;
    }

    struct CalculatorParams {
        bytes32 id;
        bytes initData;
    }

    struct SwarmParams {
        SwarmV1Params swarm;
        SelectorParams trainerSelector;
        SelectorParams evaluatorSelector;
        CalculatorParams calculatorFactory;
    }

    constructor(address logicContract, address selectorFactory) {
        _logicContract = logicContract;
        _selectorFactory = selectorFactory;
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);

    }

    function createSwarm(
        bytes32 salt,
        SwarmParams memory params
    ) external returns (address) {
        bytes memory bytecode = abi.encodePacked(
            type(ERC1967Proxy).creationCode,
            abi.encode(_logicContract, bytes(""))
        );
        address proxy = Create2.deploy(0, salt, bytecode);
        
        SelectorFactory selectorFactory = SelectorFactory(_selectorFactory);
        bytes32 saltTrainerSelector = keccak256(abi.encodePacked(salt,"trainer-selector"));
        address trainerSelector = selectorFactory.createSelector(
            params.trainerSelector.id,
            saltTrainerSelector,
            params.trainerSelector.initData
        );

        bytes32 saltEvaluatorSelector = keccak256(abi.encodePacked(salt,"evaluator-selector"));
        address evaluatorSelector = selectorFactory.createSelector(
            params.evaluatorSelector.id,
            saltEvaluatorSelector,
            params.evaluatorSelector.initData
        );

        bytes32 saltCalculatorFactory = keccak256(abi.encodePacked(salt,"calculator-factory"));
        address calculatorFactory = CalculatorFactory(_calculatorFactory).createCalculator(
            params.calculatorFactory.id,
            saltCalculatorFactory,
            params.calculatorFactory.initData
        );

        SwarmV1 swarm = SwarmV1(proxy);
        swarm.initialize(
            params.swarm.name,
            params.swarm.symbol,
            params.swarm.aggregator,
            params.swarm.trainers,
            trainerSelector,
            evaluatorSelector,
            calculatorFactory
        );

        emit ContractCreated(proxy, msg.sender, params.swarm.name);
        return address(proxy);
    }

    function getSwarmAddress(
        bytes32 salt
    ) public view returns (address) {
        bytes memory bytecode = abi.encodePacked(
            type(ERC1967Proxy).creationCode,
            abi.encode(_logicContract, bytes(""))
        );
        return Create2.computeAddress(salt, keccak256(bytecode), address(this));
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

    /// @notice Get the current calculator factory address
    /// @return The address of the calculator factory
    function getCalculatorFactory() external view returns (address) {
        return _calculatorFactory;
    }

    /// @notice Set the calculator factory address
    /// @dev Only callable by DEFAULT_ADMIN_ROLE
    /// @param calculatorFactory The new calculator factory address
    function setCalculatorFactory(address calculatorFactory) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(calculatorFactory != address(0), "calculator factory cannot be null");
        
        address oldFactory = _calculatorFactory;
        _calculatorFactory = calculatorFactory;
        
        emit CalculatorFactoryUpdated(oldFactory, calculatorFactory);
    }
}
