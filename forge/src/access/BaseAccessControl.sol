// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import {AccessControlUpgradeable} from "@openzeppelin-contracts-upgradeable-5.2.0/access/AccessControlUpgradeable.sol";
import {EIP712} from "@openzeppelin-contracts-5.2.0/utils/cryptography/EIP712.sol";
import {IAccessControl} from "./IAccessControl.sol";

contract BaseAccessControl is AccessControlUpgradeable, IAccessControl, EIP712 {
    bytes32 constant AGGREGATOR_ROLE = keccak256("AGGREGATOR");
    bytes32 constant TRAINER_ROLE = keccak256("TRAINER");
    bytes32 constant EVALUATOR_ROLE = keccak256("EVALUATOR_ROLE");

    string private constant _VERSION = "base-access-control-v1.0.0";

    constructor() EIP712("BaseAccessControl", _VERSION) {}

    modifier onlyTrainer(address account) {
        _checkTrainer(account);
        _;
    }

    modifier onlyAggregator(address account) {
        _checkAggregator(account);
        _;
    }

    modifier onlyEvaluator(address tester) {
        _checkEvaluator(tester);
        _;
    }

    function initialize(
        address aggregator,
        address[] memory initialTrainers,
        address[] memory initialEvaluators
    ) external initializer {
        __FLAccessControl_init(aggregator, initialTrainers, initialEvaluators);
    }

    function __FLAccessControl_init(
        address aggregator,
        address[] memory initialTrainers,
        address[] memory initialEvaluators
    ) internal onlyInitializing {
        _grantRole(AGGREGATOR_ROLE, aggregator);
        for (uint8 i = 0; i < initialTrainers.length; i++) {
            _grantRole(TRAINER_ROLE, initialTrainers[i]);
        }
        for (uint8 i = 0; i < initialEvaluators.length; i++) {
            _grantRole(EVALUATOR_ROLE, initialEvaluators[i]);
        }
    }

    function addTrainer(address trainer) public onlyAggregator(msg.sender) {
        _grantRole(TRAINER_ROLE, trainer);
    }

    function isTrainer(address trainer) public view returns (bool) {
        return hasRole(TRAINER_ROLE, trainer);
    }

    function _checkTrainer(address trainer) internal view virtual {
        _checkRole(TRAINER_ROLE, trainer);
    }

    function addAggregator(
        address aggregator
    ) public onlyAggregator(msg.sender) {
        _grantRole(AGGREGATOR_ROLE, aggregator);
    }

    function isAggregator(address aggregator) public view returns (bool) {
        return hasRole(AGGREGATOR_ROLE, aggregator);
    }

    function _checkAggregator(address aggregator) internal view virtual {
        _checkRole(AGGREGATOR_ROLE, aggregator);
    }

    function addEvaluator(address evaluator) public onlyAggregator(msg.sender) {
        _grantRole(EVALUATOR_ROLE, evaluator);
    }

    function isEvaluator(address evaluator) public view returns (bool) {
        return hasRole(EVALUATOR_ROLE, evaluator);
    }

    function _checkEvaluator(address evaluator) internal view virtual {
        _checkRole(EVALUATOR_ROLE, evaluator);
    }

    function supportsInterface(
        bytes4 interfaceId
    ) public view virtual override returns (bool) {
        return
            AccessControlUpgradeable.supportsInterface(interfaceId) ||
            interfaceId == type(IAccessControl).interfaceId ||
            interfaceId == this.addTrainer.selector ||
            interfaceId == this.isTrainer.selector ||
            interfaceId == this.addAggregator.selector ||
            interfaceId == this.isAggregator.selector ||
            interfaceId == this.addEvaluator.selector ||
            interfaceId == this.isEvaluator.selector;
    }
}
