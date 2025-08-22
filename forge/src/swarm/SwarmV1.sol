// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {EIP712Upgradeable} from "@openzeppelin-contracts-upgradeable-5.2.0/utils/cryptography/EIP712Upgradeable.sol";
import {ContextUpgradeable} from "@openzeppelin-contracts-upgradeable-5.2.0/utils/ContextUpgradeable.sol";

import {FLAccessControl} from "../access/FLAccessControl.sol";
import {SimpleMintCompensation} from "../compensation/SimpleMintCompensation.sol";
import {RoundTraining} from "../training/RoundTraining.sol";
import {CertificateRegistry} from "./registry/CertificateRegistry.sol";
import {SwarmCore} from "./registry/SwarmCore.sol";
import {ISelector} from "../sampling/ISelector.sol";

contract SwarmV1 is
    FLAccessControl,
    SimpleMintCompensation,
    EIP712Upgradeable,
    RoundTraining,
    CertificateRegistry,
    SwarmCore
{
    string private constant _VERSION = "swarm-v1.0.0";

    function initialize(
        string memory name,
        string memory symbol,
        address aggregator,
        address[] memory initialTrainers,
        address initialTrainerSelector,
        address initialEvaluatorSelector
    ) external initializer {
        __EIP712_init(name, _VERSION);
        __SimpleMintCompensation_init(name, symbol, 10 ** 20);
        __FLAccessControl_init(aggregator, initialTrainers);
        __RoundTraining_init();
        __SwarmCore_init(initialTrainerSelector, initialEvaluatorSelector);
    }

    function canTrain(
        address trainer,
        uint256 roundId
    ) public view returns (bool) {
        ISelector selector = ISelector(getTrainerSelector());
        return isTrainer(trainer) && selector.isSelected(trainer, roundId);
    }

    function updateTrainerSelector(address newTrainerSelector) external onlyAggregator(msg.sender) {
        _updateTrainerSelector(newTrainerSelector);
    }

    function updateEvaluatorSelector(address newEvaluatorSelector) external onlyAggregator(msg.sender) {
        _updateEvaluatorSelector(newEvaluatorSelector);
    }

    function distribute(
        address[] calldata trainers,
        uint64[] calldata contributions
    ) external onlyAggregator(msg.sender) {
        _distribute(trainers, contributions);
    }

    function _msgSender()
        internal
        view
        virtual
        override(ContextUpgradeable)
        returns (address)
    {
        return ContextUpgradeable._msgSender();
    }

    function _msgData()
        internal
        view
        virtual
        override(ContextUpgradeable)
        returns (bytes calldata)
    {
        return ContextUpgradeable._msgData();
    }

    function _contextSuffixLength()
        internal
        view
        virtual
        override(ContextUpgradeable)
        returns (uint256)
    {
        return ContextUpgradeable._contextSuffixLength();
    }

    /**
     * @dev The version parameter for the EIP712 domain.
     */
    // solhint-disable-next-line func-name-mixedcase
    function _EIP712Version()
        internal
        pure
        override(EIP712Upgradeable)
        returns (string memory)
    {
        return _VERSION;
    }

    function supportsInterface(
        bytes4 interfaceId
    )
        public
        view
        virtual
        override(FLAccessControl, RoundTraining, CertificateRegistry)
        returns (bool)
    {
        return
            FLAccessControl.supportsInterface(interfaceId) ||
            RoundTraining.supportsInterface(interfaceId) ||
            CertificateRegistry.supportsInterface(interfaceId) ||
            interfaceId == this.canTrain.selector ||
            interfaceId == this.distribute.selector;
    }

    function setCertificate(
        bytes32 id,
        bytes calldata value
    ) external override onlyAggregator(msg.sender) {
        _setCertificate(id, value);
    }
}
