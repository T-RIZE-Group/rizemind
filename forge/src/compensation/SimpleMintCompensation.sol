// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import {CompensationSent} from "./types.sol";
import {TrainerContributed} from "../contribution/types.sol";
import {ERC20Upgradeable} from "@openzeppelin-contracts-upgradeable-5.2.0/token/ERC20/ERC20Upgradeable.sol";
import {ICompensation} from "./types.sol";
import {EIP712} from "@openzeppelin-contracts-5.2.0/utils/cryptography/EIP712.sol";
import {AccessControlUpgradeable} from "@openzeppelin-contracts-upgradeable-5.2.0/access/AccessControlUpgradeable.sol";

contract SimpleMintCompensation is ERC20Upgradeable, ICompensation, EIP712, AccessControlUpgradeable {

    string private constant _VERSION = "simple-mint-compensation-v1.0.0";
    uint8 constant CONTRIBUTION_DECIMALS = 6;
    uint256 private _targetRewardsPerRound;
    mapping(uint256 => uint256) private _totalContributions;

    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    error BadRewards();

    constructor() EIP712("SimpleMintCompensation", _VERSION) {}

    function initialize(
        string memory name,
        string memory symbol,
        uint256 targetRewards,
        address initialAdmin,
        address minter
    ) external initializer {
        __SimpleMintCompensation_init(name, symbol, targetRewards);
        __AccessControl_init();
        _grantRole(DEFAULT_ADMIN_ROLE, initialAdmin);
        _grantRole(MINTER_ROLE, minter);
    }

    function __SimpleMintCompensation_init(
        string memory name,
        string memory symbol,
        uint256 maxRewards
    ) internal onlyInitializing {
        __ERC20_init(name, symbol);
        _targetRewardsPerRound = maxRewards;
    }

    function distribute(
        uint256 roundId,
        address[] memory trainers,
        uint64[] memory contributions
    ) external onlyRole(MINTER_ROLE) {
        _distribute(roundId, trainers, contributions);
    }

    function _distribute(
        uint256 roundId,
        address[] memory trainers,
        uint64[] memory contributions
    ) internal {
        uint256 nTrainers = trainers.length;
        uint256 totalContributions = 0;
        if (nTrainers != contributions.length) {
            revert BadRewards();
        }
        for (uint16 i = 0; i < nTrainers; i++) {
            address trainer = trainers[i];
            uint64 contribution = contributions[i];
            totalContributions += contribution;
            emit TrainerContributed(trainer, contribution);
            uint256 rewards = _calculateRewards(contribution);
            _mint(trainer, rewards);
            emit CompensationSent(trainer, rewards);
        }
        _totalContributions[roundId] += totalContributions;
    }


    function _calculateRewards(
        uint64 contribution
    ) internal view returns (uint256) {
        return
            (uint256(contribution) * _targetRewardsPerRound) /
            (10 ** CONTRIBUTION_DECIMALS);
    }
}
