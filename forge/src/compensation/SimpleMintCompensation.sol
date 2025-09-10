// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import {CompensationSent} from "./types.sol";
import {TrainerContributed} from "../contribution/types.sol";
import {ERC20Upgradeable} from "@openzeppelin-contracts-upgradeable-5.2.0/token/ERC20/ERC20Upgradeable.sol";

abstract contract SimpleMintCompensation is ERC20Upgradeable {
    uint8 constant CONTRIBUTION_DECIMALS = 6;
    uint256 private _maxRewards;

    error BadRewards();

    function __SimpleMintCompensation_init(
        string memory name,
        string memory symbol,
        uint256 maxRewards
    ) internal onlyInitializing {
        __ERC20_init(name, symbol);
        _maxRewards = maxRewards;
    }

    function _distribute(
        address[] memory trainers,
        uint64[] memory contributions
    ) internal {
        uint256 nTrainers = trainers.length;

        if (nTrainers != contributions.length) {
            revert BadRewards();
        }
        for (uint16 i = 0; i < nTrainers; i++) {
            address trainer = trainers[i];
            uint64 contribution = contributions[i];
            emit TrainerContributed(trainer, contribution);
            uint256 rewards = _calculateRewards(contribution);
            _mint(trainer, rewards);
            emit CompensationSent(trainer, rewards);
        }
    }

    function _calculateRewards(
        uint64 contribution
    ) internal view returns (uint256) {
        return
            (uint256(contribution) * _maxRewards) /
            (10 ** CONTRIBUTION_DECIMALS);
    }
}
