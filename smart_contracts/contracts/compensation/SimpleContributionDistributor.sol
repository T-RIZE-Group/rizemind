// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import { ERC20Upgradeable } from "@ozupgradeable/contracts/token/ERC20/ERC20Upgradeable.sol";

abstract contract SimpleContributionDistributor is ERC20Upgradeable {

  uint8 constant CONTRIBUTION_DECIMALS = 6;
  uint256 private _maxRewards;

  error BadRewards();

  function __SimpleContributionDistributor_init(
    string memory name,
    string memory symbol,
    uint256 maxRewards
  ) internal onlyInitializing {
    __ERC20_init(name, symbol);
    _maxRewards = maxRewards;
  }

  function _distribute(
    address[] calldata trainers, 
    uint64[] calldata contributions
  ) internal {
    uint256 nTrainers = trainers.length;
    if (nTrainers != contributions.length) {
      revert BadRewards();
    }
    for (uint16 i = 0; i < nTrainers; i++) {
      uint256 rewards = _calculateRewards(contributions[i]);
      _mint(trainers[i], rewards);
    }
  }

  function _calculateRewards(uint64 contribution) internal returns(uint256) {
    return uint256(contribution) * _maxRewards / (10 ** CONTRIBUTION_DECIMALS);
  }
}
