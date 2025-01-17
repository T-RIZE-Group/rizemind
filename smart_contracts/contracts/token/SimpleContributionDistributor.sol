// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import { ERC20 } from "@openzeppelin/contracts/token/ERC20/ERC20.sol";

abstract contract SimpleContributionDistribution is ERC20 {

  uint8 constant CONTRIBUTION_DECIMALS = 6;
  uint256 private _maxRewards;

  error BadRewards();

  constructor(
    string memory name,
    string memory symbol,
    uint256 maxRewards
  ) ERC20(name, symbol) {
    _maxRewards = maxRewards;
  }
  
  function _distribute(
    address[] calldata trainers, 
    uint64[] calldata contributions
  ) internal {
    uint256 nTrainers = trainers.length;
    if(nTrainers != contributions.length) {
      revert BadRewards();
    }
    uint16 i = 0;
    for(i = 0; i < nTrainers; i++) {
      uint256 rewards = _calculateRewards(contributions[i]);
      _mint(trainers[i], rewards);
    }
  }

  function _calculateRewards(uint64 contribution) internal returns(uint256) {
    return uint256(contribution) * _maxRewards / CONTRIBUTION_DECIMALS;
  }
}