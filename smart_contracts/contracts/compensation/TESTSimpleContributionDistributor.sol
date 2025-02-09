// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import { SimpleContributionDistributor } from "./SimpleContributionDistributor.sol";

contract TESTSimpleContributionDistributor is SimpleContributionDistributor {

  function initialize(
    string memory name,
    string memory symbol,
    uint256 maxRewards
  ) public initializer {
    __SimpleContributionDistributor_init(name, symbol, maxRewards);
  }

  function distribute(
    address[] calldata trainers, 
    uint64[] calldata contributions
  ) external {
    _distribute(trainers, contributions);
  }
}