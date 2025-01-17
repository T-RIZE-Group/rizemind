// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import {AccessControl} from "@openzeppelin/contracts/access/AccessControl.sol";

contract FLAccessControl is AccessControl {

  bytes32 constant AGGREGATOR_ROLE = keccak256("AGGREGATOR");
  bytes32 constant TRAINER_ROLE = keccak256("TRAINER");

  modifier onlyTrainer(address account) {
    _checkTrainer(account);
    _;
  }

  modifier onlyAggregator(address account) {
    _checkAggregator(account);
    _;
  }

  constructor(
    address[] memory initialTrainers
  ) {

  }

  function isTrainer(address trainer) public view returns(bool) {
    return hasRole(TRAINER_ROLE, trainer);
  }

  function _checkTrainer(address trainer) internal view virtual {
    _checkRole(TRAINER_ROLE, trainer);
  }

  function isAggregator(address aggregator) public view returns(bool) {
    return hasRole(AGGREGATOR_ROLE, aggregator);
  }

    function _checkAggregator(address aggregator) internal view virtual {
    _checkRole(AGGREGATOR_ROLE, aggregator);
  }

}