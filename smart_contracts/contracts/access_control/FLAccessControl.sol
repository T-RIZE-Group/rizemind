// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import {AccessControl} from "@openzeppelin/contracts/access/AccessControl.sol";
import { Initializable } from "@ozupgradeable/contracts/proxy/utils/Initializable.sol";

contract FLAccessControl is AccessControl, Initializable {

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

  function __FLAccessControl_init(address aggregator, address[] memory initialTrainers) internal onlyInitializing {
    _grantRole(AGGREGATOR_ROLE, aggregator);
    for(uint8 i = 0; i < initialTrainers.length; i++) {
      _grantRole(TRAINER_ROLE, initialTrainers[i]);
    }
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

contract InitializableFLAccessControl is FLAccessControl {

  function initialize(address aggregator, address[] memory trainers) public initializer {
    __FLAccessControl_init(aggregator, trainers);
  }

}