// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import { EIP712Upgradeable } from "@ozupgradeable/contracts/utils/cryptography/EIP712Upgradeable.sol";
import {Context} from "@openzeppelin/contracts/utils/Context.sol";
import { ContextUpgradeable } from "@ozupgradeable/contracts/utils/ContextUpgradeable.sol";

import { IModelRegistry, RoundSummary } from "./IModelRegistry.sol";
import { FLAccessControl } from "../access_control/FLAccessControl.sol";
import {SimpleContributionDistributor} from "../compensation/SimpleContributionDistributor.sol";

contract ModelRegistryV1 is IModelRegistry, FLAccessControl, SimpleContributionDistributor, EIP712Upgradeable {

  uint256 private _round = 0;

  event RoundFinished(uint256 indexed roundId, uint64 trainer, uint64 modelScore, uint128 totalContribution);

  error RoundMismatch(uint256 currentRound, uint256 givenRound);

  function initialize(    
    string memory name,
    string memory symbol,
    address aggregator,
    address[] memory initialTrainers
  ) public initializer {
    __EIP712_init(name, "1.0.0");
    __SimpleContributionDistributor_init(name, symbol, 10**20);
    __FLAccessControl_init(aggregator, initialTrainers);
  }

  function canTrain(address trainer, uint256 roundId) public returns (bool) {
    return isTrainer(trainer);
  }

  function curentRound() public view returns(uint256){
    return _round;
  }

  function nextRound(RoundSummary calldata summary) external {
    uint256 currentRound = _round;
    if(currentRound != summary.roundId) {
      revert RoundMismatch(currentRound, summary.roundId);
    }
    _round++;
    emit RoundFinished(summary.roundId, summary.nTrainers, summary.modelScore, summary.totalContributions);
  }

  function distribute(    
    address[] calldata trainers, 
    uint64[] calldata contributions
  ) external onlyAggregator(msg.sender) {
      _distribute(trainers, contributions);
  }

   function _msgSender() internal view virtual override(Context, ContextUpgradeable) returns (address) {
    return ContextUpgradeable._msgSender();
  }

  function _msgData() internal view virtual override(Context, ContextUpgradeable) returns (bytes calldata) {
    return ContextUpgradeable._msgData();
  }

  function _contextSuffixLength() internal view virtual override(Context, ContextUpgradeable) returns (uint256) {
    return ContextUpgradeable._contextSuffixLength();
  }

}