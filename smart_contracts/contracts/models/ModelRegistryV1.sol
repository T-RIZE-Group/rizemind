// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import { EIP712 } from "@openzeppelin/contracts/utils/cryptography/EIP712.sol";

import { IModelRegistry, RoundSummary } from "./IModelRegistry.sol";
import { FLAccessControl } from "../access_control/FLAccessControl.sol";
import {SimpleContributionDistribution} from "../token/SimpleContributionDistributor.sol";

contract ModelRegistryV1 is IModelRegistry, FLAccessControl, SimpleContributionDistribution, EIP712 {

  uint256 private _round = 0;

  event RoundFinished(uint256 indexed roundId, uint64 trainer, uint64 modelScore, uint128 totalContribution);

  error RoundMismatch(uint256 currentRound, uint256 givenRound);
  constructor(
    string memory name,
    string memory symbol,
    address[] memory initialTrainers
  ) EIP712(name, "1.0.0") FLAccessControl(initialTrainers) SimpleContributionDistribution(name, symbol, 1**20){

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
    uint64[] calldata contributions) external onlyAggregator(msg.sender) {
      _distribute(trainers, contributions);
    }

}