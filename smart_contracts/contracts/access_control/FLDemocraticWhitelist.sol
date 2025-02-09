// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FLDemocraticWhitelist {
    mapping(address => uint256) public trainerVotes; // Number of votes for each trainer
    mapping(address => mapping(address => bool)) private hasVotedForTrainer; // Tracks if a voter has voted for a trainer
    uint256 public approvalThreshold; // Minimum votes required for a trainer to be whitelisted

    event TrainerApproved(address indexed voter, address indexed trainer, uint256 totalVotes);
    event TrainerVoteRemoved(address indexed voter, address indexed trainer, uint256 totalVotes);

    constructor(uint256 _threshold) {
        approvalThreshold = _threshold;
    }

    function approveTrainer(address trainer) external {
        require(!hasVotedForTrainer[msg.sender][trainer], "Already approved this trainer");

        hasVotedForTrainer[msg.sender][trainer] = true;
        trainerVotes[trainer] += 1;

        emit TrainerApproved(msg.sender, trainer, trainerVotes[trainer]);
    }

    function revokeTrainerApproval(address trainer) external {
        require(hasVotedForTrainer[msg.sender][trainer], "No approval to revoke");

        hasVotedForTrainer[msg.sender][trainer] = false;
        trainerVotes[trainer] -= 1;

        emit TrainerVoteRemoved(msg.sender, trainer, trainerVotes[trainer]);
    }

    function isWhitelisted(address trainer) external view returns (bool) {
        return trainerVotes[trainer] >= approvalThreshold;
    }
}
