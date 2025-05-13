// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

import "@openzeppelin/contracts/access/Ownable.sol";

contract PrivateShapley is Ownable {
    // Struct to hold coalition data
    struct CoalitionInfo {
        bytes32 bitfield; // 256-bit field with bit i set if trainer i is in coalition
        bytes32 merkleRoot; // Merkle root of trainer commitments
        bool isPublished; // Flag to check if data is published
    }

    // Maximum number of trainers (limited by bitfield size)
    uint8 public constant MAX_TRAINERS = 255;

    // Mapping of trainer address to their unique index
    mapping(address => uint8) public addressToIndex;

    // Mapping of index to trainer address
    mapping(uint8 => address) public indexToAddress;

    // Count of registered trainers
    uint8 public trainerCount;

    // Mapping from coalition ID to its information
    mapping(bytes32 => CoalitionInfo) public coalitionData;

    // Mapping from coalition ID to its result (score)
    mapping(bytes32 => uint256) public coalitionResults;

    // Mapping from round + coalition ID + trainer address to claimed status
    mapping(uint256 => mapping(bytes32 => mapping(address => bool)))
        public trainerClaims;

    // Mapping from round + nonce to used status (prevent nonce reuse)
    mapping(uint256 => mapping(bytes32 => bool)) public nonceUsedInRound;

    // Events
    event TrainerRegistered(address indexed trainer, uint8 index);
    event CoalitionDataPublished(
        bytes32 indexed coalitionId,
        bytes32 bitfield,
        bytes32 merkleRoot
    );
    event ResultPublished(bytes32 indexed coalitionId, uint256 result);
    event RewardClaimed(
        address indexed trainer,
        bytes32 indexed coalitionId,
        uint256 reward
    );

    constructor() Ownable(msg.sender) {}

    /**
     * @dev Register a new trainer and assign them a unique index.
     * @param trainer The address of the trainer to register
     * @return The assigned index
     */
    function registerTrainer(
        address trainer
    ) external onlyOwner returns (uint8) {
        require(trainer != address(0), "Invalid trainer address");
        require(addressToIndex[trainer] == 0, "Trainer already registered");
        require(trainerCount < MAX_TRAINERS, "Maximum trainers reached");

        // Increment trainer count (starts at 0, so first trainer gets index 1)
        // We reserve index 0 to mean "not registered"
        trainerCount++;

        // Assign the new index
        addressToIndex[trainer] = trainerCount;
        indexToAddress[trainerCount] = trainer;

        emit TrainerRegistered(trainer, trainerCount);
        return trainerCount;
    }

    /**
     * @dev Check if an address is a registered trainer.
     * @param trainer The address to check
     * @return True if registered, false otherwise
     */
    function isRegisteredTrainer(address trainer) public view returns (bool) {
        return addressToIndex[trainer] != 0;
    }

    /**
     * @dev Publish coalition data (bitfield and merkle root). Only callable by the coordinator (owner).
     * @param coalitionId The unique identifier for the coalition
     * @param bitfield The bitfield representing trainer membership in the coalition
     * @param merkleRoot The merkle root of the coalition's commitments
     */
    function publishCoalitionData(
        bytes32 coalitionId,
        bytes32 bitfield,
        bytes32 merkleRoot
    ) external onlyOwner {
        require(
            !coalitionData[coalitionId].isPublished,
            "Coalition data already published"
        );

        coalitionData[coalitionId] = CoalitionInfo({
            bitfield: bitfield,
            merkleRoot: merkleRoot,
            isPublished: true
        });

        emit CoalitionDataPublished(coalitionId, bitfield, merkleRoot);
    }

    /**
     * @dev Publish a result for a coalition. Can be called by anyone (e.g., the tester).
     * @param coalitionId The unique identifier for the coalition
     * @param result The evaluation score/result for the coalition
     */
    function publishResult(bytes32 coalitionId, uint256 result) external {
        require(coalitionResults[coalitionId] == 0, "Result already published");
        coalitionResults[coalitionId] = result;
        emit ResultPublished(coalitionId, result);
    }

    /**
     * @dev Claim a reward for a trainer that was part of a coalition.
     * @param roundId The ID of the training round
     * @param coalitionId The unique identifier for the coalition
     * @param nonce The secret nonce used by the trainer when creating the commitment
     * @param merkleProof The merkle proof showing the trainer was part of the coalition
     */
    function claimReward(
        uint256 roundId,
        bytes32 coalitionId,
        bytes32 nonce,
        bytes32[] calldata merkleProof
    ) external {
        // Check trainer is registered
        uint8 trainerIndex = addressToIndex[msg.sender];
        require(trainerIndex != 0, "Trainer not registered");

        // Check coalition data is published
        CoalitionInfo storage coalition = coalitionData[coalitionId];
        require(coalition.isPublished, "Coalition data not published");

        // Check coalition has a published result
        uint256 result = coalitionResults[coalitionId];
        require(result > 0, "Coalition result not published");

        // Check this trainer hasn't already claimed for this coalition
        require(
            !trainerClaims[roundId][coalitionId][msg.sender],
            "Reward already claimed"
        );

        // Check nonce hasn't been used in this round
        require(
            !nonceUsedInRound[roundId][nonce],
            "Nonce already used in this round"
        );

        // Check if trainer is in the coalition bitfield (check if bit at trainerIndex is set)
        bool isInBitfield = (uint256(coalition.bitfield) >>
            (trainerIndex - 1)) &
            1 ==
            1;
        require(isInBitfield, "Trainer not in coalition bitfield");

        // Calculate the leaf node from trainer address, nonce, and roundId
        bytes32 leaf = keccak256(abi.encodePacked(msg.sender, nonce, roundId));

        // Verify the Merkle proof
        require(
            verifyMerkleProof(merkleProof, coalition.merkleRoot, leaf),
            "Invalid Merkle proof"
        );

        // Mark nonce as used in this round
        nonceUsedInRound[roundId][nonce] = true;

        // Mark as claimed to prevent double-claiming
        trainerClaims[roundId][coalitionId][msg.sender] = true;

        // Calculate reward (in a real system, this would be more complex)
        uint256 reward = calculateReward(result);

        // In a real contract, this would transfer tokens
        // token.transfer(msg.sender, reward);

        emit RewardClaimed(msg.sender, coalitionId, reward);
    }

    /**
     * @dev Calculate a reward based on the coalition result.
     * @param result The coalition's evaluation result
     * @return The calculated reward amount
     */
    function calculateReward(uint256 result) internal pure returns (uint256) {
        // Simple example - in a real system this would be more sophisticated
        return result * 10; // Scale up the result to get a more substantial reward
    }

    /**
     * @dev Verify a Merkle proof.
     * @param proof The Merkle proof (array of sibling hashes)
     * @param root The Merkle root to verify against
     * @param leaf The leaf node being verified
     * @return True if the proof is valid, false otherwise
     */
    function verifyMerkleProof(
        bytes32[] calldata proof,
        bytes32 root,
        bytes32 leaf
    ) public pure returns (bool) {
        bytes32 computedHash = leaf;

        for (uint256 i = 0; i < proof.length; i++) {
            bytes32 proofElement = proof[i];

            if (computedHash <= proofElement) {
                // Hash(current computed hash + current element of the proof)
                computedHash = keccak256(
                    abi.encodePacked(computedHash, proofElement)
                );
            } else {
                // Hash(current element of the proof + current computed hash)
                computedHash = keccak256(
                    abi.encodePacked(proofElement, computedHash)
                );
            }
        }

        // Check if the computed hash equals the root of the Merkle tree
        return computedHash == root;
    }

    /**
     * @dev Check if a trainer is part of a coalition based on the bitfield.
     * @param coalitionId The unique identifier for the coalition
     * @param trainer The address of the trainer to check
     * @return True if the trainer is part of the coalition, false otherwise
     */
    function isTrainerInCoalition(
        bytes32 coalitionId,
        address trainer
    ) public view returns (bool) {
        uint8 trainerIndex = addressToIndex[trainer];
        if (trainerIndex == 0) return false; // Not a registered trainer

        CoalitionInfo storage coalition = coalitionData[coalitionId];
        if (!coalition.isPublished) return false; // Coalition data not published

        // Check if bit at trainerIndex is set in the bitfield
        return (uint256(coalition.bitfield) >> (trainerIndex - 1)) & 1 == 1;
    }
}
