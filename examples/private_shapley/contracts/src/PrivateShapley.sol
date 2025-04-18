// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

import "@openzeppelin/contracts/access/Ownable.sol";

contract PrivateShapley is Ownable {
    // Mapping from coalition ID to its merkle root
    mapping(bytes32 => bytes32) public coalitionRoots;
    
    // Mapping from coalition ID to its result (score)
    mapping(bytes32 => uint256) public coalitionResults;
    
    // Mapping from trainer address + coalition ID to claimed status
    mapping(address => mapping(bytes32 => bool)) public trainerClaims;
    
    // Events
    event CoalitionRootPublished(bytes32 indexed coalitionId, bytes32 merkleRoot);
    event ResultPublished(bytes32 indexed coalitionId, uint256 result);
    event RewardClaimed(address indexed trainer, bytes32 indexed coalitionId, uint256 reward);
    
    constructor() Ownable(msg.sender) {}
    
    /**
     * @dev Publish a coalition's merkle root. Only callable by the coordinator (owner).
     * @param coalitionId The unique identifier for the coalition
     * @param merkleRoot The merkle root of the coalition's commitments
     */
    function publishCoalitionRoot(bytes32 coalitionId, bytes32 merkleRoot) external onlyOwner {
        require(coalitionRoots[coalitionId] == bytes32(0), "Coalition root already published");
        coalitionRoots[coalitionId] = merkleRoot;
        emit CoalitionRootPublished(coalitionId, merkleRoot);
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
        // Check this trainer hasn't already claimed for this coalition
        require(!trainerClaims[msg.sender][coalitionId], "Reward already claimed");
        
        // Check coalition has a published root
        bytes32 merkleRoot = coalitionRoots[coalitionId];
        require(merkleRoot != bytes32(0), "Coalition root not published");
        
        // Check coalition has a published result
        uint256 result = coalitionResults[coalitionId];
        require(result > 0, "Coalition result not published");
        
        // Calculate the leaf node from trainer address, nonce, and roundId
        bytes32 leaf = keccak256(abi.encodePacked(msg.sender, nonce, roundId));
        
        // Verify the Merkle proof
        require(verifyMerkleProof(merkleProof, merkleRoot, leaf), "Invalid Merkle proof");
        
        // Mark as claimed to prevent double-claiming
        trainerClaims[msg.sender][coalitionId] = true;
        
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
    ) internal pure returns (bool) {
        bytes32 computedHash = leaf;
        
        for (uint256 i = 0; i < proof.length; i++) {
            bytes32 proofElement = proof[i];
            
            if (computedHash <= proofElement) {
                // Hash(current computed hash + current element of the proof)
                computedHash = keccak256(abi.encodePacked(computedHash, proofElement));
            } else {
                // Hash(current element of the proof + current computed hash)
                computedHash = keccak256(abi.encodePacked(proofElement, computedHash));
            }
        }
        
        // Check if the computed hash equals the root of the Merkle tree
        return computedHash == root;
    }
}