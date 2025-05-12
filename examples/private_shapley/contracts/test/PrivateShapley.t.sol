// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

import "forge-std/Test.sol";
import "../src/PrivateShapley.sol";

contract PrivateShapleyTest is Test {
    PrivateShapley private privateShapley;
    
    address private coordinator;
    address private tester;
    address private trainer1;
    address private trainer2;
    address private trainer3;
    address private unregisteredTrainer;
    
    bytes32 private coalitionId;
    bytes32 private bitfield;
    bytes32 private merkleRoot;
    uint256 private roundId;
    
    // Trainer data
    struct TrainerData {
        uint8 index;
        bytes32 nonce;
        bytes32 leaf;
        bytes32[] proof;
    }
    
    mapping(address => TrainerData) private trainerData;
    
    function setUp() public {
        // Set up addresses
        coordinator = address(0x1);
        tester = address(0x2);
        trainer1 = address(0x3);
        trainer2 = address(0x4);
        trainer3 = address(0x5);
        unregisteredTrainer = address(0x6);
        
        // Deploy contract as coordinator
        vm.prank(coordinator);
        privateShapley = new PrivateShapley();
        
        // Register trainers
        vm.prank(coordinator);
        trainerData[trainer1].index = privateShapley.registerTrainer(trainer1);
        
        vm.prank(coordinator);
        trainerData[trainer2].index = privateShapley.registerTrainer(trainer2);
        
        vm.prank(coordinator);
        trainerData[trainer3].index = privateShapley.registerTrainer(trainer3);
        
        // Setup test data
        coalitionId = bytes32(uint256(1));
        roundId = 42;
        
        // Create trainer nonces
        trainerData[trainer1].nonce = bytes32(uint256(10));
        trainerData[trainer2].nonce = bytes32(uint256(20));
        trainerData[trainer3].nonce = bytes32(uint256(30));
        
        // Create leaf nodes (commitments)
        trainerData[trainer1].leaf = keccak256(abi.encodePacked(trainer1, trainerData[trainer1].nonce, roundId));
        trainerData[trainer2].leaf = keccak256(abi.encodePacked(trainer2, trainerData[trainer2].nonce, roundId));
        trainerData[trainer3].leaf = keccak256(abi.encodePacked(trainer3, trainerData[trainer3].nonce, roundId));
        
        // Build actual Merkle tree
        bytes32[] memory leaves = new bytes32[](3);
        leaves[0] = trainerData[trainer1].leaf;
        leaves[1] = trainerData[trainer2].leaf;
        leaves[2] = trainerData[trainer3].leaf;
        
        // Calculate intermediate nodes
        bytes32 node1_2 = hashPair(leaves[0], leaves[1]);
        bytes32 node3_3 = hashPair(leaves[2], leaves[2]); // Duplicate for odd number of nodes
        
        // Calculate root
        merkleRoot = hashPair(node1_2, node3_3);
        
        // Create proofs for each trainer
        // Proof for trainer1
        trainerData[trainer1].proof = new bytes32[](2);
        trainerData[trainer1].proof[0] = leaves[1]; // Sibling 1
        trainerData[trainer1].proof[1] = node3_3;   // Sibling 2
        
        // Proof for trainer2
        trainerData[trainer2].proof = new bytes32[](2);
        trainerData[trainer2].proof[0] = leaves[0]; // Sibling 1
        trainerData[trainer2].proof[1] = node3_3;   // Sibling 2
        
        // Proof for trainer3
        trainerData[trainer3].proof = new bytes32[](2);
        trainerData[trainer3].proof[0] = leaves[2]; // Since we duplicate leaf3, this is its own hash
        trainerData[trainer3].proof[1] = node1_2;   // Sibling 2
        
        // Create bitfield - set bits for trainer1 (index 1), trainer2 (index 2), and trainer3 (index 3)
        // 0x7 = 0b111 (first 3 bits set)
        bitfield = bytes32(uint256(7));
    }
    
    function hashPair(bytes32 a, bytes32 b) internal pure returns (bytes32) {
        return a < b ? keccak256(abi.encodePacked(a, b)) : keccak256(abi.encodePacked(b, a));
    }
    
    // -------------- Registration Tests --------------
    
    function testRegisterTrainer() public {
        address newTrainer = address(0x7);
        
        vm.prank(coordinator);
        uint8 index = privateShapley.registerTrainer(newTrainer);
        
        assertEq(index, 4); // Fourth trainer to be registered
        assertEq(privateShapley.addressToIndex(newTrainer), 4);
        assertEq(privateShapley.indexToAddress(4), newTrainer);
        assertTrue(privateShapley.isRegisteredTrainer(newTrainer));
    }
    
    function testCannotRegisterTrainerTwice() public {
        vm.prank(coordinator);
        vm.expectRevert("Trainer already registered");
        privateShapley.registerTrainer(trainer1);
    }
    
    function testOnlyOwnerCanRegisterTrainer() public {
        address newTrainer = address(0x7);
        
        vm.prank(tester);
        vm.expectRevert(abi.encodeWithSelector(Ownable.OwnableUnauthorizedAccount.selector, tester));
        privateShapley.registerTrainer(newTrainer);
    }
    
    // -------------- Coalition Data Publication Tests --------------
    
    function testPublishCoalitionData() public {
        vm.prank(coordinator);
        privateShapley.publishCoalitionData(coalitionId, bitfield, merkleRoot);
        
        (bytes32 storedBitfield, bytes32 storedRoot, bool isPublished) = privateShapley.coalitionData(coalitionId);
        
        assertEq(storedBitfield, bitfield);
        assertEq(storedRoot, merkleRoot);
        assertTrue(isPublished);
    }
    
    function testCannotPublishCoalitionDataTwice() public {
        vm.prank(coordinator);
        privateShapley.publishCoalitionData(coalitionId, bitfield, merkleRoot);
        
        vm.prank(coordinator);
        vm.expectRevert("Coalition data already published");
        privateShapley.publishCoalitionData(coalitionId, bitfield, merkleRoot);
    }
    
    function testOnlyOwnerCanPublishCoalitionData() public {
        vm.prank(tester);
        vm.expectRevert(abi.encodeWithSelector(Ownable.OwnableUnauthorizedAccount.selector, tester));
        privateShapley.publishCoalitionData(coalitionId, bitfield, merkleRoot);
    }
    
    // -------------- Result Publication Tests --------------
    
    function testPublishResult() public {
        uint256 result = 85;
        
        vm.prank(tester);
        privateShapley.publishResult(coalitionId, result);
        
        assertEq(privateShapley.coalitionResults(coalitionId), result);
    }
    
    function testCannotPublishResultTwice() public {
        uint256 result = 85;
        
        vm.prank(tester);
        privateShapley.publishResult(coalitionId, result);
        
        vm.prank(tester);
        vm.expectRevert("Result already published");
        privateShapley.publishResult(coalitionId, 90);
    }
    
    // -------------- Coalition Membership Tests --------------
    
    function testIsTrainerInCoalition() public {
        vm.prank(coordinator);
        privateShapley.publishCoalitionData(coalitionId, bitfield, merkleRoot);
        
        assertTrue(privateShapley.isTrainerInCoalition(coalitionId, trainer1));
        assertTrue(privateShapley.isTrainerInCoalition(coalitionId, trainer2));
        assertTrue(privateShapley.isTrainerInCoalition(coalitionId, trainer3));
        assertFalse(privateShapley.isTrainerInCoalition(coalitionId, unregisteredTrainer));
    }
    
    function testIsTrainerInCoalitionWithSpecificBitfield() public {
        // Create a bitfield with only trainer1 and trainer3 (not trainer2)
        // 0x5 = 0b101 (bits 0 and 2 set)
        bytes32 specificBitfield = bytes32(uint256(5));
        
        vm.prank(coordinator);
        privateShapley.publishCoalitionData(coalitionId, specificBitfield, merkleRoot);
        
        assertTrue(privateShapley.isTrainerInCoalition(coalitionId, trainer1));
        assertFalse(privateShapley.isTrainerInCoalition(coalitionId, trainer2));
        assertTrue(privateShapley.isTrainerInCoalition(coalitionId, trainer3));
    }
    
    // -------------- Claim Tests --------------
    
    function testSuccessfulClaim() public {
        // 1. Publish coalition data
        vm.prank(coordinator);
        privateShapley.publishCoalitionData(coalitionId, bitfield, merkleRoot);
        
        // 2. Publish result
        uint256 result = 85;
        vm.prank(tester);
        privateShapley.publishResult(coalitionId, result);
        
        // 3. Claim reward for trainer1
        vm.prank(trainer1);
        privateShapley.claimReward(
            roundId, 
            coalitionId, 
            trainerData[trainer1].nonce, 
            trainerData[trainer1].proof
        );
        
        // 4. Verify claim is recorded
        assertTrue(privateShapley.trainerClaims(roundId, coalitionId, trainer1));
        
        // 5. Verify nonce is marked as used
        assertTrue(privateShapley.nonceUsedInRound(roundId, trainerData[trainer1].nonce));
    }
    
    function testAllTrainersClaim() public {
        // 1. Publish coalition data
        vm.prank(coordinator);
        privateShapley.publishCoalitionData(coalitionId, bitfield, merkleRoot);
        
        // 2. Publish result
        uint256 result = 85;
        vm.prank(tester);
        privateShapley.publishResult(coalitionId, result);
        
        // 3. All trainers claim
        vm.prank(trainer1);
        privateShapley.claimReward(
            roundId, 
            coalitionId, 
            trainerData[trainer1].nonce, 
            trainerData[trainer1].proof
        );
        
        vm.prank(trainer2);
        privateShapley.claimReward(
            roundId, 
            coalitionId, 
            trainerData[trainer2].nonce, 
            trainerData[trainer2].proof
        );
        
        vm.prank(trainer3);
        privateShapley.claimReward(
            roundId, 
            coalitionId, 
            trainerData[trainer3].nonce, 
            trainerData[trainer3].proof
        );
        
        // 4. Verify all claims are recorded
        assertTrue(privateShapley.trainerClaims(roundId, coalitionId, trainer1));
        assertTrue(privateShapley.trainerClaims(roundId, coalitionId, trainer2));
        assertTrue(privateShapley.trainerClaims(roundId, coalitionId, trainer3));
    }
    
    // -------------- Claim Failure Tests --------------
    
    function testCannotClaimIfNotRegistered() public {
        // 1. Publish coalition data
        vm.prank(coordinator);
        privateShapley.publishCoalitionData(coalitionId, bitfield, merkleRoot);
        
        // 2. Publish result
        uint256 result = 85;
        vm.prank(tester);
        privateShapley.publishResult(coalitionId, result);
        
        // 3. Try to claim as unregistered trainer
        vm.prank(unregisteredTrainer);
        vm.expectRevert("Trainer not registered");
        privateShapley.claimReward(
            roundId, 
            coalitionId, 
            bytes32(uint256(40)), 
            new bytes32[](0)
        );
    }
    
    function testCannotClaimBeforeCoalitionDataPublished() public {
        // 1. Publish result (but not coalition data)
        uint256 result = 85;
        vm.prank(tester);
        privateShapley.publishResult(coalitionId, result);
        
        // 2. Try to claim
        vm.prank(trainer1);
        vm.expectRevert("Coalition data not published");
        privateShapley.claimReward(
            roundId, 
            coalitionId, 
            trainerData[trainer1].nonce, 
            trainerData[trainer1].proof
        );
    }
    
    function testCannotClaimBeforeResultPublished() public {
        // 1. Publish coalition data (but not result)
        vm.prank(coordinator);
        privateShapley.publishCoalitionData(coalitionId, bitfield, merkleRoot);
        
        // 2. Try to claim
        vm.prank(trainer1);
        vm.expectRevert("Coalition result not published");
        privateShapley.claimReward(
            roundId, 
            coalitionId, 
            trainerData[trainer1].nonce, 
            trainerData[trainer1].proof
        );
    }
    
    function testCannotClaimTwice() public {
        // 1. Publish coalition data
        vm.prank(coordinator);
        privateShapley.publishCoalitionData(coalitionId, bitfield, merkleRoot);
        
        // 2. Publish result
        uint256 result = 85;
        vm.prank(tester);
        privateShapley.publishResult(coalitionId, result);
        
        // 3. First claim succeeds
        vm.prank(trainer1);
        privateShapley.claimReward(
            roundId, 
            coalitionId, 
            trainerData[trainer1].nonce, 
            trainerData[trainer1].proof
        );
        
        // 4. Second claim fails
        vm.prank(trainer1);
        vm.expectRevert("Reward already claimed");
        privateShapley.claimReward(
            roundId, 
            coalitionId, 
            trainerData[trainer1].nonce, 
            trainerData[trainer1].proof
        );
    }
    
    function testCannotReuseNonce() public {
        // 1. Publish coalition data for two different coalitions
        vm.prank(coordinator);
        privateShapley.publishCoalitionData(coalitionId, bitfield, merkleRoot);
        
        bytes32 coalitionId2 = bytes32(uint256(2));
        vm.prank(coordinator);
        privateShapley.publishCoalitionData(coalitionId2, bitfield, merkleRoot);
        
        // 2. Publish results for both coalitions
        uint256 result = 85;
        vm.prank(tester);
        privateShapley.publishResult(coalitionId, result);
        
        vm.prank(tester);
        privateShapley.publishResult(coalitionId2, result);
        
        // 3. Claim for the first coalition succeeds
        vm.prank(trainer1);
        privateShapley.claimReward(
            roundId, 
            coalitionId, 
            trainerData[trainer1].nonce, 
            trainerData[trainer1].proof
        );
        
        // 4. Try to claim for the second coalition with same nonce
        vm.prank(trainer1);
        vm.expectRevert("Nonce already used in this round");
        privateShapley.claimReward(
            roundId, 
            coalitionId2, 
            trainerData[trainer1].nonce, 
            trainerData[trainer1].proof
        );
    }
    
    function testCannotClaimIfNotInBitfield() public {
        // Create a bitfield with only trainer1 and trainer3 (not trainer2)
        // 0x5 = 0b101 (bits 0 and 2 set)
        bytes32 specificBitfield = bytes32(uint256(5));
        
        // 1. Publish coalition data with specific bitfield
        vm.prank(coordinator);
        privateShapley.publishCoalitionData(coalitionId, specificBitfield, merkleRoot);
        
        // 2. Publish result
        uint256 result = 85;
        vm.prank(tester);
        privateShapley.publishResult(coalitionId, result);
        
        // 3. Trainer2 tries to claim but should fail because not in bitfield
        vm.prank(trainer2);
        vm.expectRevert("Trainer not in coalition bitfield");
        privateShapley.claimReward(
            roundId, 
            coalitionId, 
            trainerData[trainer2].nonce, 
            trainerData[trainer2].proof
        );
    }
    
    function testCannotClaimWithInvalidProof() public {
        // 1. Publish coalition data
        vm.prank(coordinator);
        privateShapley.publishCoalitionData(coalitionId, bitfield, merkleRoot);
        
        // 2. Publish result
        uint256 result = 85;
        vm.prank(tester);
        privateShapley.publishResult(coalitionId, result);
        
        // 3. Try to claim with invalid proof (use trainer2's proof for trainer1)
        vm.prank(trainer1);
        vm.expectRevert("Invalid Merkle proof");
        privateShapley.claimReward(
            roundId, 
            coalitionId, 
            trainerData[trainer1].nonce, 
            trainerData[trainer2].proof
        );
    }
    
    function testCannotClaimWithInvalidNonce() public {
        // 1. Publish coalition data
        vm.prank(coordinator);
        privateShapley.publishCoalitionData(coalitionId, bitfield, merkleRoot);
        
        // 2. Publish result
        uint256 result = 85;
        vm.prank(tester);
        privateShapley.publishResult(coalitionId, result);
        
        // 3. Try to claim with invalid nonce
        vm.prank(trainer1);
        vm.expectRevert("Invalid Merkle proof");
        privateShapley.claimReward(
            roundId, 
            coalitionId, 
            bytes32(uint256(999)), // Invalid nonce
            trainerData[trainer1].proof
        );
    }
    
    function testVerifyMerkleProof() public {
        for (uint i = 1; i <= 3; i++) {
            address trainer;
            if (i == 1) trainer = trainer1;
            else if (i == 2) trainer = trainer2;
            else trainer = trainer3;
            
            bytes32 leaf = trainerData[trainer].leaf;
            bytes32[] memory proof = trainerData[trainer].proof;
            
            bool isValid = privateShapley.verifyMerkleProof(proof, merkleRoot, leaf);
            assertTrue(isValid, string(abi.encodePacked("Merkle proof for trainer", uintToString(i), " failed")));
        }
    }
    
    function uintToString(uint v) internal pure returns (string memory) {
        uint maxlength = 100;
        bytes memory reversed = new bytes(maxlength);
        uint i = 0;
        while (v != 0) {
            uint remainder = v % 10;
            v = v / 10;
            reversed[i++] = bytes1(uint8(48 + remainder));
        }
        bytes memory s = new bytes(i);
        for (uint j = 0; j < i; j++) {
            s[j] = reversed[i - 1 - j];
        }
        return string(s);
    }
}