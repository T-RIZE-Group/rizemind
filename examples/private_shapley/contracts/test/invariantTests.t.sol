// // SPDX-License-Identifier: MIT
// pragma solidity ^0.8.20;

// import "forge-std/Test.sol";
// import "../src/ImprovedPrivateShapley.sol";
// import "../src/MockERC20.sol";

// contract PrivateShapleyInvariantTest is Test {
//     ImprovedPrivateShapley private privateShapley;
//     MockERC20 private mockToken;

//     // Test accounts
//     address private owner;
//     address[] private trainers;
//     address[] private testers;

//     // Test state
//     uint256 private roundId;
//     mapping(bytes32 => bool) private coalitionExists;
//     mapping(bytes32 => bytes32) private coalitionBitfields;
//     mapping(bytes32 => bytes32) private coalitionNonces;
//     mapping(bytes32 => bool) private coalitionRevealed;
//     mapping(address => mapping(bytes32 => bool)) private trainerHasClaimed;

//     // Counters for statistics
//     uint256 private commitCount;
//     uint256 private publishCount;
//     uint256 private revealCount;
//     uint256 private claimCount;

//     function setUp() public {
//         // Setup accounts
//         owner = address(this);

//         // Create trainers (8)
//         for (uint256 i = 0; i < 8; i++) {
//             trainers.push(address(uint160(0x1000 + i)));
//             vm.label(
//                 trainers[i],
//                 string(abi.encodePacked("Trainer", vm.toString(i)))
//             );
//         }

//         // Create testers (3)
//         for (uint256 i = 0; i < 3; i++) {
//             testers.push(address(uint160(0x2000 + i)));
//             vm.label(
//                 testers[i],
//                 string(abi.encodePacked("Tester", vm.toString(i)))
//             );
//         }

//         // Deploy token contract
//         mockToken = new MockERC20("Reward Token", "RWD");

//         // Deploy contract
//         privateShapley = new ImprovedPrivateShapley(address(mockToken));

//         // Mint tokens to contract
//         mockToken.mint(address(privateShapley), 1_000_000 * 10 ** 18);

//         // Register trainers
//         privateShapley.registerTrainers(trainers);

//         // Register testers
//         bool[] memory flags = new bool[](testers.length);
//         for (uint256 i = 0; i < testers.length; i++) {
//             flags[i] = true;
//         }
//         privateShapley.setTesters(testers, flags);

//         // Create round
//         roundId = 1;
//         uint256 currentTime = block.timestamp;
//         privateShapley.createRound(roundId, currentTime, currentTime + 30 days);
//     }

//     // Helper to generate random coalition data
//     function _generateRandomCoalition()
//         internal
//         returns (
//             bytes32 id,
//             bytes32 bitfield,
//             bytes32 nonce,
//             bytes32 commitment
//         )
//     {
//         // Generate random ID
//         id = keccak256(
//             abi.encodePacked(
//                 "coalition",
//                 commitCount,
//                 block.timestamp,
//                 block.prevrandao
//             )
//         );

//         // Create random bitfield (ensure at least one trainer)
//         uint256 bitMask = uint256(keccak256(abi.encodePacked("bitfield", id))) %
//             (2 ** trainers.length);
//         if (bitMask == 0) bitMask = 1; // Ensure at least one trainer
//         bitfield = bytes32(bitMask);

//         // Generate random nonce
//         nonce = keccak256(abi.encodePacked("nonce", id, block.prevrandao));

//         // Create commitment
//         commitment = keccak256(abi.encodePacked(bitfield, nonce));

//         // Store for invariant checks
//         coalitionExists[id] = true;
//         coalitionBitfields[id] = bitfield;
//         coalitionNonces[id] = nonce;
//     }

//     // Actions

//     function commitRandomCoalition() public {
//         // Skip if we've committed too many
//         if (commitCount >= 10) return;

//         // Generate random coalition
//         (
//             bytes32 id,
//             bytes32 bitfield,
//             bytes32 nonce,
//             bytes32 commitment
//         ) = _generateRandomCoalition();

//         // Create arrays for the function call
//         bytes32[] memory ids = new bytes32[](1);
//         bytes32[] memory commitments = new bytes32[](1);
//         ids[0] = id;
//         commitments[0] = commitment;

//         // Commit coalition
//         privateShapley.commitCoalitions(roundId, ids, commitments);
//         commitCount++;
//     }

//     function publishRandomResults() public {
//         // Skip if no coalitions exist
//         if (commitCount == 0) return;

//         // Find all committed coalitions
//         bytes32[] memory committedIds = new bytes32[](commitCount);
//         uint256 count = 0;

//         for (uint256 i = 0; i < 1000; i++) {
//             bytes32 potentialId = keccak256(
//                 abi.encodePacked("coalition", i, uint256(0), uint256(0))
//             );
//             if (coalitionExists[potentialId]) {
//                 (
//                     bytes32 commitment,
//                     ,
//                     ,
//                     ,
//                     bool isCommitted,
//                     ,
//                     ,

//                 ) = privateShapley.coalitionData(potentialId);

//                 if (isCommitted && commitment != bytes32(0)) {
//                     committedIds[count] = potentialId;
//                     count++;
//                     if (count >= commitCount) break;
//                 }
//             }
//         }

//         // Generate random scores
//         uint256[] memory scores = new uint256[](count);
//         for (uint256 i = 0; i < count; i++) {
//             scores[i] =
//                 100 +
//                 (uint256(
//                     keccak256(abi.encodePacked("score", i, block.timestamp))
//                 ) % 900);
//         }

//         // Skip if no committed coalitions found
//         if (count == 0) return;

//         // Trim arrays
//         bytes32[] memory ids = new bytes32[](count);
//         for (uint256 i = 0; i < count; i++) {
//             ids[i] = committedIds[i];
//         }

//         // Publish results
//         address tester = testers[
//             uint256(keccak256(abi.encodePacked("tester", block.timestamp))) %
//                 testers.length
//         ];
//         vm.prank(tester);
//         privateShapley.publishResults(roundId, ids, scores);
//         publishCount++;
//     }

//     function revealRandomCoalitions() public {
//         // Skip if no coalitions exist
//         if (commitCount == 0) return;

//         // Find all committed but unrevealed coalitions
//         bytes32[] memory committedIds = new bytes32[](commitCount);
//         bytes32[] memory bitfields = new bytes32[](commitCount);
//         bytes32[] memory nonces = new bytes32[](commitCount);
//         uint256 count = 0;

//         for (uint256 i = 0; i < 1000; i++) {
//             bytes32 potentialId = keccak256(
//                 abi.encodePacked("coalition", i, uint256(0), uint256(0))
//             );
//             if (
//                 coalitionExists[potentialId] && !coalitionRevealed[potentialId]
//             ) {
//                 (
//                     ,
//                     ,
//                     ,
//                     uint256 result,
//                     bool isCommitted,
//                     bool isRevealed,
//                     ,

//                 ) = privateShapley.coalitionData(potentialId);

//                 if (isCommitted && !isRevealed && result > 0) {
//                     committedIds[count] = potentialId;
//                     bitfields[count] = coalitionBitfields[potentialId];
//                     nonces[count] = coalitionNonces[potentialId];
//                     count++;
//                     if (count >= commitCount) break;
//                 }
//             }
//         }

//         // Skip if no valid coalitions found
//         if (count == 0) return;

//         // Trim arrays
//         bytes32[] memory ids = new bytes32[](count);
//         bytes32[] memory finalBitfields = new bytes32[](count);
//         bytes32[] memory finalNonces = new bytes32[](count);
//         for (uint256 i = 0; i < count; i++) {
//             ids[i] = committedIds[i];
//             finalBitfields[i] = bitfields[i];
//             finalNonces[i] = nonces[i];
//             coalitionRevealed[ids[i]] = true;
//         }

//         // Reveal coalitions
//         privateShapley.revealCoalitions(
//             roundId,
//             ids,
//             finalBitfields,
//             finalNonces
//         );
//         revealCount++;
//     }

//     function claimRandomRewards() public {
//         // Skip if no coalitions revealed
//         if (revealCount == 0) return;

//         // Pick a random trainer
//         uint256 trainerIdx = uint256(
//             keccak256(abi.encodePacked("trainerIdx", block.timestamp))
//         ) % trainers.length;
//         address trainer = trainers[trainerIdx];

//         // Find all revealed coalitions this trainer is part of and hasn't claimed
//         bytes32[] memory claimableIds = new bytes32[](commitCount);
//         uint256 count = 0;

//         for (uint256 i = 0; i < 1000; i++) {
//             bytes32 potentialId = keccak256(
//                 abi.encodePacked("coalition", i, uint256(0), uint256(0))
//             );
//             if (
//                 coalitionExists[potentialId] &&
//                 coalitionRevealed[potentialId] &&
//                 !trainerHasClaimed[trainer][potentialId]
//             ) {
//                 // Check if trainer is in coalition
//                 bytes32 bitfield = coalitionBitfields[potentialId];
//                 uint256 trainerBit = 1 << trainerIdx;

//                 if (uint256(bitfield) & trainerBit != 0) {
//                     claimableIds[count] = potentialId;
//                     count++;
//                     if (count >= commitCount) break;
//                 }
//             }
//         }

//         // Skip if no claimable coalitions found
//         if (count == 0) return;

//         // Trim array
//         bytes32[] memory ids = new bytes32[](count);
//         for (uint256 i = 0; i < count; i++) {
//             ids[i] = claimableIds[i];
//             trainerHasClaimed[trainer][ids[i]] = true;
//         }

//         // Claim rewards
//         vm.prank(trainer);
//         privateShapley.claimRewards(roundId, ids);
//         claimCount++;
//     }

//     // Invariant tests

//     function invariant_committedCoalitionsHaveValidCommitment() public {
//         for (uint256 i = 0; i < 1000; i++) {
//             bytes32 potentialId = keccak256(
//                 abi.encodePacked("coalition", i, uint256(0), uint256(0))
//             );
//             if (coalitionExists[potentialId]) {
//                 (
//                     bytes32 commitment,
//                     ,
//                     ,
//                     ,
//                     bool isCommitted,
//                     ,
//                     ,

//                 ) = privateShapley.coalitionData(potentialId);

//                 if (isCommitted) {
//                     // If committed, must have a non-zero commitment
//                     assertTrue(
//                         commitment != bytes32(0),
//                         "Committed coalition has zero commitment"
//                     );
//                 }
//             }
//         }
//     }

//     function invariant_revealedCoalitionsHaveValidBitfieldAndNonce() public {
//         for (uint256 i = 0; i < 1000; i++) {
//             bytes32 potentialId = keccak256(
//                 abi.encodePacked("coalition", i, uint256(0), uint256(0))
//             );
//             if (
//                 coalitionExists[potentialId] && coalitionRevealed[potentialId]
//             ) {
//                 (
//                     bytes32 commitment,
//                     bytes32 bitfield,
//                     bytes32 nonce,
//                     ,
//                     bool isCommitted,
//                     bool isRevealed,
//                     ,

//                 ) = privateShapley.coalitionData(potentialId);

//                 if (isRevealed) {
//                     // If revealed, must have matching commitment
//                     bytes32 expectedCommitment = keccak256(
//                         abi.encodePacked(bitfield, nonce)
//                     );
//                     assertEq(
//                         commitment,
//                         expectedCommitment,
//                         "Revealed coalition has invalid commitment"
//                     );

//                     // Bitfield must have at least one trainer
//                     assertTrue(
//                         uint256(bitfield) > 0,
//                         "Revealed coalition has no trainers"
//                     );
//                 }
//             }
//         }
//     }

//     function invariant_claimedRewardsAreOnlyForMemberTrainers() public {
//         for (
//             uint256 trainerIdx = 0;
//             trainerIdx < trainers.length;
//             trainerIdx++
//         ) {
//             address trainer = trainers[trainerIdx];

//             for (uint256 i = 0; i < 1000; i++) {
//                 bytes32 potentialId = keccak256(
//                     abi.encodePacked("coalition", i, uint256(0), uint256(0))
//                 );
//                 if (
//                     coalitionExists[potentialId] &&
//                     trainerHasClaimed[trainer][potentialId]
//                 ) {
//                     // If claimed, trainer must be a member
//                     bytes32 bitfield = coalitionBitfields[potentialId];
//                     uint256 trainerBit = 1 << trainerIdx;

//                     assertTrue(
//                         uint256(bitfield) & trainerBit != 0,
//                         "Trainer claimed rewards for coalition they're not part of"
//                     );

//                     // Check contract state
//                     bool claimed = privateShapley.trainerClaims(
//                         roundId,
//                         potentialId,
//                         trainer
//                     );
//                     assertTrue(
//                         claimed,
//                         "Contract doesn't show claim for trainer"
//                     );
//                 }
//             }
//         }
//     }

//     function invariant_resultsCannotBePublishedBeforeCommitment() public {
//         for (uint256 i = 0; i < 1000; i++) {
//             bytes32 potentialId = keccak256(
//                 abi.encodePacked("nonexistent", i, uint256(0), uint256(0))
//             );
//             if (!coalitionExists[potentialId]) {
//                 // Create arrays for the function call
//                 bytes32[] memory ids = new bytes32[](1);
//                 uint256[] memory scores = new uint256[](1);
//                 ids[0] = potentialId;
//                 scores[0] = 100;

//                 // Try to publish results for non-existent coalition
//                 vm.prank(testers[0]);
//                 vm.expectRevert("Coalition not committed");
//                 privateShapley.publishResults(roundId, ids, scores);

//                 // Only need to check one
//                 break;
//             }
//         }
//     }

//     function invariant_statisticsConsistency() public {
//         console.log("Test statistics:");
//         console.log("Commits:", commitCount);
//         console.log("Publishes:", publishCount);
//         console.log("Reveals:", revealCount);
//         console.log("Claims:", claimCount);

//         // Must have at least some operations to be meaningful
//         if (commitCount > 0) {
//             assertTrue(commitCount >= revealCount, "More reveals than commits");
//             assertTrue(revealCount >= claimCount, "More claims than reveals");
//         }
//     }
// }
