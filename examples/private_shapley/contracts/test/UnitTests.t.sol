// // SPDX-License-Identifier: MIT
// pragma solidity ^0.8.20;

// import "forge-std/Test.sol";
// import "../src/ImprovedPrivateShapley.sol";
// import "../src/MockERC20.sol";

// contract PrivateShapleyTest is Test {
//     ImprovedPrivateShapley private privateShapley;
//     MockERC20 private mockToken;

//     // Test accounts
//     address private owner;
//     address[] private trainers;
//     address[] private testers;
//     address private nonParticipant;

//     // Test variables
//     uint256 private roundId;
//     bytes32[] private coalitionIds;
//     bytes32[] private bitfields;
//     bytes32[] private nonces;
//     bytes32[] private commitments;

//     // Constants
//     uint256 private constant DAY = 1 days;

//     function setUp() public {
//         // Setup accounts
//         owner = address(this);

//         // Create trainers (5)
//         for (uint256 i = 0; i < 5; i++) {
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

//         nonParticipant = address(0x3000);
//         vm.label(nonParticipant, "NonParticipant");

//         // Deploy token contract
//         mockToken = new MockERC20("Reward Token", "RWD");

//         // Deploy contract
//         privateShapley = new ImprovedPrivateShapley(address(mockToken));

//         // Mint tokens to contract
//         mockToken.mint(address(privateShapley), 1_000_000 * 10 ** 18);

//         // Default roundId
//         roundId = 1;

//         // Generate test coalition data
//         generateCoalitionData(3, 5);
//     }

//     // Helper function to generate test coalition data
//     function generateCoalitionData(
//         uint256 numCoalitions,
//         uint256 numTrainers
//     ) private {
//         // Clear previous data
//         delete coalitionIds;
//         delete bitfields;
//         delete nonces;
//         delete commitments;

//         for (uint256 i = 0; i < numCoalitions; i++) {
//             // Generate random ID
//             bytes32 id = keccak256(
//                 abi.encodePacked("coalition", i, block.timestamp)
//             );
//             coalitionIds.push(id);

//             // Create bitfield - simple pattern based on index
//             uint256 bitMask;
//             if (i == 0) {
//                 // Coalition 0: Trainer 0 only
//                 bitMask = 1;
//             } else if (i == 1) {
//                 // Coalition 1: Trainer 1 only
//                 bitMask = 2;
//             } else {
//                 // Other coalitions: Mix of trainers
//                 bitMask = (1 << numTrainers) - 1; // All trainers
//             }

//             bitfields.push(bytes32(bitMask));

//             // Generate nonce
//             bytes32 nonce = keccak256(
//                 abi.encodePacked("nonce", i, block.timestamp)
//             );
//             nonces.push(nonce);

//             // Create commitment
//             bytes32 commitment = keccak256(
//                 abi.encodePacked(bitfields[i], nonces[i])
//             );
//             commitments.push(commitment);
//         }
//     }

//     // Helper to setup the basic scenario
//     function setupBasicScenario() public {
//         // Register trainers
//         privateShapley.registerTrainers(trainers);

//         // Register testers
//         bool[] memory flags = new bool[](testers.length);
//         for (uint256 i = 0; i < testers.length; i++) {
//             flags[i] = true;
//         }
//         privateShapley.setTesters(testers, flags);

//         // Create round
//         uint256 currentTime = block.timestamp;
//         privateShapley.createRound(roundId, currentTime, currentTime + DAY);
//     }

//     // Helper to complete lifecycle up to reveal
//     function completeLifecycleUpToReveal() public {
//         // First setup basic scenario
//         setupBasicScenario();

//         // Commit coalitions
//         privateShapley.commitCoalitions(roundId, coalitionIds, commitments);

//         // Publish results
//         uint256[] memory scores = new uint256[](coalitionIds.length);
//         for (uint256 i = 0; i < scores.length; i++) {
//             scores[i] = (i + 1) * 100; // 100, 200, 300, ...
//         }

//         vm.prank(testers[0]);
//         privateShapley.publishResults(roundId, coalitionIds, scores);

//         // Reveal coalitions
//         privateShapley.revealCoalitions(
//             roundId,
//             coalitionIds,
//             bitfields,
//             nonces
//         );
//     }

//     /* ========== TRAINER REGISTRATION TESTS ========== */

//     function testRegisterTrainers() public {
//         // Register trainers
//         address[] memory trainerAddresses = new address[](trainers.length);
//         for (uint256 i = 0; i < trainers.length; i++) {
//             trainerAddresses[i] = trainers[i];
//         }

//         privateShapley.registerTrainers(trainerAddresses);

//         // Check trainer count
//         assertEq(privateShapley.trainerCount(), trainers.length);

//         // Check mappings
//         for (uint256 i = 0; i < trainers.length; i++) {
//             assertEq(privateShapley.addressToIndex(trainers[i]), uint8(i + 1)); // 1-indexed
//             assertEq(privateShapley.indexToAddress(uint8(i + 1)), trainers[i]);
//         }
//     }

//     function testRegisterTrainersNonOwner() public {
//         vm.prank(nonParticipant);
//         vm.expectRevert(
//             abi.encodeWithSignature(
//                 "OwnableUnauthorizedAccount(address)",
//                 nonParticipant
//             )
//         );
//         privateShapley.registerTrainers(trainers);
//     }

//     function testRegisterZeroAddress() public {
//         address[] memory invalidTrainers = new address[](1);
//         invalidTrainers[0] = address(0);

//         vm.expectRevert("Invalid trainer address: zero address not allowed");
//         privateShapley.registerTrainers(invalidTrainers);
//     }

//     function testRegisterDuplicateTrainer() public {
//         // First registration
//         address[] memory singleTrainer = new address[](1);
//         singleTrainer[0] = trainers[0];
//         privateShapley.registerTrainers(singleTrainer);

//         // Second registration should fail
//         vm.expectRevert("Trainer already registered");
//         privateShapley.registerTrainers(singleTrainer);
//     }

//     /* ========== TESTER REGISTRATION TESTS ========== */

//     function testRegisterTesters() public {
//         // Register testers
//         bool[] memory flags = new bool[](testers.length);
//         for (uint256 i = 0; i < testers.length; i++) {
//             flags[i] = true;
//         }

//         privateShapley.setTesters(testers, flags);

//         // Check tester status
//         for (uint256 i = 0; i < testers.length; i++) {
//             assertTrue(privateShapley.isTester(testers[i]));
//         }
//     }

//     function testRegisterTestersArrayMismatch() public {
//         // Mismatched array lengths
//         bool[] memory flags = new bool[](testers.length + 1);

//         vm.expectRevert("Array lengths must match");
//         privateShapley.setTesters(testers, flags);
//     }

//     function testDisableTesters() public {
//         // First enable
//         bool[] memory flags = new bool[](1);
//         flags[0] = true;
//         address[] memory singleTester = new address[](1);
//         singleTester[0] = testers[0];

//         privateShapley.setTesters(singleTester, flags);
//         assertTrue(privateShapley.isTester(testers[0]));

//         // Then disable
//         flags[0] = false;
//         privateShapley.setTesters(singleTester, flags);
//         assertFalse(privateShapley.isTester(testers[0]));
//     }

//     /* ========== ROUND MANAGEMENT TESTS ========== */

//     function testCreateRound() public {
//         uint256 startTime = block.timestamp + 3600; // 1 hour from now
//         uint256 endTime = startTime + DAY; // 1 day duration

//         privateShapley.createRound(roundId, startTime, endTime);

//         // Check round data
//         (
//             uint256 actualStartTime,
//             uint256 actualEndTime,
//             bool isActive
//         ) = privateShapley.rounds(roundId);
//         assertEq(actualStartTime, startTime);
//         assertEq(actualEndTime, endTime);
//         assertTrue(isActive);
//     }

//     function testCreateRoundInvalidTimes() public {
//         uint256 startTime = block.timestamp + 3600;
//         uint256 endTime = startTime - 1800; // End before start

//         vm.expectRevert("End time must be after start time");
//         privateShapley.createRound(roundId, startTime, endTime);
//     }

//     function testCreateDuplicateRound() public {
//         uint256 startTime = block.timestamp;
//         uint256 endTime = startTime + DAY;

//         // First creation
//         privateShapley.createRound(roundId, startTime, endTime);

//         // Second creation should fail
//         vm.expectRevert("Round already exists");
//         privateShapley.createRound(roundId, startTime, endTime);
//     }

//     function testUpdateRound() public {
//         uint256 startTime = block.timestamp;
//         uint256 endTime = startTime + DAY;

//         // Create round
//         privateShapley.createRound(roundId, startTime, endTime);

//         // Update round
//         uint256 newStartTime = startTime + 1800;
//         uint256 newEndTime = endTime + 1800;

//         privateShapley.updateRound(roundId, newStartTime, newEndTime, true);

//         // Check updated data
//         (
//             uint256 actualStartTime,
//             uint256 actualEndTime,
//             bool isActive
//         ) = privateShapley.rounds(roundId);
//         assertEq(actualStartTime, newStartTime);
//         assertEq(actualEndTime, newEndTime);
//         assertTrue(isActive);
//     }

//     /* ========== COALITION LIFECYCLE TESTS ========== */

//     function testCommitCoalitions() public {
//         // Setup basic scenario
//         setupBasicScenario();

//         // Commit coalitions
//         privateShapley.commitCoalitions(roundId, coalitionIds, commitments);

//         // Check coalition data
//         for (uint256 i = 0; i < coalitionIds.length; i++) {
//             (
//                 bytes32 commitment,
//                 bytes32 bitfield,
//                 bytes32 nonce,
//                 uint256 result,
//                 bool isCommitted,
//                 bool isRevealed,
//                 uint256 commitTime,
//                 uint256 revealDeadline
//             ) = privateShapley.coalitionData(coalitionIds[i]);

//             assertEq(commitment, commitments[i]);
//             assertTrue(isCommitted);
//             assertFalse(isRevealed);
//             assertGt(revealDeadline, block.timestamp);
//         }
//     }

//     function testPublishResults() public {
//         // Setup and commit
//         setupBasicScenario();
//         privateShapley.commitCoalitions(roundId, coalitionIds, commitments);

//         // Publish results
//         uint256[] memory scores = new uint256[](coalitionIds.length);
//         for (uint256 i = 0; i < scores.length; i++) {
//             scores[i] = (i + 1) * 100; // 100, 200, 300, ...
//         }

//         vm.prank(testers[0]);
//         privateShapley.publishResults(roundId, coalitionIds, scores);

//         // Check results
//         for (uint256 i = 0; i < coalitionIds.length; i++) {
//             (
//                 bytes32 commitment,
//                 bytes32 bitfield,
//                 bytes32 nonce,
//                 uint256 result,
//                 bool isCommitted,
//                 bool isRevealed,
//                 uint256 commitTime,
//                 uint256 revealDeadline
//             ) = privateShapley.coalitionData(coalitionIds[i]);
//             assertEq(result, scores[i]);
//         }
//     }

//     function testAggregateResults() public {
//         // Setup and commit
//         setupBasicScenario();
//         privateShapley.commitCoalitions(roundId, coalitionIds, commitments);

//         // First tester publishes
//         uint256[] memory scores1 = new uint256[](coalitionIds.length);
//         for (uint256 i = 0; i < scores1.length; i++) {
//             scores1[i] = 100; // All 100
//         }

//         vm.prank(testers[0]);
//         privateShapley.publishResults(roundId, coalitionIds, scores1);

//         // Second tester publishes
//         uint256[] memory scores2 = new uint256[](coalitionIds.length);
//         for (uint256 i = 0; i < scores2.length; i++) {
//             scores2[i] = 200; // All 200
//         }

//         vm.prank(testers[1]);
//         privateShapley.publishResults(roundId, coalitionIds, scores2);

//         // Check aggregated results (should be average = 150)
//         for (uint256 i = 0; i < coalitionIds.length; i++) {
//             (
//                 bytes32 commitment,
//                 bytes32 bitfield,
//                 bytes32 nonce,
//                 uint256 result,
//                 bool isCommitted,
//                 bool isRevealed,
//                 uint256 commitTime,
//                 uint256 revealDeadline
//             ) = privateShapley.coalitionData(coalitionIds[i]);
//             assertEq(result, 150);
//         }
//     }

//     function testRevealCoalitions() public {
//         // Setup, commit, and publish
//         setupBasicScenario();
//         privateShapley.commitCoalitions(roundId, coalitionIds, commitments);

//         uint256[] memory scores = new uint256[](coalitionIds.length);
//         for (uint256 i = 0; i < scores.length; i++) {
//             scores[i] = (i + 1) * 100;
//         }

//         vm.prank(testers[0]);
//         privateShapley.publishResults(roundId, coalitionIds, scores);

//         // Reveal
//         privateShapley.revealCoalitions(
//             roundId,
//             coalitionIds,
//             bitfields,
//             nonces
//         );

//         // Check revealed data
//         for (uint256 i = 0; i < coalitionIds.length; i++) {
//             (
//                 bytes32 commitment,
//                 bytes32 bitfield,
//                 bytes32 nonce,
//                 uint256 result,
//                 bool isCommitted,
//                 bool isRevealed,
//                 ,

//             ) = privateShapley.coalitionData(coalitionIds[i]);

//             assertEq(bitfield, bitfields[i]);
//             assertEq(nonce, nonces[i]);
//             assertTrue(isRevealed);
//         }
//     }

//     function testRevealWithIncorrectBitfield() public {
//         // Setup, commit, and publish
//         setupBasicScenario();
//         privateShapley.commitCoalitions(roundId, coalitionIds, commitments);

//         uint256[] memory scores = new uint256[](coalitionIds.length);
//         for (uint256 i = 0; i < scores.length; i++) {
//             scores[i] = (i + 1) * 100;
//         }

//         vm.prank(testers[0]);
//         privateShapley.publishResults(roundId, coalitionIds, scores);

//         // Modify first bitfield
//         bytes32[] memory incorrectBitfields = new bytes32[](bitfields.length);
//         for (uint256 i = 0; i < bitfields.length; i++) {
//             incorrectBitfields[i] = bitfields[i];
//         }
//         incorrectBitfields[0] = bytes32(uint256(999));

//         // Reveal should fail
//         vm.expectRevert(
//             "Invalid commitment: bitfield and nonce do not match original commitment"
//         );
//         privateShapley.revealCoalitions(
//             roundId,
//             coalitionIds,
//             incorrectBitfields,
//             nonces
//         );
//     }

//     function testRevealAfterDeadline() public {
//         // Setup, commit, and publish
//         setupBasicScenario();
//         privateShapley.commitCoalitions(roundId, coalitionIds, commitments);

//         uint256[] memory scores = new uint256[](coalitionIds.length);
//         for (uint256 i = 0; i < scores.length; i++) {
//             scores[i] = (i + 1) * 100;
//         }

//         vm.prank(testers[0]);
//         privateShapley.publishResults(roundId, coalitionIds, scores);

//         // Advance time past reveal deadline
//         uint256 revealWindow = privateShapley.COMMIT_REVEAL_WINDOW();
//         vm.warp(block.timestamp + revealWindow + 1);

//         // Reveal should fail
//         vm.expectRevert("Round is not in progress");
//         privateShapley.revealCoalitions(
//             roundId,
//             coalitionIds,
//             bitfields,
//             nonces
//         );
//     }

//     /* ========== REWARD CLAIMING TESTS ========== */

//     function testClaimRewards() public {
//         // Complete lifecycle up to reveal
//         completeLifecycleUpToReveal();

//         // Trainer 0 claims from coalition 0 (where they are the only member)
//         bytes32[] memory claimIds = new bytes32[](1);
//         claimIds[0] = coalitionIds[0];

//         vm.prank(trainers[0]);
//         privateShapley.claimRewards(roundId, claimIds);

//         // Check claim is marked
//         assertTrue(
//             privateShapley.trainerClaims(roundId, coalitionIds[0], trainers[0])
//         );
//     }

//     function testClaimFromNonMemberCoalition() public {
//         // Complete lifecycle up to reveal
//         completeLifecycleUpToReveal();

//         // Trainer 0 tries to claim from coalition 1 (where they aren't a member)
//         bytes32[] memory claimIds = new bytes32[](1);
//         claimIds[0] = coalitionIds[1];

//         vm.prank(trainers[0]);
//         vm.expectRevert("Trainer is not a member of this coalition");
//         privateShapley.claimRewards(roundId, claimIds);
//     }

//     function testDoubleClaiming() public {
//         // Complete lifecycle up to reveal
//         completeLifecycleUpToReveal();

//         // First claim works
//         bytes32[] memory claimIds = new bytes32[](1);
//         claimIds[0] = coalitionIds[0];

//         vm.prank(trainers[0]);
//         privateShapley.claimRewards(roundId, claimIds);

//         // Second claim should fail
//         vm.prank(trainers[0]);
//         vm.expectRevert("Rewards for this coalition have already been claimed");
//         privateShapley.claimRewards(roundId, claimIds);
//     }

//     function testBatchClaiming() public {
//         // Complete lifecycle up to reveal
//         completeLifecycleUpToReveal();

//         // Trainer 0 claims from coalitions 0 and 2 (where they are a member)
//         bytes32[] memory claimIds = new bytes32[](2);
//         claimIds[0] = coalitionIds[0];
//         claimIds[1] = coalitionIds[2]; // All trainers coalition

//         vm.prank(trainers[0]);
//         privateShapley.claimRewards(roundId, claimIds);

//         // Check claims are marked
//         assertTrue(
//             privateShapley.trainerClaims(roundId, coalitionIds[0], trainers[0])
//         );
//         assertTrue(
//             privateShapley.trainerClaims(roundId, coalitionIds[2], trainers[0])
//         );
//     }

//     /* ========== SECURITY AND EDGE CASES ========== */

//     function testNonOwnerCreateRound() public {
//         vm.prank(nonParticipant);
//         vm.expectRevert(
//             abi.encodeWithSignature(
//                 "OwnableUnauthorizedAccount(address)",
//                 nonParticipant
//             )
//         );
//         privateShapley.createRound(
//             roundId,
//             block.timestamp,
//             block.timestamp + DAY
//         );
//     }

//     function testPublishBeforeCommit() public {
//         // Setup but don't commit
//         setupBasicScenario();

//         // Try to publish without commitment
//         uint256[] memory scores = new uint256[](coalitionIds.length);

//         vm.prank(testers[0]);
//         vm.expectRevert("Score below minimum threshold");
//         privateShapley.publishResults(roundId, coalitionIds, scores);
//     }

//     function testRevealBeforeCommit() public {
//         // Setup but don't commit
//         setupBasicScenario();

//         // Try to reveal without commitment
//         vm.expectRevert("Coalition must be committed before revealing");
//         privateShapley.revealCoalitions(
//             roundId,
//             coalitionIds,
//             bitfields,
//             nonces
//         );
//     }

//     function testClaimBeforeReveal() public {
//         // Setup and commit but don't reveal
//         setupBasicScenario();
//         privateShapley.commitCoalitions(roundId, coalitionIds, commitments);

//         // Try to claim without reveal
//         bytes32[] memory claimIds = new bytes32[](1);
//         claimIds[0] = coalitionIds[0];

//         vm.prank(trainers[0]);
//         vm.expectRevert("Coalition has not been revealed yet");
//         privateShapley.claimRewards(roundId, claimIds);
//     }

//     /* ========== GAS ANALYSIS ========== */

//     function testGasCommit() public {
//         setupBasicScenario();

//         // Measure gas for commit
//         uint256 gasStart = gasleft();
//         privateShapley.commitCoalitions(roundId, coalitionIds, commitments);
//         uint256 gasUsed = gasStart - gasleft();

//         console.log(
//             "Gas used for committing %d coalitions: %d",
//             coalitionIds.length,
//             gasUsed
//         );
//     }

//     function testGasPublish() public {
//         setupBasicScenario();
//         privateShapley.commitCoalitions(roundId, coalitionIds, commitments);

//         uint256[] memory scores = new uint256[](coalitionIds.length);
//         for (uint256 i = 0; i < scores.length; i++) {
//             scores[i] = 100;
//         }

//         // Measure gas for publish
//         vm.prank(testers[0]);
//         uint256 gasStart = gasleft();
//         privateShapley.publishResults(roundId, coalitionIds, scores);
//         uint256 gasUsed = gasStart - gasleft();

//         console.log(
//             "Gas used for publishing %d results: %d",
//             coalitionIds.length,
//             gasUsed
//         );
//     }

//     function testGasReveal() public {
//         setupBasicScenario();
//         privateShapley.commitCoalitions(roundId, coalitionIds, commitments);

//         uint256[] memory scores = new uint256[](coalitionIds.length);
//         for (uint256 i = 0; i < scores.length; i++) {
//             scores[i] = 100;
//         }

//         vm.prank(testers[0]);
//         privateShapley.publishResults(roundId, coalitionIds, scores);

//         // Measure gas for reveal
//         uint256 gasStart = gasleft();
//         privateShapley.revealCoalitions(
//             roundId,
//             coalitionIds,
//             bitfields,
//             nonces
//         );
//         uint256 gasUsed = gasStart - gasleft();

//         console.log(
//             "Gas used for revealing %d coalitions: %d",
//             coalitionIds.length,
//             gasUsed
//         );
//     }

//     function testGasClaim() public {
//         // Complete lifecycle up to reveal
//         completeLifecycleUpToReveal();

//         // Measure gas for claim
//         bytes32[] memory claimIds = new bytes32[](1);
//         claimIds[0] = coalitionIds[0];

//         vm.prank(trainers[0]);
//         uint256 gasStart = gasleft();
//         privateShapley.claimRewards(roundId, claimIds);
//         uint256 gasUsed = gasStart - gasleft();

//         console.log("Gas used for claiming 1 coalition: %d", gasUsed);
//     }
// }
