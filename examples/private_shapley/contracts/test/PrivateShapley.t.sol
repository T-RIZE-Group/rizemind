// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "../src/ImprovedPrivateShapley.sol";
import "../src/MockERC20.sol";

contract ImprovedPrivateShapleyTest is Test {
    ImprovedPrivateShapley private privateShapley;
    MockERC20 private mockToken;

    // Test accounts
    address private owner;
    address[] private trainers;
    address[] private testers;
    address private nonParticipant;

    // Test variables
    uint256 private roundId;
    bytes32[] private coalitionIds;
    bytes32[] private bitfields;
    bytes32[] private nonces;
    bytes32[] private commitments;
    bytes32[] private trainerSalts;

    // Constants
    uint256 private constant DAY = 1 days;

    function setUp() public {
        // Setup accounts
        owner = address(this);

        // Create trainers (5)
        for (uint256 i = 0; i < 5; i++) {
            trainers.push(address(uint160(0x1000 + i)));
            vm.label(
                trainers[i],
                string(abi.encodePacked("Trainer", vm.toString(i)))
            );
        }

        // Create testers (3)
        for (uint256 i = 0; i < 3; i++) {
            testers.push(address(uint160(0x2000 + i)));
            vm.label(
                testers[i],
                string(abi.encodePacked("Tester", vm.toString(i)))
            );
        }

        nonParticipant = address(0x3000);
        vm.label(nonParticipant, "NonParticipant");

        // Deploy token contract
        mockToken = new MockERC20("Reward Token", "RWD");

        // Deploy contract
        privateShapley = new ImprovedPrivateShapley(address(mockToken));

        // Mint tokens to contract
        mockToken.mint(address(privateShapley), 1_000_000 * 10 ** 18);

        // Default roundId
        roundId = 1;

        // Generate test coalition data
        generateCoalitionData(3, 5);
    }

    // Helper function to generate test coalition data
    function generateCoalitionData(
        uint256 numCoalitions,
        uint256 numTrainers
    ) private {
        // Clear previous data
        delete coalitionIds;
        delete bitfields;
        delete nonces;
        delete commitments;
        delete trainerSalts;

        // Generate trainer salts
        for (uint256 i = 0; i < numTrainers; i++) {
            trainerSalts.push(
                keccak256(abi.encodePacked("salt", i, block.timestamp))
            );
        }

        for (uint256 i = 0; i < numCoalitions; i++) {
            // Generate random ID
            bytes32 id = keccak256(
                abi.encodePacked("coalition", i, block.timestamp)
            );
            coalitionIds.push(id);

            // Create bitfield - simple pattern based on index
            uint256 bitMask;
            if (i == 0) {
                bitMask = 1; // Coalition 0: Trainer 0 only
            } else if (i == 1) {
                bitMask = 2; // Coalition 1: Trainer 1 only
            } else {
                bitMask = (1 << numTrainers) - 1; // All trainers
            }

            bitfields.push(bytes32(bitMask));

            // Generate nonce
            bytes32 nonce = keccak256(
                abi.encodePacked("nonce", i, block.timestamp)
            );
            nonces.push(nonce);

            // Create commitment
            bytes32 commitment = keccak256(
                abi.encodePacked(bitfields[i], nonces[i])
            );
            commitments.push(commitment);
        }
    }

    // Helper to setup the basic scenario with dynamic mapping
    function setupBasicScenarioWithMapping() public {
        // First register trainers in the allow-list
        bool[] memory flags = new bool[](trainers.length);
        for (uint256 i = 0; i < trainers.length; i++) {
            flags[i] = true;
        }
        privateShapley.setTrainers(trainers, flags);

        // Register testers
        bool[] memory testerFlags = new bool[](testers.length);
        for (uint256 i = 0; i < testers.length; i++) {
            testerFlags[i] = true;
        }
        privateShapley.setTesters(testers, testerFlags);

        // Create round
        uint256 currentTime = block.timestamp;
        privateShapley.createRound(roundId, currentTime, currentTime + DAY);

        // Commit trainer mapping for the round
        bytes32 mappingCommitment = keccak256(
            abi.encodePacked(trainers, trainerSalts)
        );
        privateShapley.commitTrainerMapping(roundId, mappingCommitment);

        // Reveal trainer mapping
        privateShapley.revealTrainerMapping(roundId, trainers, trainerSalts);
    }

    // Helper to setup Shapley coalition values
    function setupShapleyValues() public {
        // For a simple 2-player case matching testCases.json
        uint256[][] memory coalitions = new uint256[][](4);
        uint256[] memory values = new uint256[](4);

        // Empty coalition
        coalitions[0] = new uint256[](0);
        values[0] = 0;

        // Player 1 only - v({1}) = 99
        coalitions[1] = new uint256[](1);
        coalitions[1][0] = 1;
        values[1] = 99_000000; // 99

        // Player 2 only - v({2}) = 88
        coalitions[2] = new uint256[](1);
        coalitions[2][0] = 2;
        values[2] = 88_000000; // 88

        // Both players - v({1,2}) = 77
        coalitions[3] = new uint256[](2);
        coalitions[3][0] = 1;
        coalitions[3][1] = 2;
        values[3] = 77_000000; // 77

        privateShapley.setShapleyCoalitionValues(roundId, coalitions, values);
    }

    // Helper to complete lifecycle up to reveal
    function completeLifecycleUpToReveal() public {
        // First setup basic scenario
        setupBasicScenarioWithMapping();

        // Setup Shapley values
        setupShapleyValues();

        // Commit coalitions
        privateShapley.commitCoalitions(roundId, coalitionIds, commitments);

        // Publish results
        uint256[] memory scores = new uint256[](coalitionIds.length);
        for (uint256 i = 0; i < scores.length; i++) {
            scores[i] = (i + 1) * 1000; // 1000, 2000, 3000, ...
        }

        vm.prank(testers[0]);
        privateShapley.publishResults(roundId, coalitionIds, scores);

        // Reveal coalitions
        privateShapley.revealCoalitions(
            roundId,
            coalitionIds,
            bitfields,
            nonces
        );
    }

    /* ========== TRAINER REGISTRATION TESTS ========== */

    function testSetTrainers() public {
        // Register trainers
        bool[] memory flags = new bool[](trainers.length);
        for (uint256 i = 0; i < trainers.length; i++) {
            flags[i] = true;
        }

        privateShapley.setTrainers(trainers, flags);

        // Check trainer status
        for (uint256 i = 0; i < trainers.length; i++) {
            assertTrue(privateShapley.isRegisteredTrainer(trainers[i]));
        }
    }

    function testSetTrainersNonOwner() public {
        bool[] memory flags = new bool[](trainers.length);
        for (uint256 i = 0; i < trainers.length; i++) {
            flags[i] = true;
        }

        vm.prank(nonParticipant);
        vm.expectRevert(
            abi.encodeWithSignature(
                "OwnableUnauthorizedAccount(address)",
                nonParticipant
            )
        );
        privateShapley.setTrainers(trainers, flags);
    }

    /* ========== TESTER REGISTRATION TESTS ========== */

    function testRegisterTesters() public {
        // Register testers
        bool[] memory flags = new bool[](testers.length);
        for (uint256 i = 0; i < testers.length; i++) {
            flags[i] = true;
        }

        privateShapley.setTesters(testers, flags);

        // Check tester status
        for (uint256 i = 0; i < testers.length; i++) {
            assertTrue(privateShapley.isTester(testers[i]));
        }
    }

    function testDisableTesters() public {
        // First enable
        bool[] memory flags = new bool[](1);
        flags[0] = true;
        address[] memory singleTester = new address[](1);
        singleTester[0] = testers[0];

        privateShapley.setTesters(singleTester, flags);
        assertTrue(privateShapley.isTester(testers[0]));

        // Then disable
        flags[0] = false;
        privateShapley.setTesters(singleTester, flags);
        assertFalse(privateShapley.isTester(testers[0]));
    }

    /* ========== ROUND MANAGEMENT TESTS ========== */

    function testCreateRound() public {
        uint256 startTime = block.timestamp + 3600;
        uint256 endTime = startTime + DAY;

        privateShapley.createRound(roundId, startTime, endTime);

        // Check round data
        (
            uint256 actualStartTime,
            uint256 actualEndTime,
            bool isActive
        ) = privateShapley.rounds(roundId);
        assertEq(actualStartTime, startTime);
        assertEq(actualEndTime, endTime);
        assertTrue(isActive);
    }

    function testCreateDuplicateRound() public {
        uint256 startTime = block.timestamp;
        uint256 endTime = startTime + DAY;

        // First creation
        privateShapley.createRound(roundId, startTime, endTime);

        // Second creation should fail
        vm.expectRevert("Round already exists");
        privateShapley.createRound(roundId, startTime, endTime);
    }

    /* ========== TRAINER MAPPING TESTS ========== */

    function testCommitAndRevealTrainerMapping() public {
        // Setup
        bool[] memory flags = new bool[](trainers.length);
        for (uint256 i = 0; i < trainers.length; i++) {
            flags[i] = true;
        }
        privateShapley.setTrainers(trainers, flags);

        // Create round
        uint256 currentTime = block.timestamp;
        privateShapley.createRound(roundId, currentTime, currentTime + DAY);

        // Commit mapping
        bytes32 commitment = keccak256(
            abi.encodePacked(trainers, trainerSalts)
        );
        privateShapley.commitTrainerMapping(roundId, commitment);

        // Check commitment
        (bytes32 storedCommitment, bool revealed) = privateShapley
            .mappingCommit(roundId);
        assertEq(storedCommitment, commitment);
        assertFalse(revealed);

        // Reveal mapping
        privateShapley.revealTrainerMapping(roundId, trainers, trainerSalts);

        // Check mappings
        for (uint256 i = 0; i < trainers.length; i++) {
            uint8 idx = privateShapley.roundAddrToIdx(roundId, trainers[i]);
            assertEq(idx, i + 1); // 1-based indexing
            assertEq(
                privateShapley.roundIdxToAddr(roundId, uint8(i + 1)),
                trainers[i]
            );
        }
    }

    function testRevealWithWrongCommitment() public {
        // Setup
        bool[] memory flags = new bool[](trainers.length);
        for (uint256 i = 0; i < trainers.length; i++) {
            flags[i] = true;
        }
        privateShapley.setTrainers(trainers, flags);

        // Create round
        uint256 currentTime = block.timestamp;
        privateShapley.createRound(roundId, currentTime, currentTime + DAY);

        // Commit with different data
        bytes32 commitment = keccak256("wrong commitment");
        privateShapley.commitTrainerMapping(roundId, commitment);

        // Try to reveal with correct data - should fail
        vm.expectRevert("commit mismatch");
        privateShapley.revealTrainerMapping(roundId, trainers, trainerSalts);
    }

    /* ========== COALITION LIFECYCLE TESTS ========== */

    function testCommitCoalitions() public {
        setupBasicScenarioWithMapping();

        // Commit coalitions
        privateShapley.commitCoalitions(roundId, coalitionIds, commitments);

        // Check coalition data
        for (uint256 i = 0; i < coalitionIds.length; i++) {
            (
                bytes32 commitment,
                bytes32 bitfield,
                bytes32 nonce,
                uint256 result,
                bool isCommitted,
                bool isRevealed,
                uint256 revealDeadline,
                uint256 sumScores,
                uint256 numScores
            ) = privateShapley.coalitionData(coalitionIds[i]);

            assertEq(commitment, commitments[i]);
            assertTrue(isCommitted);
            assertFalse(isRevealed);
            assertGt(revealDeadline, block.timestamp);
        }
    }

    function testPublishResults() public {
        setupBasicScenarioWithMapping();
        privateShapley.commitCoalitions(roundId, coalitionIds, commitments);

        // Publish results
        uint256[] memory scores = new uint256[](coalitionIds.length);
        for (uint256 i = 0; i < scores.length; i++) {
            scores[i] = (i + 1) * 100;
        }

        vm.prank(testers[0]);
        privateShapley.publishResults(roundId, coalitionIds, scores);

        // Check results
        for (uint256 i = 0; i < coalitionIds.length; i++) {
            (, , , uint256 result, , , , , ) = privateShapley.coalitionData(
                coalitionIds[i]
            );
            assertEq(result, scores[i]);
        }
    }

    function testRevealCoalitions() public {
        setupBasicScenarioWithMapping();
        privateShapley.commitCoalitions(roundId, coalitionIds, commitments);

        uint256[] memory scores = new uint256[](coalitionIds.length);
        for (uint256 i = 0; i < scores.length; i++) {
            scores[i] = (i + 1) * 100;
        }

        vm.prank(testers[0]);
        privateShapley.publishResults(roundId, coalitionIds, scores);

        // Reveal
        privateShapley.revealCoalitions(
            roundId,
            coalitionIds,
            bitfields,
            nonces
        );

        // Check revealed data
        for (uint256 i = 0; i < coalitionIds.length; i++) {
            (
                bytes32 commitment,
                bytes32 bitfield,
                bytes32 nonce,
                uint256 result,
                bool isCommitted,
                bool isRevealed,
                uint256 revealDeadline,
                uint256 sumScores,
                uint256 numScores
            ) = privateShapley.coalitionData(coalitionIds[i]);

            assertEq(bitfield, bitfields[i]);
            assertEq(nonce, nonces[i]);
            assertTrue(isRevealed);
        }
    }

    /* ========== SHAPLEY VALUE TESTS ========== */

    function testSetShapleyCoalitionValues() public {
        setupBasicScenarioWithMapping();

        // Setup coalition values
        uint256[][] memory coalitions = new uint256[][](2);
        uint256[] memory values = new uint256[](2);

        coalitions[0] = new uint256[](1);
        coalitions[0][0] = 1;
        values[0] = 50_000000;

        coalitions[1] = new uint256[](2);
        coalitions[1][0] = 1;
        coalitions[1][1] = 2;
        values[1] = 80_000000;

        privateShapley.setShapleyCoalitionValues(roundId, coalitions, values);

        // Check values were set
        uint256 mask1 = 1; // trainer 1
        uint256 mask2 = 3; // trainers 1 and 2
        assertEq(
            privateShapley.roundCoalitionValues(roundId, mask1),
            values[0]
        );
        assertEq(
            privateShapley.roundCoalitionValues(roundId, mask2),
            values[1]
        );
    }

    function testGetTrainerShapleyValueNegative() public {
        // First register trainers in the allow-list
        address[] memory testTrainers = new address[](2);
        testTrainers[0] = trainers[0];
        testTrainers[1] = trainers[1];

        bool[] memory flags = new bool[](2);
        flags[0] = true;
        flags[1] = true;

        privateShapley.setTrainers(testTrainers, flags);

        // Create round
        uint256 currentTime = block.timestamp;
        privateShapley.createRound(roundId, currentTime, currentTime + DAY);

        // Commit and reveal trainer mapping
        bytes32[] memory testSalts = new bytes32[](2);
        for (uint256 i = 0; i < 2; i++) {
            testSalts[i] = keccak256(abi.encodePacked("salt", roundId, i));
        }
        bytes32 mappingCommitment = keccak256(
            abi.encodePacked(testTrainers, testSalts)
        );
        privateShapley.commitTrainerMapping(roundId, mappingCommitment);
        privateShapley.revealTrainerMapping(roundId, testTrainers, testSalts);

        // Setup a case where Shapley values can be negative
        // Using "Two Player Case" from testCases.json
        uint256[][] memory coalitions = new uint256[][](4);
        uint256[] memory values = new uint256[](4);

        // Empty coalition
        coalitions[0] = new uint256[](0);
        values[0] = 0;

        // Player 1 only - v({1}) = 60
        coalitions[1] = new uint256[](1);
        coalitions[1][0] = 1;
        values[1] = 60_000000;

        // Player 2 only - v({2}) = 70
        coalitions[2] = new uint256[](1);
        coalitions[2][0] = 2;
        values[2] = 70_000000;

        // Both players - v({1,2}) = 5
        coalitions[3] = new uint256[](2);
        coalitions[3][0] = 1;
        coalitions[3][1] = 2;
        values[3] = 5_000000;

        privateShapley.setShapleyCoalitionValues(roundId, coalitions, values);

        // Get Shapley values
        (int256 value1, bool isPositive1) = privateShapley
            .getTrainerShapleyValue(roundId, testTrainers[0]);
        (int256 value2, bool isPositive2) = privateShapley
            .getTrainerShapleyValue(roundId, testTrainers[1]);

        // Expected: Player 1 = -2.5, Player 2 = 7.5
        // Calculation:
        // For Player 1: 0.5 * (60 - 0) + 0.5 * (5 - 70) = 30 - 32.5 = -2.5
        // For Player 2: 0.5 * (70 - 0) + 0.5 * (5 - 60) = 35 - 27.5 = 7.5
        assertFalse(isPositive1);
        assertEq(value1, 2_500000); // -2.5 (returned as positive with flag)

        assertTrue(isPositive2);
        assertEq(value2, 7_500000); // 7.5
    }

    /* ========== REWARD CLAIMING TESTS ========== */

    function testClaimRewards() public {
        completeLifecycleUpToReveal();

        // Trainer 0 claims from coalition 0
        bytes32[] memory claimIds = new bytes32[](1);
        claimIds[0] = coalitionIds[0];

        vm.prank(trainers[0]);
        privateShapley.claimRewards(roundId, claimIds, trainerSalts[0]);

        // Check claim is marked
        assertTrue(privateShapley.roundRewardClaimed(roundId, trainers[0]));
    }

    function testClaimWithWrongSalt() public {
        completeLifecycleUpToReveal();

        bytes32[] memory claimIds = new bytes32[](1);
        claimIds[0] = coalitionIds[0];

        vm.prank(trainers[0]);
        vm.expectRevert("bad salt");
        privateShapley.claimRewards(roundId, claimIds, bytes32("wrong salt"));
    }

    function testDoubleClaiming() public {
        completeLifecycleUpToReveal();

        bytes32[] memory claimIds = new bytes32[](1);
        claimIds[0] = coalitionIds[0];

        // First claim
        vm.prank(trainers[0]);
        privateShapley.claimRewards(roundId, claimIds, trainerSalts[0]);

        // Second claim should fail
        vm.prank(trainers[0]);
        vm.expectRevert("Already claimed");
        privateShapley.claimRewards(roundId, claimIds, trainerSalts[0]);
    }

    function testClaimFromNonMemberCoalition() public {
        completeLifecycleUpToReveal();

        // Trainer 0 tries to claim from coalition 1 (where they aren't a member)
        bytes32[] memory claimIds = new bytes32[](1);
        claimIds[0] = coalitionIds[1];

        vm.prank(trainers[0]);
        vm.expectRevert("Not member");
        privateShapley.claimRewards(roundId, claimIds, trainerSalts[0]);
    }

    /* ========== COMPREHENSIVE SHAPLEY VALUE TESTS ========== */

    function testShapleyValuesAgainstTestCases() public {
        // Test Case 1: Two Player Case Simple
        // v({1}) = 99, v({2}) = 88, v({1,2}) = 77
        // Expected: Player 1 = 44, Player 2 = 33
        testTwoPlayerCaseSimple();

        // Test Case 2: Two Player Case with negative values
        // v({1}) = 60, v({2}) = 70, v({1,2}) = 5
        // Expected: Player 1 = -2.5, Player 2 = 7.5
        testTwoPlayerCaseNegative();

        // Test Case 3: Three Player Case
        testThreePlayerCase();
    }

    function testTwoPlayerCaseSimple() private {
        // Reset and setup for this specific test
        uint256 testRound = 10;
        setupRoundWithNTrainers(testRound, 2);

        uint256[][] memory coalitions = new uint256[][](4);
        uint256[] memory values = new uint256[](4);

        // Empty coalition
        coalitions[0] = new uint256[](0);
        values[0] = 0;

        // Player 1 only - v({1}) = 99
        coalitions[1] = new uint256[](1);
        coalitions[1][0] = 1;
        values[1] = 99_000000;

        // Player 2 only - v({2}) = 88
        coalitions[2] = new uint256[](1);
        coalitions[2][0] = 2;
        values[2] = 88_000000;

        // Both players - v({1,2}) = 77
        coalitions[3] = new uint256[](2);
        coalitions[3][0] = 1;
        coalitions[3][1] = 2;
        values[3] = 77_000000;

        privateShapley.setShapleyCoalitionValues(testRound, coalitions, values);

        // Check Shapley values
        (int256 value1, bool isPositive1) = privateShapley
            .getTrainerShapleyValue(testRound, trainers[0]);
        (int256 value2, bool isPositive2) = privateShapley
            .getTrainerShapleyValue(testRound, trainers[1]);

        assertTrue(isPositive1);
        assertEq(value1, 44_000000);
        assertTrue(isPositive2);
        assertEq(value2, 33_000000);
    }

    function testTwoPlayerCaseNegative() private {
        uint256 testRound = 11;
        setupRoundWithNTrainers(testRound, 2);

        uint256[][] memory coalitions = new uint256[][](4);
        uint256[] memory values = new uint256[](4);

        coalitions[0] = new uint256[](0);
        values[0] = 0;

        coalitions[1] = new uint256[](1);
        coalitions[1][0] = 1;
        values[1] = 60_000000;

        coalitions[2] = new uint256[](1);
        coalitions[2][0] = 2;
        values[2] = 70_000000;

        coalitions[3] = new uint256[](2);
        coalitions[3][0] = 1;
        coalitions[3][1] = 2;
        values[3] = 5_000000;

        privateShapley.setShapleyCoalitionValues(testRound, coalitions, values);

        (int256 value1, bool isPositive1) = privateShapley
            .getTrainerShapleyValue(testRound, trainers[0]);
        (int256 value2, bool isPositive2) = privateShapley
            .getTrainerShapleyValue(testRound, trainers[1]);

        assertFalse(isPositive1);
        assertEq(value1, 2_500000); // -2.5
        assertTrue(isPositive2);
        assertEq(value2, 7_500000); // 7.5
    }

    function testThreePlayerCase() private {
        uint256 testRound = 12;
        setupRoundWithNTrainers(testRound, 3);

        // From testCases.json Three Player Case
        // Expected: Player 1 = 7.5, Player 2 = 36, Player 3 = 6.5
        uint256[][] memory coalitions = new uint256[][](8);
        uint256[] memory values = new uint256[](8);

        // Empty coalition
        coalitions[0] = new uint256[](0);
        values[0] = 0;

        // Single player coalitions
        coalitions[1] = new uint256[](1);
        coalitions[1][0] = 1;
        values[1] = 90_000000; // v({1}) = 90

        coalitions[2] = new uint256[](1);
        coalitions[2][0] = 2;
        values[2] = 70_000000; // v({2}) = 70

        coalitions[3] = new uint256[](1);
        coalitions[3][0] = 3;
        values[3] = 55_000000; // v({3}) = 55

        // Two player coalitions
        coalitions[4] = new uint256[](2);
        coalitions[4][0] = 1;
        coalitions[4][1] = 2;
        values[4] = 66_000000; // v({1,2}) = 66

        coalitions[5] = new uint256[](2);
        coalitions[5][0] = 1;
        coalitions[5][1] = 3;
        values[5] = 22_000000; // v({1,3}) = 22

        coalitions[6] = new uint256[](2);
        coalitions[6][0] = 2;
        coalitions[6][1] = 3;
        values[6] = 99_000000; // v({2,3}) = 99

        // All players
        coalitions[7] = new uint256[](3);
        coalitions[7][0] = 1;
        coalitions[7][1] = 2;
        coalitions[7][2] = 3;
        values[7] = 50_000000; // v({1,2,3}) = 50

        privateShapley.setShapleyCoalitionValues(testRound, coalitions, values);

        // Check Shapley values
        (int256 value1, bool isPositive1) = privateShapley
            .getTrainerShapleyValue(testRound, trainers[0]);
        (int256 value2, bool isPositive2) = privateShapley
            .getTrainerShapleyValue(testRound, trainers[1]);
        (int256 value3, bool isPositive3) = privateShapley
            .getTrainerShapleyValue(testRound, trainers[2]);

        assertTrue(isPositive1);
        assertApproxEqAbs(value1, 7_500000, 100); // 7.5 ± 0.0001
        assertTrue(isPositive2);
        assertApproxEqAbs(value2, 36_000000, 100); // 36 ± 0.0001
        assertTrue(isPositive3);
        assertApproxEqAbs(value3, 6_500000, 100); // 6.5 ± 0.0001
    }

    function setupRoundWithNTrainers(uint256 testRound, uint256 n) private {
        // Register n trainers
        address[] memory testTrainers = new address[](n);
        bytes32[] memory testSalts = new bytes32[](n);
        bool[] memory flags = new bool[](n);

        for (uint256 i = 0; i < n; i++) {
            testTrainers[i] = trainers[i];
            testSalts[i] = keccak256(abi.encodePacked("salt", testRound, i));
            flags[i] = true;
        }

        privateShapley.setTrainers(testTrainers, flags);

        // Create round
        uint256 currentTime = block.timestamp;
        privateShapley.createRound(testRound, currentTime, currentTime + DAY);

        // Commit and reveal mapping
        bytes32 mappingCommitment = keccak256(
            abi.encodePacked(testTrainers, testSalts)
        );
        privateShapley.commitTrainerMapping(testRound, mappingCommitment);
        privateShapley.revealTrainerMapping(testRound, testTrainers, testSalts);
    }

    function testIsTrainerInCoalition() public {
        completeLifecycleUpToReveal();

        // Trainer 0 should be in coalition 0
        assertTrue(
            privateShapley.isTrainerInCoalition(
                coalitionIds[0],
                trainers[0],
                roundId
            )
        );

        // Trainer 0 should not be in coalition 1
        assertFalse(
            privateShapley.isTrainerInCoalition(
                coalitionIds[1],
                trainers[0],
                roundId
            )
        );
    }

    function testGetCoalitionResult() public {
        completeLifecycleUpToReveal();

        (uint256 score, uint256 testerCount) = privateShapley
            .getCoalitionResult(coalitionIds[0]);
        assertEq(score, 1000); // First coalition score
        assertEq(testerCount, 1); // One tester
    }

    function testClaimFailsWithNegativeShapley() public {
        // Setup round with negative Shapley values
        uint256 testRound = 20;
        setupRoundWithNTrainers(testRound, 2);

        // Register tester
        bool[] memory testerFlags = new bool[](1);
        testerFlags[0] = true;
        address[] memory singleTester = new address[](1);
        singleTester[0] = testers[0];
        privateShapley.setTesters(singleTester, testerFlags);

        // Set up coalition values that result in negative Shapley
        uint256[][] memory coalitions = new uint256[][](4);
        uint256[] memory values = new uint256[](4);

        coalitions[0] = new uint256[](0);
        values[0] = 0;

        coalitions[1] = new uint256[](1);
        coalitions[1][0] = 1;
        values[1] = 60_000000;

        coalitions[2] = new uint256[](1);
        coalitions[2][0] = 2;
        values[2] = 70_000000;

        coalitions[3] = new uint256[](2);
        coalitions[3][0] = 1;
        coalitions[3][1] = 2;
        values[3] = 5_000000;

        privateShapley.setShapleyCoalitionValues(testRound, coalitions, values);

        // Create and reveal a coalition
        bytes32 cId = keccak256("test-coalition");
        bytes32 cNonce = keccak256("test-nonce");
        bytes32 cBitfield = bytes32(uint256(3)); // Both trainers
        bytes32 cCommitment = keccak256(abi.encodePacked(cBitfield, cNonce));

        bytes32[] memory ids = new bytes32[](1);
        bytes32[] memory comms = new bytes32[](1);
        ids[0] = cId;
        comms[0] = cCommitment;

        privateShapley.commitCoalitions(testRound, ids, comms);

        // Publish result
        uint256[] memory scores = new uint256[](1);
        scores[0] = 1000;
        vm.prank(testers[0]);
        privateShapley.publishResults(testRound, ids, scores);

        // Reveal
        bytes32[] memory bfs = new bytes32[](1);
        bytes32[] memory ncs = new bytes32[](1);
        bfs[0] = cBitfield;
        ncs[0] = cNonce;
        privateShapley.revealCoalitions(testRound, ids, bfs, ncs);

        // Try to claim with trainer 0 (who has negative Shapley value)
        bytes32[] memory claimIds = new bytes32[](1);
        claimIds[0] = cId;

        vm.prank(trainers[0]);
        vm.expectRevert("Non-positive Shapley value");
        privateShapley.claimRewards(
            testRound,
            claimIds,
            keccak256(abi.encodePacked("salt", testRound, uint256(0)))
        );
    }

    function testGasCommit() public {
        setupBasicScenarioWithMapping();

        uint256 gasStart = gasleft();
        privateShapley.commitCoalitions(roundId, coalitionIds, commitments);
        uint256 gasUsed = gasStart - gasleft();

        console.log(
            "Gas used for committing %d coalitions: %d",
            coalitionIds.length,
            gasUsed
        );
    }

    function testGasReveal() public {
        setupBasicScenarioWithMapping();
        privateShapley.commitCoalitions(roundId, coalitionIds, commitments);

        uint256[] memory scores = new uint256[](coalitionIds.length);
        for (uint256 i = 0; i < scores.length; i++) {
            scores[i] = 100;
        }

        vm.prank(testers[0]);
        privateShapley.publishResults(roundId, coalitionIds, scores);

        uint256 gasStart = gasleft();
        privateShapley.revealCoalitions(
            roundId,
            coalitionIds,
            bitfields,
            nonces
        );
        uint256 gasUsed = gasStart - gasleft();

        console.log(
            "Gas used for revealing %d coalitions: %d",
            coalitionIds.length,
            gasUsed
        );
    }

    function testGasClaim() public {
        completeLifecycleUpToReveal();

        bytes32[] memory claimIds = new bytes32[](1);
        claimIds[0] = coalitionIds[0];

        vm.prank(trainers[0]);
        uint256 gasStart = gasleft();
        privateShapley.claimRewards(roundId, claimIds, trainerSalts[0]);
        uint256 gasUsed = gasStart - gasleft();

        console.log("Gas used for claiming 1 coalition: %d", gasUsed);
    }
}
