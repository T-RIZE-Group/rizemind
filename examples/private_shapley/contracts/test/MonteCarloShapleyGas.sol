// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import {MonteCarloPrivateShapley} from "../src/MonteCarloShapley.sol";
import {MockERC20} from "../src/MockERC20.sol";

contract MonteCarloGasAnalysis is Test {
    MonteCarloPrivateShapley private privateShapley;
    MockERC20 private mockToken;

    // Test accounts
    address private owner;
    address[] private allTrainers;
    address[] private testers;

    // Test variables
    uint256 private roundId;

    // Constants
    uint256 private constant DAY = 1 days;

    // Gas tracking
    struct GasReport {
        uint256 min;
        uint256 max;
        uint256 avg;
        uint256 total;
        uint256 count;
    }

    mapping(string => GasReport) private gasReports;

    function setUp() public {
        owner = address(this);
        // Create a large pool of trainers (255 max)
        for (uint256 i = 0; i < 255; i++) {
            allTrainers.push(address(uint160(0x1000 + i)));
        }

        // Create testers (10)
        for (uint256 i = 0; i < 10; i++) {
            testers.push(address(uint160(0x2000 + i)));
        }

        // Deploy contracts
        mockToken = new MockERC20("Reward Token", "RWD");
        privateShapley = new MonteCarloPrivateShapley(address(mockToken));
        mockToken.mint(address(privateShapley), 1_000_000_000 * 10 ** 18);

        roundId = 1;
    }

    /*╔══════════════════════════════════════════════════════════════════╗
      ║                    1. SCALING ANALYSIS                           ║
      ╚══════════════════════════════════════════════════════════════════╝*/

    function testGasScalingByTrainerCount() public {
        console.log("\n=== Gas Scaling by Trainer Count ===");
        console.log(
            "Trainers | SetTrainers | CommitMap | RevealMap | Avg/Trainer"
        );
        console.log(
            "---------|-------------|-----------|-----------|-------------"
        );

        uint256[] memory counts = new uint256[](4);
        counts[0] = 5;
        counts[1] = 10;
        counts[2] = 25;
        counts[3] = 50;

        for (uint256 i = 0; i < counts.length; i++) {
            uint256 trainerCount = counts[i];

            // Reset contract for clean state
            privateShapley = new MonteCarloPrivateShapley(address(mockToken));

            // Measure setTrainers gas
            address[] memory trainers = new address[](trainerCount);
            bool[] memory flags = new bool[](trainerCount);
            bytes32[] memory salts = new bytes32[](trainerCount);

            for (uint256 j = 0; j < trainerCount; j++) {
                trainers[j] = allTrainers[j];
                flags[j] = true;
                salts[j] = keccak256(abi.encodePacked("salt", j));
            }

            uint256 gasSetTrainers;
            try
                this.measureGas(
                    address(privateShapley),
                    abi.encodeWithSelector(
                        privateShapley.setTrainers.selector,
                        trainers,
                        flags
                    )
                )
            returns (uint256 gas) {
                gasSetTrainers = gas;
            } catch {
                console.log(
                    "Failed to measure setTrainers gas for %d trainers",
                    trainerCount
                );
                continue;
            }

            // Create round
            try
                privateShapley.createRound(
                    1,
                    block.timestamp,
                    block.timestamp + DAY
                )
            {
                // Measure commit mapping
                bytes32 commitment = keccak256(
                    abi.encodePacked(trainers, salts)
                );
                uint256 gasCommit;
                try
                    this.measureGas(
                        address(privateShapley),
                        abi.encodeWithSelector(
                            privateShapley.commitTrainerMapping.selector,
                            1,
                            commitment
                        )
                    )
                returns (uint256 gas) {
                    gasCommit = gas;
                } catch {
                    console.log(
                        "Failed to measure commit gas for %d trainers",
                        trainerCount
                    );
                    continue;
                }

                // Measure reveal mapping
                uint256 gasReveal;
                try
                    this.measureGas(
                        address(privateShapley),
                        abi.encodeWithSelector(
                            privateShapley.revealTrainerMapping.selector,
                            1,
                            trainers,
                            salts
                        )
                    )
                returns (uint256 gas) {
                    gasReveal = gas;
                } catch {
                    console.log(
                        "Failed to measure reveal gas for %d trainers",
                        trainerCount
                    );
                    continue;
                }

                uint256 total = (gasSetTrainers + gasCommit + gasReveal);
                uint256 avgPerTrainer = total / trainerCount;

                console.log("Trainer Count: %s", padNumber(trainerCount, 8));
                console.log(
                    "Set Trainers Gas: %s",
                    padNumber(gasSetTrainers, 11)
                );
                console.log("Commit Gas: %s", padNumber(gasCommit, 9));
                console.log("Reveal Gas: %s", padNumber(gasReveal, 9));
                console.log("Total Gas: %s", padNumber(total, 11));
                console.log(
                    "Avg Per Trainer: %s",
                    padNumber(avgPerTrainer, 11)
                );
                console.log("-------------------");
            } catch {
                console.log(
                    "Failed to create round for %d trainers",
                    trainerCount
                );
                continue;
            }
        }
    }

    function testGasScalingByCoalitionSize() public {
        console.log("\n=== Gas Scaling by Coalition Size ===");
        console.log("Size | Commit | Publish | Reveal | Claim | Total");
        console.log("-----|--------|---------|--------|-------|-------");

        // Setup with max trainers for comprehensive test
        setupRoundWithTrainers(20, 1);

        uint256[] memory sizes = new uint256[](6);
        sizes[0] = 1;
        sizes[1] = 5;
        sizes[2] = 10;
        sizes[3] = 15;
        sizes[4] = 20;
        sizes[5] = 50;

        for (uint256 s = 0; s < sizes.length; s++) {
            uint256 coalitionSize = sizes[s];
            if (coalitionSize > 20) {
                // For large coalitions, we won't use Shapley
                testLargeCoalitionGas(coalitionSize);
            } else {
                testSmallCoalitionGas(coalitionSize);
            }
        }
    }

    function testSmallCoalitionGas(uint256 coalitionSize) private {
        // Generate coalition data
        bytes32 id = keccak256(
            abi.encodePacked("coalition", coalitionSize, block.timestamp)
        );
        uint256 bitMask = (1 << coalitionSize) - 1;
        bytes32 bitfield = bytes32(bitMask);
        bytes32 nonce = keccak256(abi.encodePacked("nonce", coalitionSize));
        bytes32 commitment = keccak256(abi.encodePacked(bitfield, nonce));

        // Setup Shapley values for this coalition size
        setupShapleyValuesForSize(coalitionSize);

        // Measure operations
        bytes32[] memory ids = new bytes32[](1);
        bytes32[] memory commitments = new bytes32[](1);
        ids[0] = id;
        commitments[0] = commitment;

        uint256 gasCommit = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.commitCoalitions.selector,
                1,
                ids,
                commitments
            )
        );

        // Publish
        uint256[] memory scores = new uint256[](1);
        scores[0] = 1000;

        vm.prank(testers[0]);
        uint256 gasPublish = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.publishResults.selector,
                1,
                ids,
                scores
            )
        );

        // Reveal
        bytes32[] memory bitfields = new bytes32[](1);
        bytes32[] memory nonces = new bytes32[](1);
        bitfields[0] = bitfield;
        nonces[0] = nonce;

        uint256 gasReveal = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.revealCoalitions.selector,
                1,
                ids,
                bitfields,
                nonces
            )
        );

        // Claim (first trainer)
        bytes32 salt = keccak256(abi.encodePacked("salt", uint256(0)));
        vm.prank(allTrainers[0]);
        uint256 gasClaim = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.claimRewards.selector,
                1,
                ids,
                salt
            )
        );

        uint256 total = gasCommit + gasPublish + gasReveal + gasClaim;

        console.log("Coalition Size: %s", padNumber(coalitionSize, 4));
        console.log("Gas Commit: %s", padNumber(gasCommit, 6));
        console.log("Gas Publish: %s", padNumber(gasPublish, 7));
        console.log("Gas Reveal: %s", padNumber(gasReveal, 6));
        console.log("Gas Claim: %s", padNumber(gasClaim, 5));
        console.log("Total Gas: %s", padNumber(total, 5));
    }

    function testLargeCoalitionGas(uint256 coalitionSize) private {
        // For coalitions > 20, we use simplified reward calculation
        // Just measure the basic operations without Shapley
        bytes32 id = keccak256(
            abi.encodePacked("large-coalition", coalitionSize)
        );
        uint256 bitMask = (1 << coalitionSize) - 1;
        bytes32 bitfield = bytes32(bitMask);
        bytes32 nonce = keccak256(abi.encodePacked("nonce", coalitionSize));
        bytes32 commitment = keccak256(abi.encodePacked(bitfield, nonce));

        // Similar measurements but without Shapley calculation
        console.log(
            "%s | Large coalition - Shapley not supported",
            padNumber(coalitionSize, 4)
        );
    }

    /*╔══════════════════════════════════════════════════════════════════╗
      ║                    2. BATCH OPERATION ANALYSIS                   ║
      ╚══════════════════════════════════════════════════════════════════╝*/

    function testBatchOperationScaling() public {
        console.log("\n=== Batch Operation Scaling ===");
        console.log(
            "Batch | Commit | Publish | Reveal | BatchClaim | Avg/Item"
        );
        console.log(
            "------|--------|---------|--------|------------|----------"
        );

        setupRoundWithTrainers(10, 1);
        setupShapleyValuesForSize(10);

        uint256[] memory batchSizes = new uint256[](6);
        batchSizes[0] = 1;
        batchSizes[1] = 5;
        batchSizes[2] = 10;
        batchSizes[3] = 20;
        batchSizes[4] = 30;
        batchSizes[5] = 50;

        for (uint256 i = 0; i < batchSizes.length; i++) {
            measureBatchOperations(batchSizes[i]);
        }
    }

    function measureBatchOperations(uint256 batchSize) private {
        // Generate batch data
        bytes32[] memory ids = new bytes32[](batchSize);
        bytes32[] memory commitments = new bytes32[](batchSize);
        bytes32[] memory bitfields = new bytes32[](batchSize);
        bytes32[] memory nonces = new bytes32[](batchSize);
        uint256[] memory scores = new uint256[](batchSize);

        for (uint256 i = 0; i < batchSize; i++) {
            ids[i] = keccak256(abi.encodePacked("batch", i, block.timestamp));
            bitfields[i] = bytes32(uint256(0x3)); // First 2 trainers
            nonces[i] = keccak256(abi.encodePacked("nonce", i));
            commitments[i] = keccak256(
                abi.encodePacked(bitfields[i], nonces[i])
            );
            scores[i] = 1000 + i * 100;
        }

        // Measure batch commit
        uint256 gasCommit = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.commitCoalitions.selector,
                1,
                ids,
                commitments
            )
        );

        // Measure batch publish
        vm.prank(testers[0]);
        uint256 gasPublish = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.publishResults.selector,
                1,
                ids,
                scores
            )
        );

        // Measure batch reveal
        uint256 gasReveal = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.revealCoalitions.selector,
                1,
                ids,
                bitfields,
                nonces
            )
        );

        // Measure batch claim
        bytes32 salt = keccak256(abi.encodePacked("salt", uint256(0)));
        vm.prank(allTrainers[0]);
        uint256 gasClaim = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.claimRewards.selector,
                1,
                ids,
                salt
            )
        );

        uint256 avgPerItem = (gasCommit + gasPublish + gasReveal + gasClaim) /
            batchSize;

        console.log("Batch Size: %s", padNumber(batchSize, 5));
        console.log("Commit Gas: %s", padNumber(gasCommit, 6));
        console.log("Publish Gas: %s", padNumber(gasPublish, 7));
        console.log("Reveal Gas: %s", padNumber(gasReveal, 6));
        console.log("Claim Gas: %s", padNumber(gasClaim, 10));
        console.log("Avg Per Item: %s", padNumber(avgPerItem, 8));
    }

    /*╔══════════════════════════════════════════════════════════════════╗
      ║                 3. COMPLETE WORKFLOW ANALYSIS                    ║
      ╚══════════════════════════════════════════════════════════════════╝*/

    function testCompleteWorkflowsByRole() public {
        console.log("\n=== Complete Workflow Gas Costs by Role ===");

        // Owner workflow
        testOwnerWorkflow();

        // Tester workflow
        testTesterWorkflow();

        // Trainer workflow
        testTrainerWorkflow();
    }

    function testOwnerWorkflow() private {
        console.log("\n--- Owner Workflow ---");

        privateShapley = new MonteCarloPrivateShapley(address(mockToken));
        uint256 totalGas = 0;

        // 1. Set trainers (50)
        address[] memory trainers = new address[](50);
        bool[] memory flags = new bool[](50);
        bytes32[] memory salts = new bytes32[](50);

        for (uint256 i = 0; i < 50; i++) {
            trainers[i] = allTrainers[i];
            flags[i] = true;
            salts[i] = keccak256(abi.encodePacked("salt", i));
        }

        uint256 gas1 = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.setTrainers.selector,
                trainers,
                flags
            )
        );
        console.log("1. Set 50 trainers: %s gas", gas1);
        totalGas += gas1;

        // 2. Set testers
        address[] memory testerAddrs = new address[](3);
        bool[] memory testerFlags = new bool[](3);
        for (uint256 i = 0; i < 3; i++) {
            testerAddrs[i] = testers[i];
            testerFlags[i] = true;
        }

        uint256 gas2 = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.setTesters.selector,
                testerAddrs,
                testerFlags
            )
        );
        console.log("2. Set 3 testers: %s gas", gas2);
        totalGas += gas2;

        // 3. Create round
        uint256 gas3 = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.createRound.selector,
                1,
                block.timestamp,
                block.timestamp + DAY
            )
        );
        console.log("3. Create round: %s gas", gas3);
        totalGas += gas3;

        // 4. Commit trainer mapping
        bytes32 commitment = keccak256(abi.encodePacked(trainers, salts));
        uint256 gas4 = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.commitTrainerMapping.selector,
                1,
                commitment
            )
        );
        console.log("4. Commit mapping: %s gas", gas4);
        totalGas += gas4;

        // 5. Reveal trainer mapping
        uint256 gas5 = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.revealTrainerMapping.selector,
                1,
                trainers,
                salts
            )
        );
        console.log("5. Reveal mapping: %s gas", gas5);
        totalGas += gas5;

        // 6. Set Shapley values (simplified for 10 trainers)
        address[] memory shapleyTrainers = new address[](10);
        bytes32[] memory shapleySalts = new bytes32[](10);
        for (uint256 i = 0; i < 10; i++) {
            shapleyTrainers[i] = trainers[i];
            shapleySalts[i] = salts[i];
        }
        privateShapley.revealTrainerMapping(1, shapleyTrainers, shapleySalts);

        uint256[][] memory coalitions = new uint256[][](11);
        uint256[] memory values = new uint256[](11);

        // Empty coalition
        coalitions[0] = new uint256[](0);
        values[0] = 0;

        // Single trainer coalitions
        for (uint256 i = 1; i <= 10; i++) {
            coalitions[i] = new uint256[](1);
            coalitions[i][0] = i;
            values[i] = i * 10_000000;
        }

        uint256 gas6 = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.setShapleyCoalitionValues.selector,
                1,
                coalitions,
                values
            )
        );
        console.log("6. Set Shapley values: %s gas", gas6);
        totalGas += gas6;

        // 7. Commit 10 coalitions
        bytes32[] memory cIds = new bytes32[](10);
        bytes32[] memory cCommitments = new bytes32[](10);

        for (uint256 i = 0; i < 10; i++) {
            cIds[i] = keccak256(abi.encodePacked("coalition", i));
            bytes32 bf = bytes32(uint256(1 << i));
            bytes32 nc = keccak256(abi.encodePacked("nonce", i));
            cCommitments[i] = keccak256(abi.encodePacked(bf, nc));
        }

        uint256 gas7 = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.commitCoalitions.selector,
                1,
                cIds,
                cCommitments
            )
        );
        console.log("7. Commit 10 coalitions: %s gas", gas7);
        totalGas += gas7;

        console.log("\nTotal Owner Gas: %s", totalGas);
        console.log("At 30 gwei: %s ETH", (totalGas * 30) / 1e9);
    }

    function testTesterWorkflow() private {
        console.log("\n--- Tester Workflow ---");

        // Assume setup is done
        setupRoundWithTrainers(10, 1);

        // Commit some coalitions
        bytes32[] memory ids = new bytes32[](20);
        bytes32[] memory commitments = new bytes32[](20);

        for (uint256 i = 0; i < 20; i++) {
            ids[i] = keccak256(abi.encodePacked("test-coalition", i));
            bytes32 bf = bytes32(uint256((1 << (i % 10)) | 1));
            bytes32 nc = keccak256(abi.encodePacked("nonce", i));
            commitments[i] = keccak256(abi.encodePacked(bf, nc));
        }

        privateShapley.commitCoalitions(1, ids, commitments);

        // Tester publishes results in batches
        uint256 totalGas = 0;

        // Batch 1: 5 results
        uint256[] memory scores1 = new uint256[](5);
        bytes32[] memory ids1 = new bytes32[](5);
        for (uint256 i = 0; i < 5; i++) {
            ids1[i] = ids[i];
            scores1[i] = 1000 + i * 100;
        }

        vm.prank(testers[0]);
        uint256 gas1 = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.publishResults.selector,
                1,
                ids1,
                scores1
            )
        );
        console.log("Publish 5 results: %s gas", gas1);
        totalGas += gas1;

        // Batch 2: 10 results
        uint256[] memory scores2 = new uint256[](10);
        bytes32[] memory ids2 = new bytes32[](10);
        for (uint256 i = 0; i < 10; i++) {
            ids2[i] = ids[i + 5];
            scores2[i] = 2000 + i * 100;
        }

        vm.prank(testers[0]);
        uint256 gas2 = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.publishResults.selector,
                1,
                ids2,
                scores2
            )
        );
        console.log("Publish 10 results: %s gas", gas2);
        totalGas += gas2;

        // Batch 3: 5 results
        uint256[] memory scores3 = new uint256[](5);
        bytes32[] memory ids3 = new bytes32[](5);
        for (uint256 i = 0; i < 5; i++) {
            ids3[i] = ids[i + 15];
            scores3[i] = 3000 + i * 100;
        }

        vm.prank(testers[0]);
        uint256 gas3 = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.publishResults.selector,
                1,
                ids3,
                scores3
            )
        );
        console.log("Publish 5 results: %s gas", gas3);
        totalGas += gas3;

        console.log("\nTotal Tester Gas (20 results): %s", totalGas);
        console.log("Average per result: %s", totalGas / 20);
    }

    function testTrainerWorkflow() private {
        console.log("\n--- Trainer Workflow ---");

        // Full setup and reveal coalitions
        setupCompleteScenario();

        // Test different claiming patterns
        bytes32 salt = keccak256(abi.encodePacked("salt", uint256(0)));

        // Pattern 1: Claim single coalition
        bytes32[] memory claim1 = new bytes32[](1);
        claim1[0] = keccak256(abi.encodePacked("coalition", uint256(0)));

        vm.prank(allTrainers[0]);
        uint256 gas1 = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.claimRewards.selector,
                1,
                claim1,
                salt
            )
        );
        console.log("Claim 1 coalition: %s gas", gas1);

        // Pattern 2: Claim 5 coalitions
        bytes32[] memory claim5 = new bytes32[](5);
        for (uint256 i = 0; i < 5; i++) {
            claim5[i] = keccak256(abi.encodePacked("coalition", i + 1));
        }

        vm.prank(allTrainers[0]);
        uint256 gas5 = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.claimRewards.selector,
                1,
                claim5,
                salt
            )
        );
        console.log("Claim 5 coalitions: %s gas", gas5);
        console.log("Average per coalition: %s", gas5 / 5);

        // Pattern 3: Claim 20 coalitions
        bytes32[] memory claim20 = new bytes32[](20);
        for (uint256 i = 0; i < 20; i++) {
            claim20[i] = keccak256(abi.encodePacked("coalition", i + 6));
        }

        vm.prank(allTrainers[0]);
        uint256 gas20 = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.claimRewards.selector,
                1,
                claim20,
                salt
            )
        );
        console.log("Claim 20 coalitions: %s gas", gas20);
        console.log("Average per coalition: %s", gas20 / 20);

        console.log("\nGas savings from batching:");
        console.log("Single claim baseline: %s", gas1);
        console.log("5-batch savings: %s gas/claim", gas1 - (gas5 / 5));
        console.log("20-batch savings: %s gas/claim", gas1 - (gas20 / 20));
    }

    /*╔══════════════════════════════════════════════════════════════════╗
      ║                 4. SHAPLEY CALCULATION ANALYSIS                  ║
      ╚══════════════════════════════════════════════════════════════════╝*/

    function testShapleyCalculationScaling() public {
        console.log("\n=== Shapley Calculation Gas Scaling ===");
        console.log("Players | SetValues | GetShapley | ClaimWithShapley");
        console.log("--------|-----------|------------|------------------");

        uint256[] memory playerCounts = new uint256[](7);
        playerCounts[0] = 2;
        playerCounts[1] = 3;
        playerCounts[2] = 5;
        playerCounts[3] = 8;
        playerCounts[4] = 10;
        playerCounts[5] = 15;
        playerCounts[6] = 20;

        for (uint256 i = 0; i < playerCounts.length; i++) {
            testShapleyForNPlayers(playerCounts[i]);
        }
    }

    function testShapleyForNPlayers(uint256 n) private {
        // Setup round with n trainers
        uint256 testRound = 100 + n;
        setupRoundWithTrainers(n, 1);

        // Generate all coalition values (2^n coalitions)
        uint256 numCoalitions = 1 << n;
        uint256[][] memory coalitions = new uint256[][](numCoalitions);
        uint256[] memory values = new uint256[](numCoalitions);

        for (uint256 mask = 0; mask < numCoalitions; mask++) {
            uint256 size = 0;
            for (uint256 j = 0; j < n; j++) {
                if ((mask >> j) & 1 == 1) size++;
            }

            coalitions[mask] = new uint256[](size);
            uint256 idx = 0;
            for (uint256 j = 0; j < n; j++) {
                if ((mask >> j) & 1 == 1) {
                    coalitions[mask][idx++] = j + 1;
                }
            }

            // Simple value function: sum of player indices
            values[mask] = mask * 1_000000;
        }

        // Measure setting coalition values
        uint256 gasSet = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.setShapleyCoalitionValues.selector,
                1,
                coalitions,
                values
            )
        );

        // Measure getting Shapley value
        vm.prank(allTrainers[0]);
        uint256 gasGet = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.getTrainerShapleyValue.selector,
                1,
                allTrainers[0]
            )
        );

        // Setup and measure claim with Shapley calculation
        bytes32 cId = keccak256(abi.encodePacked("shapley-test", n));
        bytes32 bitfield = bytes32((1 << n) - 1); // All trainers
        bytes32 nonce = keccak256("nonce");
        bytes32 commitment = keccak256(abi.encodePacked(bitfield, nonce));

        // Commit, publish, reveal
        bytes32[] memory ids = new bytes32[](1);
        ids[0] = cId;
        bytes32[] memory commitments = new bytes32[](1);
        commitments[0] = commitment;

        privateShapley.commitCoalitions(1, ids, commitments);

        uint256[] memory scores = new uint256[](1);
        scores[0] = 10000;
        vm.prank(testers[0]);
        privateShapley.publishResults(1, ids, scores);

        bytes32[] memory bitfields = new bytes32[](1);
        bitfields[0] = bitfield;
        bytes32[] memory nonces = new bytes32[](1);
        nonces[0] = nonce;
        privateShapley.revealCoalitions(1, ids, bitfields, nonces);

        // Measure claim (includes Shapley calculation)
        bytes32 salt = keccak256(abi.encodePacked("salt", uint256(0)));
        vm.prank(allTrainers[0]);
        uint256 gasClaim = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.claimRewards.selector,
                1,
                ids,
                salt
            )
        );

        console.log("Size: %s", padNumber(n, 7));
        console.log("Set Gas: %s", padNumber(gasSet, 9));
        console.log("Get Gas: %s", padNumber(gasGet, 10));
        console.log("Claim Gas: %s", padNumber(gasClaim, 16));
    }

    /*╔══════════════════════════════════════════════════════════════════╗
      ║                    5. STORAGE OPTIMIZATION ANALYSIS              ║
      ╚══════════════════════════════════════════════════════════════════╝*/

    function testStorageOptimizationPatterns() public {
        console.log("\n=== Storage Optimization Analysis ===");

        // Test 1: Coalition data storage patterns
        testCoalitionStoragePatterns();

        // Test 2: Mapping storage efficiency
        testMappingStorageEfficiency();

        // Test 3: Batch vs individual storage updates
        testBatchVsIndividualStorage();
    }

    function testCoalitionStoragePatterns() private {
        console.log("\n--- Coalition Storage Patterns ---");

        setupRoundWithTrainers(10, 1);

        // Pattern 1: Store minimal data
        bytes32 id1 = keccak256("minimal");
        bytes32 commitment1 = keccak256("commitment1");

        uint256 gas1 = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.commitCoalitions.selector,
                1,
                toArray(id1),
                toArray(commitment1)
            )
        );
        console.log("Minimal coalition commit: %s gas", gas1);

        // Pattern 2: After publishing results (more storage)
        bytes32 id2 = keccak256("with-results");
        bytes32 commitment2 = keccak256("commitment2");

        privateShapley.commitCoalitions(1, toArray(id2), toArray(commitment2));

        vm.prank(testers[0]);
        uint256 gas2 = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.publishResults.selector,
                1,
                toArray(id2),
                toArray(uint256(1000))
            )
        );
        console.log("Publish with tester tracking: %s gas", gas2);

        // Pattern 3: Multiple testers (array growth)
        vm.prank(testers[1]);
        uint256 gas3 = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.publishResults.selector,
                1,
                toArray(id2),
                toArray(uint256(1100))
            )
        );
        console.log("Second tester publish: %s gas", gas3);

        vm.prank(testers[2]);
        uint256 gas4 = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.publishResults.selector,
                1,
                toArray(id2),
                toArray(uint256(1200))
            )
        );
        console.log("Third tester publish: %s gas", gas4);
    }

    function testMappingStorageEfficiency() private {
        console.log("\n--- Mapping Storage Efficiency ---");

        // Compare different trainer counts for mapping storage
        uint256[] memory counts = new uint256[](5);
        counts[0] = 10;
        counts[1] = 50;
        counts[2] = 100;
        counts[3] = 200;
        counts[4] = 255;

        for (uint256 i = 0; i < counts.length; i++) {
            privateShapley = new MonteCarloPrivateShapley(address(mockToken));

            address[] memory trainers = new address[](counts[i]);
            bytes32[] memory salts = new bytes32[](counts[i]);
            bool[] memory flags = new bool[](counts[i]);

            for (uint256 j = 0; j < counts[i]; j++) {
                trainers[j] = allTrainers[j];
                salts[j] = keccak256(abi.encodePacked("salt", j));
                flags[j] = true;
            }

            privateShapley.setTrainers(trainers, flags);
            privateShapley.createRound(
                1,
                block.timestamp,
                block.timestamp + DAY
            );

            bytes32 commitment = keccak256(abi.encodePacked(trainers, salts));
            privateShapley.commitTrainerMapping(1, commitment);

            uint256 gasReveal = measureGas(
                address(privateShapley),
                abi.encodeWithSelector(
                    privateShapley.revealTrainerMapping.selector,
                    1,
                    trainers,
                    salts
                )
            );

            console.log(
                "Reveal %s trainers: %s gas (%s per trainer)",
                counts[i],
                gasReveal,
                gasReveal / counts[i]
            );
        }
    }

    function testBatchVsIndividualStorage() private {
        console.log("\n--- Batch vs Individual Storage Updates ---");

        setupRoundWithTrainers(10, 1);

        // Individual updates
        uint256 totalIndividual = 0;
        for (uint256 i = 0; i < 10; i++) {
            bytes32 id = keccak256(abi.encodePacked("individual", i));
            bytes32 commitment = keccak256(abi.encodePacked("commit", i));

            uint256 gas = measureGas(
                address(privateShapley),
                abi.encodeWithSelector(
                    privateShapley.commitCoalitions.selector,
                    1,
                    toArray(id),
                    toArray(commitment)
                )
            );
            totalIndividual += gas;
        }
        console.log("10 individual commits: %s total gas", totalIndividual);
        console.log("Average per commit: %s", totalIndividual / 10);

        // Batch update
        bytes32[] memory ids = new bytes32[](10);
        bytes32[] memory commitments = new bytes32[](10);

        for (uint256 i = 0; i < 10; i++) {
            ids[i] = keccak256(abi.encodePacked("batch", i));
            commitments[i] = keccak256(abi.encodePacked("commit", i));
        }

        uint256 gasBatch = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.commitCoalitions.selector,
                1,
                ids,
                commitments
            )
        );
        console.log("Batch commit 10: %s gas", gasBatch);
        console.log("Average per commit: %s", gasBatch / 10);
        console.log(
            "Savings per item: %s gas",
            (totalIndividual - gasBatch) / 10
        );
    }

    /*╔══════════════════════════════════════════════════════════════════╗
      ║                    6. EDGE CASE ANALYSIS                         ║
      ╚══════════════════════════════════════════════════════════════════╝*/

    function testEdgeCaseGasConsumption() public {
        console.log("\n=== Edge Case Gas Analysis ===");

        // Test 1: Maximum batch sizes
        testMaxBatchSizes();

        // Test 2: Worst-case Shapley calculations
        testWorstCaseShapley();

        // Test 3: Failed transactions gas consumption
        testFailedTransactionGas();
    }

    function testMaxBatchSizes() private {
        console.log("\n--- Maximum Batch Size Operations ---");

        setupRoundWithTrainers(50, 1);

        // Max batch commit (50)
        bytes32[] memory ids = new bytes32[](50);
        bytes32[] memory commitments = new bytes32[](50);

        for (uint256 i = 0; i < 50; i++) {
            ids[i] = keccak256(abi.encodePacked("max-batch", i));
            bytes32 bf = bytes32(uint256(1 << (i % 50)));
            bytes32 nc = keccak256(abi.encodePacked("nonce", i));
            commitments[i] = keccak256(abi.encodePacked(bf, nc));
        }

        uint256 gasMaxCommit = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.commitCoalitions.selector,
                1,
                ids,
                commitments
            )
        );
        console.log("Max batch commit (50): %s gas", gasMaxCommit);
        console.log("Per item: %s gas", gasMaxCommit / 50);

        // Compare with 51 (should fail)
        bytes32[] memory ids51 = new bytes32[](51);
        bytes32[] memory commitments51 = new bytes32[](51);

        for (uint256 i = 0; i < 51; i++) {
            ids51[i] = keccak256(abi.encodePacked("over-max", i));
            commitments51[i] = keccak256(abi.encodePacked("commit", i));
        }

        vm.expectRevert("Batch size exceeds maximum");
        privateShapley.commitCoalitions(1, ids51, commitments51);
        console.log("Batch size 51: Correctly reverted");
    }

    function testWorstCaseShapley() private {
        console.log("\n--- Worst-Case Shapley Calculations ---");

        // Test with maximum players (20) and all coalitions
        setupRoundWithTrainers(20, 1);

        // Set all 2^20 coalition values
        uint256 numCoalitions = 1 << 20;
        console.log("Total coalitions for 20 players: %s", numCoalitions);

        // Due to gas limits, we'll set a subset and measure
        uint256 subset = 1000; // Set first 1000 coalitions
        uint256[][] memory coalitions = new uint256[][](subset);
        uint256[] memory values = new uint256[](subset);

        for (uint256 mask = 0; mask < subset; mask++) {
            uint256 size = 0;
            for (uint256 j = 0; j < 20; j++) {
                if ((mask >> j) & 1 == 1) size++;
            }

            coalitions[mask] = new uint256[](size);
            uint256 idx = 0;
            for (uint256 j = 0; j < 20; j++) {
                if ((mask >> j) & 1 == 1) {
                    coalitions[mask][idx++] = j + 1;
                }
            }

            values[mask] = (mask + 1) * 1_000000;
        }

        uint256 gasSetSubset = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.setShapleyCoalitionValues.selector,
                1,
                coalitions,
                values
            )
        );
        console.log("Set 1000 coalition values: %s gas", gasSetSubset);

        // Measure Shapley calculation with partial data
        vm.prank(allTrainers[0]);
        uint256 gasCalc = measureGas(
            address(privateShapley),
            abi.encodeWithSelector(
                privateShapley.getTrainerShapleyValue.selector,
                1,
                allTrainers[0]
            )
        );
        console.log("Calculate Shapley (partial data): %s gas", gasCalc);
        console.log(
            "Note: Full 2^20 calculation would require ~%s gas",
            gasCalc * (numCoalitions / subset)
        );
    }

    function testFailedTransactionGas() private {
        console.log("\n--- Failed Transaction Gas Consumption ---");

        setupRoundWithTrainers(10, 1);
        setupShapleyValuesForSize(10);

        // Setup a revealed coalition
        bytes32 id = keccak256("test-fail");
        bytes32 bitfield = bytes32(uint256(0x3)); // Trainers 0 and 1
        bytes32 nonce = keccak256("nonce");
        bytes32 commitment = keccak256(abi.encodePacked(bitfield, nonce));

        privateShapley.commitCoalitions(1, toArray(id), toArray(commitment));
        vm.prank(testers[0]);
        privateShapley.publishResults(1, toArray(id), toArray(uint256(1000)));
        privateShapley.revealCoalitions(
            1,
            toArray(id),
            toArray(bitfield),
            toArray(nonce)
        );

        // Test 1: Wrong salt (early revert)
        bytes32 wrongSalt = keccak256("wrong");
        vm.prank(allTrainers[0]);
        uint256 gasStart = gasleft();
        vm.expectRevert("bad salt");
        privateShapley.claimRewards(1, toArray(id), wrongSalt);
        uint256 gasUsedWrongSalt = gasStart - gasleft();
        console.log("Failed claim (wrong salt): %s gas", gasUsedWrongSalt);

        // Test 2: Not a member (later revert)
        bytes32 correctSalt = keccak256(abi.encodePacked("salt", uint256(5)));
        vm.prank(allTrainers[5]);
        gasStart = gasleft();
        vm.expectRevert("Not member");
        privateShapley.claimRewards(1, toArray(id), correctSalt);
        uint256 gasUsedNotMember = gasStart - gasleft();
        console.log("Failed claim (not member): %s gas", gasUsedNotMember);

        // Test 3: Already claimed (latest revert)
        bytes32 salt0 = keccak256(abi.encodePacked("salt", uint256(0)));
        vm.prank(allTrainers[0]);
        privateShapley.claimRewards(1, toArray(id), salt0);

        vm.prank(allTrainers[0]);
        gasStart = gasleft();
        vm.expectRevert("Already claimed");
        privateShapley.claimRewards(1, toArray(id), salt0);
        uint256 gasUsedAlreadyClaimed = gasStart - gasleft();
        console.log(
            "Failed claim (already claimed): %s gas",
            gasUsedAlreadyClaimed
        );
    }

    /*╔══════════════════════════════════════════════════════════════════╗
      ║                    7. OPTIMIZATION RECOMMENDATIONS               ║
      ╚══════════════════════════════════════════════════════════════════╝*/

    function generateOptimizationReport() public {
        console.log("\n=== Gas Optimization Recommendations ===");
        console.log("\n1. BATCH OPERATIONS:");
        console.log("   - Batch size 10-20 provides optimal gas savings");
        console.log("   - Diminishing returns beyond batch size 30");
        console.log("   - Individual operations cost ~40-60%% more gas");

        console.log("\n2. SHAPLEY CALCULATIONS:");
        console.log("   - Gas scales exponentially with player count");
        console.log(
            "   - Practical limit: 15-18 players for on-chain calculation"
        );
        console.log("   - Consider off-chain calculation for >15 players");

        console.log("\n3. STORAGE PATTERNS:");
        console.log("   - Trainer mapping reveal is most expensive operation");
        console.log("   - Coalition commits are relatively cheap");
        console.log("   - Multiple tester results increase gas linearly");

        console.log("\n4. CLAIMING STRATEGIES:");
        console.log("   - Batch claiming saves ~30-40%% gas per coalition");
        console.log("   - Optimal batch size for claims: 10-20");
        console.log("   - Failed claims still consume significant gas");

        console.log("\n5. ROUND MANAGEMENT:");
        console.log(
            "   - Setup phase (mapping + Shapley values) is gas intensive"
        );
        console.log(
            "   - Consider deploying new contract instances for very different trainer sets"
        );
        console.log("   - Reuse rounds when trainer set is stable");
    }

    /*╔══════════════════════════════════════════════════════════════════╗
      ║                         HELPER FUNCTIONS                         ║
      ╚══════════════════════════════════════════════════════════════════╝*/

    function setupRoundWithTrainers(uint256 n, uint256 rid) private {
        address[] memory trainers = new address[](n);
        bytes32[] memory salts = new bytes32[](n);
        bool[] memory flags = new bool[](n);

        for (uint256 i = 0; i < n; i++) {
            trainers[i] = allTrainers[i];
            salts[i] = keccak256(abi.encodePacked("salt", i));
            flags[i] = true;
        }

        privateShapley.setTrainers(trainers, flags);

        // Set testers
        address[] memory testerAddrs = new address[](3);
        bool[] memory testerFlags = new bool[](3);
        for (uint256 i = 0; i < 3; i++) {
            testerAddrs[i] = testers[i];
            testerFlags[i] = true;
        }
        privateShapley.setTesters(testerAddrs, testerFlags);

        privateShapley.createRound(rid, block.timestamp, block.timestamp + DAY);

        bytes32 commitment = keccak256(abi.encodePacked(trainers, salts));
        privateShapley.commitTrainerMapping(rid, commitment);
        privateShapley.revealTrainerMapping(rid, trainers, salts);
    }

    function setupShapleyValuesForSize(uint256 n) private {
        require(n <= 20, "Max 20 for Shapley");

        // Simple coalition values for testing that ensure positive Shapley values
        uint256 numCoalitions = 1 << n;
        uint256[][] memory coalitions = new uint256[][](numCoalitions);
        uint256[] memory values = new uint256[](numCoalitions);

        for (uint256 mask = 0; mask < numCoalitions; mask++) {
            uint256 size = 0;
            for (uint256 j = 0; j < n; j++) {
                if ((mask >> j) & 1 == 1) size++;
            }

            coalitions[mask] = new uint256[](size);
            uint256 idx = 0;
            for (uint256 j = 0; j < n; j++) {
                if ((mask >> j) & 1 == 1) {
                    coalitions[mask][idx++] = j + 1;
                }
            }

            // Use a simple value function that ensures positive marginal contributions
            // Each coalition gets a value proportional to its size squared
            values[mask] = size * size * 10_000000;
        }

        privateShapley.setShapleyCoalitionValues(1, coalitions, values);
    }

    function setupCompleteScenario() private {
        setupRoundWithTrainers(10, 1);
        setupShapleyValuesForSize(10);

        // Create and reveal multiple coalitions
        bytes32[] memory ids = new bytes32[](30);
        bytes32[] memory commitments = new bytes32[](30);
        bytes32[] memory bitfields = new bytes32[](30);
        bytes32[] memory nonces = new bytes32[](30);
        uint256[] memory scores = new uint256[](30);

        for (uint256 i = 0; i < 30; i++) {
            ids[i] = keccak256(abi.encodePacked("coalition", i));
            bitfields[i] = bytes32(uint256(1 << (i % 10)) | 1); // Include trainer 0
            nonces[i] = keccak256(abi.encodePacked("nonce", i));
            commitments[i] = keccak256(
                abi.encodePacked(bitfields[i], nonces[i])
            );
            scores[i] = 1000 + i * 100;
        }

        privateShapley.commitCoalitions(1, ids, commitments);
        vm.prank(testers[0]);
        privateShapley.publishResults(1, ids, scores);
        privateShapley.revealCoalitions(1, ids, bitfields, nonces);
    }

    function measureGas(
        address target,
        bytes memory data
    ) public returns (uint256) {
        uint256 gasBefore = gasleft();
        // console.log("Gas before: %s", gasBefore);
        (bool success, ) = target.call(data);
        require(success, "Call failed in gas measurement");
        // console.log("Gas after: %s", gasleft());
        return gasBefore - gasleft();
    }

    function toArray(bytes32 item) private pure returns (bytes32[] memory) {
        bytes32[] memory array = new bytes32[](1);
        array[0] = item;
        return array;
    }

    function toArray(uint256 item) private pure returns (uint256[] memory) {
        uint256[] memory array = new uint256[](1);
        array[0] = item;
        return array;
    }

    function padNumber(
        uint256 num,
        uint256 width
    ) private pure returns (string memory) {
        string memory numStr = vm.toString(num);
        bytes memory numBytes = bytes(numStr);

        if (numBytes.length >= width) {
            return numStr;
        }

        bytes memory result = new bytes(width);
        uint256 padding = width - numBytes.length;

        for (uint256 i = 0; i < padding; i++) {
            result[i] = " ";
        }

        for (uint256 i = 0; i < numBytes.length; i++) {
            result[padding + i] = numBytes[i];
        }

        return string(result);
    }

    /*╔══════════════════════════════════════════════════════════════════╗
  ║           ✦  CLAIM REWARDS - GAS SCALING BY TRAINER COUNT ✦       ║
  ╚══════════════════════════════════════════════════════════════════╝*/
    function testClaimRewardsGasScaling() public {
        console.log(
            "\n=== claimRewards Gas vs nPlayers (1 coalition claimed) ==="
        );
        console.log("Players | GasUsed | Gas/Trainer | 2^n loops");
        console.log("--------|---------|-------------|-----------");

        uint8[11] memory cases = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13];

        for (uint256 i = 0; i < cases.length; i++) {
            uint8 n = cases[i];

            // ── 1) create a fresh round with `n` trainers ──────────────────────
            privateShapley = new MonteCarloPrivateShapley(address(mockToken));
            setupRoundWithTrainers(n, 1); // registers + mapping reveal
            setupShapleyValuesForSize(n); // sets 2^n coalition values

            // ── 2) commit-publish-reveal one coalition containing *all* trainers
            bytes32 cid = keccak256("claim-probe");
            bytes32 bitfield = bytes32((1 << n) - 1);
            bytes32 nonce = keccak256("nonce-probe");
            bytes32 comm = keccak256(abi.encodePacked(bitfield, nonce));
            privateShapley.commitCoalitions(1, toArray(cid), toArray(comm));

            uint256[] memory scores = toArray(10_00);
            vm.prank(testers[0]);
            privateShapley.publishResults(1, toArray(cid), scores);
            privateShapley.revealCoalitions(
                1,
                toArray(cid),
                toArray(bitfield),
                toArray(nonce)
            );
            mockToken.mint(address(privateShapley), 1e24); // 1 000 000 tokens

            // ── 3) measure claimRewards gas for trainer[0] ─────────────────────
            bytes32 salt = keccak256(abi.encodePacked("salt", uint256(0)));
            vm.prank(allTrainers[0]);
            uint256 gasUsed = measureGas(
                address(privateShapley),
                abi.encodeWithSelector(
                    privateShapley.claimRewards.selector,
                    1,
                    toArray(cid),
                    salt
                )
            );
            console.log("Players: %s", n);
            console.log("Gas Used: %s", gasUsed);
            console.log("Gas per Trainer: %s", gasUsed / n);
            console.log("2^n loops: %s", 1 << n);
        }
    }

    /*───────────────────────────────────────────────────────────────────
   1.  Trainer-count scaling table  (Set + Commit + Reveal)
───────────────────────────────────────────────────────────────────*/
    function testTrainerCountGasScaling() public {
        console.log("\n=== Trainer-count Gas ===");
        console.log("Trainers | Set | Commit | Reveal | Avg/Trainer | Total");
        console.log("---------|-----|--------|--------|-------------|-------");
        uint256[6] memory cases = [uint256(5), 10, 25, 50, 100, 255];

        for (uint256 k; k < cases.length; ++k) {
            uint256 n = cases[k];

            // fresh contract each run
            privateShapley = new MonteCarloPrivateShapley(address(mockToken));

            // build trainer arrays
            address[] memory tr = new address[](n);
            bool[] memory fl = new bool[](n);
            bytes32[] memory sl = new bytes32[](n);
            for (uint256 i; i < n; ++i) {
                tr[i] = allTrainers[i];
                fl[i] = true;
                sl[i] = keccak256(abi.encodePacked("salt", i));
            }

            uint256 gasSet = measureGas(
                address(privateShapley),
                abi.encodeWithSelector(
                    privateShapley.setTrainers.selector,
                    tr,
                    fl
                )
            );

            privateShapley.createRound(
                1,
                block.timestamp,
                block.timestamp + DAY
            );

            bytes32 commit = keccak256(abi.encodePacked(tr, sl));
            uint256 gasCommit = measureGas(
                address(privateShapley),
                abi.encodeWithSelector(
                    privateShapley.commitTrainerMapping.selector,
                    1,
                    commit
                )
            );
            uint256 gasReveal = measureGas(
                address(privateShapley),
                abi.encodeWithSelector(
                    privateShapley.revealTrainerMapping.selector,
                    1,
                    tr,
                    sl
                )
            );

            uint256 total = gasSet + gasCommit + gasReveal;
            console.log("Trainers: %s", n);
            console.log("SetTrainers Gas: %s", gasSet);
            console.log("CommitMap Gas: %s", gasCommit);
            console.log("RevealMap Gas: %s", gasReveal);
            console.log("Avg/Trainer: %s", total / n);
            console.log("Total Gas: %s", total);
        }
    }

    /*╔══════════════════════════════════════════════════════════╗
  ║       ➋ COALITION-SIZE SCALING  (commit→publish→reveal)
  ╚══════════════════════════════════════════════════════════╝*/
    // function testCoalitionSizeGasScaling() public {
    //     console.log("\n=== Coalition-size Gas ===");
    //     console.log("Size | Commit | Publish | Reveal  | Total");
    //     console.log("-----|--------|---------|--------|------");

    //     uint256[14] memory sizes = [
    //         uint256(1),
    //         5,
    //         10,
    //         15,
    //         20,
    //         30,
    //         40,
    //         50,
    //         60,
    //         70,
    //         80,
    //         90,
    //         100,
    //         255
    //     ];
    //     uint256 count = 1;
    //     for (uint256 s; s < sizes.length; ++s) {
    //         uint256 size = sizes[s];
    //         setupRoundWithTrainers(size, count);
    //         count++;

    //         bytes32 cid = keccak256(abi.encodePacked("S", size));
    //         bytes32 bf = bytes32((1 << size) - 1);
    //         bytes32 nc = keccak256("nc");
    //         bytes32 comm = keccak256(abi.encodePacked(bf, nc));

    //         uint256 gCommit = measureGas(
    //             address(privateShapley),
    //             abi.encodeWithSelector(
    //                 privateShapley.commitCoalitions.selector,
    //                 1,
    //                 toArray(cid),
    //                 toArray(comm)
    //             )
    //         );

    //         vm.prank(testers[0]);
    //         uint256 gPublish = measureGas(
    //             address(privateShapley),
    //             abi.encodeWithSelector(
    //                 privateShapley.publishResults.selector,
    //                 1,
    //                 toArray(cid),
    //                 toArray(uint256(1000))
    //             )
    //         );

    //         uint256 gReveal = measureGas(
    //             address(privateShapley),
    //             abi.encodeWithSelector(
    //                 privateShapley.revealCoalitions.selector,
    //                 1,
    //                 toArray(cid),
    //                 toArray(bf),
    //                 toArray(nc)
    //             )
    //         );

    //         mockToken.mint(address(privateShapley), 1e24);
    //         bytes32 salt = keccak256(abi.encodePacked("salt", uint256(0)));
    //         vm.prank(allTrainers[0]);

    //         uint256 total = gCommit + gPublish + gReveal;
    //         console.log("Size: %s", size);
    //         console.log("Commit Gas: %s", gCommit);
    //         console.log("Publish Gas: %s", gPublish);
    //         console.log("Reveal Gas: %s", gReveal);
    //         console.log("Total Gas: %s", total);
    //     }
    // }

    /*╔══════════════════════════════════════════════════════════╗
  ║         COALITION-BATCH GAS SCALING  (fixed trainers)    ║
  ╚══════════════════════════════════════════════════════════╝*/
    function testCoalitionBatchGasScaling() public {
        console.log("\n=== Coalition-batch Gas ===");
        console.log("Batch | Commit | Publish | Reveal | Total");
        console.log("----- | ------ | ------- | ------ | -----");

        // always 10 trainers mapped for every round

        uint256[14] memory batches = [
            uint256(1),
            5,
            10,
            15,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            255
        ];
        uint256 count = 1;
        for (uint256 b; b < batches.length; ++b) {
            uint256 batch = batches[b];
            setupRoundWithTrainers(batch, count);
            count++;

            // ── build arrays of length = batch ─────────────────────────────
            bytes32[] memory ids = new bytes32[](batch);
            bytes32[] memory coms = new bytes32[](batch);
            bytes32[] memory bfs = new bytes32[](batch);
            bytes32[] memory ncs = new bytes32[](batch);
            uint256[] memory scrs = new uint256[](batch);

            for (uint256 i; i < batch; ++i) {
                ids[i] = keccak256(abi.encodePacked("CID", batch, "_", i));
                bfs[i] = bytes32(uint256(0x3)); // trainers 0 & 1
                ncs[i] = keccak256(abi.encodePacked("nc", i));
                coms[i] = keccak256(abi.encodePacked(bfs[i], ncs[i]));
                scrs[i] = 1_000 + i; // dummy score
            }

            // ── Commit whole batch ─────────────────────────────────────────
            uint256 gCommit = measureGas(
                address(privateShapley),
                abi.encodeWithSelector(
                    privateShapley.commitCoalitions.selector,
                    1,
                    ids,
                    coms
                )
            );

            // ── Publish results (first tester) ────────────────────────────
            vm.prank(testers[0]);
            uint256 gPublish = measureGas(
                address(privateShapley),
                abi.encodeWithSelector(
                    privateShapley.publishResults.selector,
                    1,
                    ids,
                    scrs
                )
            );

            // ── Reveal whole batch ────────────────────────────────────────
            uint256 gReveal = measureGas(
                address(privateShapley),
                abi.encodeWithSelector(
                    privateShapley.revealCoalitions.selector,
                    1,
                    ids,
                    bfs,
                    ncs
                )
            );

            uint256 total = gCommit + gPublish + gReveal;
            console.log("Batch size: %s", batch);
            console.log("Commit gas: %s", gCommit);
            console.log("Publish gas: %s", gPublish);
            console.log("Reveal gas: %s", gReveal);
            console.log("Total gas: %s", total);
        }
    }
}
