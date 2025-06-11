// // SPDX-License-Identifier: MIT
// pragma solidity ^0.8.20;

// import "lib/forge-std/src/Test.sol";
// import "lib/forge-std/src/console.sol";
// import "../src/ImprovedPrivateShapley.sol";
// import "../src/MockERC20.sol";

// contract PrivateShapleyGasTest is Test {
//     ImprovedPrivateShapley private privateShapley;
//     MockERC20 private mockToken;

//     // Test accounts
//     address private owner;
//     address[] private allTrainers;
//     address[] private testers;

//     // Test variables
//     uint256 private roundId;

//     // Constants
//     uint256 private constant DAY = 1 days;

//     function setUp() public {
//         // Setup accounts
//         owner = address(this);

//         // Create a large pool of trainers (255)
//         for (uint256 i = 0; i < 255; i++) {
//             allTrainers.push(address(uint160(0x1000 + i)));
//             vm.label(
//                 allTrainers[i],
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

//         // Default roundId
//         roundId = 1;

//         // Create round
//         uint256 currentTime = block.timestamp;
//         privateShapley.createRound(roundId, currentTime, currentTime + DAY);

//         // Register testers
//         bool[] memory flags = new bool[](testers.length);
//         for (uint256 i = 0; i < testers.length; i++) {
//             flags[i] = true;
//         }
//         privateShapley.setTesters(testers, flags);
//     }

//     // Helper to generate a coalition for specific trainers
//     function generateCoalition(
//         address[] memory trainers
//     )
//         private
//         returns (
//             bytes32 id,
//             bytes32 bitfield,
//             bytes32 nonce,
//             bytes32 commitment
//         )
//     {
//         // Generate random ID
//         id = keccak256(
//             abi.encodePacked("coalition", block.timestamp, block.prevrandao)
//         );

//         // Create bitfield based on trainer indices
//         uint256 bitMask = 0;
//         for (uint256 i = 0; i < trainers.length; i++) {
//             // Find trainer index (1-based)
//             uint8 trainerIdx = privateShapley.addressToIndex(trainers[i]);
//             if (trainerIdx > 0) {
//                 bitMask |= (1 << (trainerIdx - 1));
//             }
//         }

//         bitfield = bytes32(bitMask);

//         // Generate random nonce
//         nonce = keccak256(abi.encodePacked("nonce", id, block.prevrandao));

//         // Create commitment
//         commitment = keccak256(abi.encodePacked(bitfield, nonce));
//     }

//     // Test to measure gas scaling from 5 trainers to 255 trainers
//     function testGasScalingByTrainerCount() public {
//         // Define trainer count checkpoints
//         uint256[] memory checkpoints = new uint256[](9);
//         checkpoints[0] = 5; // 5 trainers
//         checkpoints[1] = 10; // 10 trainers
//         checkpoints[2] = 25; // 25 trainers
//         checkpoints[3] = 50; // 50 trainers
//         checkpoints[4] = 75; // 75 trainers
//         checkpoints[5] = 100; // 100 trainers
//         checkpoints[6] = 150; // 150 trainers
//         checkpoints[7] = 200; // 200 trainers
//         checkpoints[8] = 255; // 255 trainers (maximum)

//         // Print header
//         console.log("=== Gas Usage by Trainer Count ===");
//         console.log("Trainers | Register | Commit | Publish | Reveal | Claim");
//         console.log("--------|----------|--------|---------|--------|------");

//         // Test each checkpoint
//         for (uint256 c = 0; c < checkpoints.length; c++) {
//             uint256 trainerCount = checkpoints[c];
//             address[] memory trainers = new address[](trainerCount);
//             for (uint256 i = 0; i < trainerCount; i++) {
//                 trainers[i] = allTrainers[i];
//             }

//             // Gas for registering trainers
//             uint256 gasRegister;
//             {
//                 // Reset contract
//                 privateShapley = new ImprovedPrivateShapley(address(mockToken));
//                 mockToken.mint(address(privateShapley), 1_000_000 * 10 ** 18);

//                 // Create round
//                 uint256 currentTime = block.timestamp;
//                 privateShapley.createRound(
//                     roundId,
//                     currentTime,
//                     currentTime + DAY
//                 );

//                 // Register testers
//                 bool[] memory flags = new bool[](testers.length);
//                 for (uint256 i = 0; i < testers.length; i++) {
//                     flags[i] = true;
//                 }
//                 privateShapley.setTesters(testers, flags);

//                 // Measure gas for registering trainers
//                 uint256 gasStart = gasleft();
//                 privateShapley.registerTrainers(trainers);
//                 gasRegister = gasStart - gasleft();
//             }

//             // Gas for coalition operations
//             uint256 gasCommit;
//             uint256 gasPublish;
//             uint256 gasReveal;
//             uint256 gasClaim;
//             {
//                 // Generate a coalition with all trainers
//                 (
//                     bytes32 id,
//                     bytes32 bitfield,
//                     bytes32 nonce,
//                     bytes32 commitment
//                 ) = generateCoalition(trainers);

//                 // Prepare arrays for function calls
//                 bytes32[] memory ids = new bytes32[](1);
//                 bytes32[] memory commitments = new bytes32[](1);
//                 bytes32[] memory bitfields = new bytes32[](1);
//                 bytes32[] memory nonces = new bytes32[](1);
//                 uint256[] memory scores = new uint256[](1);

//                 ids[0] = id;
//                 commitments[0] = commitment;
//                 bitfields[0] = bitfield;
//                 nonces[0] = nonce;
//                 scores[0] = 100;

//                 // Measure gas for commit
//                 uint256 gasStart = gasleft();
//                 privateShapley.commitCoalitions(roundId, ids, commitments);
//                 gasCommit = gasStart - gasleft();

//                 // Measure gas for publish
//                 vm.prank(testers[0]);
//                 gasStart = gasleft();
//                 privateShapley.publishResults(roundId, ids, scores);
//                 gasPublish = gasStart - gasleft();

//                 // Measure gas for reveal
//                 gasStart = gasleft();
//                 privateShapley.revealCoalitions(
//                     roundId,
//                     ids,
//                     bitfields,
//                     nonces
//                 );
//                 gasReveal = gasStart - gasleft();

//                 // Measure gas for claim (first trainer)
//                 vm.prank(trainers[0]);
//                 gasStart = gasleft();
//                 privateShapley.claimRewards(roundId, ids);
//                 gasClaim = gasStart - gasleft();
//             }

//             // Log results
//             console.log("Trainers: %d", trainerCount);
//             console.log("Register: %d", gasRegister);
//             console.log("Commit: %d", gasCommit);
//             console.log("Publish: %d", gasPublish);
//             console.log("Reveal: %d", gasReveal);
//             console.log("Claim: %d", gasClaim);
//         }
//     }

//     // Test to measure gas usage for registering trainers in different batch sizes
//     function testGasScalingByBatchSize() public {
//         // Define batch sizes to test
//         uint256[] memory batchSizes = new uint256[](8);
//         batchSizes[0] = 1; // 1 trainer per batch
//         batchSizes[1] = 5; // 5 trainers per batch
//         batchSizes[2] = 10; // 10 trainers per batch
//         batchSizes[3] = 25; // 25 trainers per batch
//         batchSizes[4] = 50; // 50 trainers per batch (max batch size)
//         batchSizes[5] = 100; // 100 trainers (2 batches)
//         batchSizes[6] = 200; // 200 trainers (4 batches)
//         batchSizes[7] = 255; // 255 trainers (6 batches - 50+50+50+50+50+5)

//         // Print header
//         console.log("\n=== Gas Usage by Batch Size (Trainer Registration) ===");
//         console.log(
//             "Trainers | Total Gas | Gas Per Trainer | Number of Batches"
//         );
//         console.log(
//             "---------|-----------|----------------|------------------"
//         );

//         // Test each batch size
//         for (uint256 b = 0; b < batchSizes.length; b++) {
//             uint256 trainerCount = batchSizes[b];

//             // Reset contract
//             privateShapley = new ImprovedPrivateShapley(address(mockToken));
//             mockToken.mint(address(privateShapley), 1_000_000 * 10 ** 18);

//             // Create round
//             uint256 currentTime = block.timestamp;
//             privateShapley.createRound(roundId, currentTime, currentTime + DAY);

//             // Register testers
//             bool[] memory flags = new bool[](testers.length);
//             for (uint256 i = 0; i < testers.length; i++) {
//                 flags[i] = true;
//             }
//             privateShapley.setTesters(testers, flags);

//             // Determine number of batches
//             uint256 maxBatchSize = 50; // Based on contract's MAX_BATCH_SIZE
//             uint256 numBatches = (trainerCount + maxBatchSize - 1) /
//                 maxBatchSize; // Ceiling division

//             // Register trainers in batches
//             uint256 totalGas = 0;
//             uint256 trainersRegistered = 0;

//             for (uint256 i = 0; i < numBatches; i++) {
//                 uint256 batchStart = i * maxBatchSize;
//                 uint256 batchEnd = batchStart + maxBatchSize;
//                 if (batchEnd > trainerCount) batchEnd = trainerCount;
//                 uint256 batchSize = batchEnd - batchStart;

//                 address[] memory batchTrainers = new address[](batchSize);
//                 for (uint256 j = 0; j < batchSize; j++) {
//                     batchTrainers[j] = allTrainers[batchStart + j];
//                 }

//                 uint256 gasStart = gasleft();
//                 privateShapley.registerTrainers(batchTrainers);
//                 uint256 gasUsed = gasStart - gasleft();

//                 totalGas += gasUsed;
//                 trainersRegistered += batchSize;
//             }

//             // Calculate gas per trainer
//             uint256 gasPerTrainer = totalGas / trainerCount;

//             // Log results
//             console.log("Trainers: %d", trainerCount);
//             console.log("Total Gas: %d", totalGas);
//             console.log("Gas Per Trainer: %d", gasPerTrainer);
//             console.log("Number of Batches: %d", numBatches);
//         }
//     }

//     // Test to measure gas scaling for coalition operations with different numbers of trainers
//     function testGasScalingByCoalitionSize() public {
//         // Define trainer count checkpoints
//         uint256[] memory checkpoints = new uint256[](6);
//         checkpoints[0] = 1; // 1 trainer
//         checkpoints[1] = 10; // 10 trainers
//         checkpoints[2] = 50; // 50 trainers
//         checkpoints[3] = 100; // 100 trainers
//         checkpoints[4] = 200; // 200 trainers
//         checkpoints[5] = 255; // 255 trainers (maximum)

//         // First register all trainers
//         privateShapley.registerTrainers(allTrainers);

//         // Print header
//         console.log("\n=== Gas Usage by Coalition Size ===");
//         console.log(
//             "Trainers in Coalition | Commit | Publish | Reveal | Claim"
//         );
//         console.log("--------------------|--------|---------|--------|------");

//         // Test each checkpoint
//         for (uint256 c = 0; c < checkpoints.length; c++) {
//             uint256 coalitionSize = checkpoints[c];

//             // Generate a coalition with specific number of trainers
//             address[] memory coalitionTrainers = new address[](coalitionSize);
//             for (uint256 i = 0; i < coalitionSize; i++) {
//                 coalitionTrainers[i] = allTrainers[i];
//             }

//             (
//                 bytes32 id,
//                 bytes32 bitfield,
//                 bytes32 nonce,
//                 bytes32 commitment
//             ) = generateCoalition(coalitionTrainers);

//             // Prepare arrays for function calls
//             bytes32[] memory ids = new bytes32[](1);
//             bytes32[] memory commitments = new bytes32[](1);
//             bytes32[] memory bitfields = new bytes32[](1);
//             bytes32[] memory nonces = new bytes32[](1);
//             uint256[] memory scores = new uint256[](1);

//             ids[0] = id;
//             commitments[0] = commitment;
//             bitfields[0] = bitfield;
//             nonces[0] = nonce;
//             scores[0] = 100;

//             // Measure gas for commit
//             uint256 gasStart = gasleft();
//             privateShapley.commitCoalitions(roundId, ids, commitments);
//             uint256 gasCommit = gasStart - gasleft();

//             // Measure gas for publish
//             vm.prank(testers[0]);
//             gasStart = gasleft();
//             privateShapley.publishResults(roundId, ids, scores);
//             uint256 gasPublish = gasStart - gasleft();

//             // Measure gas for reveal
//             gasStart = gasleft();
//             privateShapley.revealCoalitions(roundId, ids, bitfields, nonces);
//             uint256 gasReveal = gasStart - gasleft();

//             // Measure gas for claim (first trainer)
//             vm.prank(allTrainers[0]);
//             gasStart = gasleft();
//             privateShapley.claimRewards(roundId, ids);
//             uint256 gasClaim = gasStart - gasleft();

//             // Log results
//             console.log("Trainers in Coalition: %d", coalitionSize);
//             console.log("Commit: %d", gasCommit);
//             console.log("Publish: %d", gasPublish);
//             console.log("Reveal: %d", gasReveal);
//             console.log("Claim: %d", gasClaim);
//         }
//     }
// }
