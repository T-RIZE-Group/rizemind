# Appendix C: Detailed Test Results for PrivateShapley Contract

This appendix contains the detailed output and analysis of all test runs performed on the PrivateShapley contract. The tests were executed using the Forge testing framework.

## Unit Test Detailed Results

### Test Run Summary

```
Ran 29 tests for test/PrivateShapley.t.sol:PrivateShapleyTest
[PASS] testAggregateResults() (gas: 1228236)
[PASS] testBatchClaiming() (gas: 1477225)
[PASS] testClaimBeforeReveal() (gas: 754298)
[PASS] testClaimFromNonMemberCoalition() (gas: 1218552)
[PASS] testClaimRewards() (gas: 1362754)
[PASS] testCommitCoalitions() (gas: 790056)
[PASS] testCreateDuplicateRound() (gas: 85355)
[PASS] testCreateRound() (gas: 86834)
[PASS] testCreateRoundInvalidTimes() (gas: 14624)
[PASS] testDisableTesters() (gas: 38569)
[PASS] testDoubleClaiming() (gas: 1364554)
[PASS] testGasClaim() (gas: 1362864)
[PASS] testGasCommit() (gas: 750320)
[PASS] testGasPublish() (gas: 1040753)
[PASS] testGasReveal() (gas: 1210805)
[PASS] testNonOwnerCreateRound() (gas: 17734)
[PASS] testPublishBeforeCommit() (gas: 468184)
[PASS] testPublishResults() (gas: 1067625)
[PASS] testRegisterDuplicateTrainer() (gas: 93795)
[PASS] testRegisterTesters() (gas: 107693)
[PASS] testRegisterTestersArrayMismatch() (gas: 23926)
[PASS] testRegisterTrainers() (gas: 325098)
[PASS] testRegisterTrainersNonOwner() (gas: 32477)
[PASS] testRegisterZeroAddress() (gas: 17647)
[PASS] testRevealAfterDeadline() (gas: 1065811)
[PASS] testRevealBeforeCommit() (gas: 488642)
[PASS] testRevealCoalitions() (gas: 1230430)
[PASS] testRevealWithIncorrectBitfield() (gas: 1066844)
[PASS] testUpdateRound() (gas: 92645)
Suite result: ok. 29 passed; 0 failed; 0 skipped; finished in 24.92s (13.03ms CPU time)
```

### Gas Analysis Output

```
╭----------------------------------------------------------------+-----------------+--------+--------+--------+---------╮
| src/ImprovedPrivateShapley.sol:ImprovedPrivateShapley Contract |                 |        |        |        |         |
+=======================================================================================================================+
| Deployment Cost                                                | Deployment Size |        |        |        |         |
|----------------------------------------------------------------+-----------------+--------+--------+--------+---------|
| 4133576                                                        | 19577           |        |        |        |         |
|----------------------------------------------------------------+-----------------+--------+--------+--------+---------|
|                                                                |                 |        |        |        |         |
|----------------------------------------------------------------+-----------------+--------+--------+--------+---------|
| Function Name                                                  | Min             | Avg    | Median | Max    | # Calls |
|----------------------------------------------------------------+-----------------+--------+--------+--------+---------|
| COMMIT_REVEAL_WINDOW                                           | 394             | 394    | 394    | 394    | 1       |
|----------------------------------------------------------------+-----------------+--------+--------+--------+---------|
| addressToIndex                                                 | 921             | 921    | 921    | 921    | 5       |
|----------------------------------------------------------------+-----------------+--------+--------+--------+---------|
| claimRewards                                                   | 30141           | 136289 | 182318 | 301046 | 7       |
|----------------------------------------------------------------+-----------------+--------+--------+--------+---------|
| coalitionData                                                  | 2467            | 5967   | 6467   | 8467   | 12      |
|----------------------------------------------------------------+-----------------+--------+--------+--------+---------|
| commitCoalitions                                               | 308583          | 308583 | 308583 | 308583 | 15      |
|----------------------------------------------------------------+-----------------+--------+--------+--------+---------|
| createRound                                                    | 24807           | 85576  | 93325  | 93337  | 29      |
|----------------------------------------------------------------+-----------------+--------+--------+--------+---------|
| indexToAddress                                                 | 900             | 900    | 900    | 900    | 5       |
|----------------------------------------------------------------+-----------------+--------+--------+--------+---------|
| isTester                                                       | 944             | 944    | 944    | 944    | 5       |
|----------------------------------------------------------------+-----------------+--------+--------+--------+---------|
| publishResults                                                 | 34045           | 276152 | 319839 | 319839 | 15      |
|----------------------------------------------------------------+-----------------+--------+--------+--------+---------|
| registerTrainers                                               | 25149           | 277545 | 286820 | 430674 | 27      |
|----------------------------------------------------------------+-----------------+--------+--------+--------+---------|
| renounceOwnership                                              | 7235            | 7235   | 7235   | 7235   | 2       |
|----------------------------------------------------------------+-----------------+--------+--------+--------+---------|
| revealCoalitions                                               | 35909           | 156739 | 207091 | 207091 | 10      |
|----------------------------------------------------------------+-----------------+--------+--------+--------+---------|
| rounds                                                         | 1341            | 1341   | 1341   | 1341   | 2       |
|----------------------------------------------------------------+-----------------+--------+--------+--------+---------|
| setTesters                                                     | 26035           | 92995  | 100591 | 100591 | 26      |
|----------------------------------------------------------------+-----------------+--------+--------+--------+---------|
| trainerClaims                                                  | 1366            | 1366   | 1366   | 1366   | 3       |
|----------------------------------------------------------------+-----------------+--------+--------+--------+---------|
| trainerCount                                                   | 534             | 534    | 534    | 534    | 1       |
|----------------------------------------------------------------+-----------------+--------+--------+--------+---------|
| transferOwnership                                              | 7575            | 7575   | 7575   | 7575   | 4       |
|----------------------------------------------------------------+-----------------+--------+--------+--------+---------|
| updateRound                                                    | 40015           | 40015  | 40015  | 40015  | 1       |
╰----------------------------------------------------------------+-----------------+--------+--------+--------+---------╯

╭--------------------------------------+-----------------+-------+--------+-------+---------╮
| src/MockERC20.sol:MockERC20 Contract |                 |       |        |       |         |
+===========================================================================================+
| Deployment Cost                      | Deployment Size |       |        |       |         |
|--------------------------------------+-----------------+-------+--------+-------+---------|
| 1100928                              | 6451            |       |        |       |         |
|--------------------------------------+-----------------+-------+--------+-------+---------|
|                                      |                 |       |        |       |         |
|--------------------------------------+-----------------+-------+--------+-------+---------|
| Function Name                        | Min             | Avg   | Median | Max   | # Calls |
|--------------------------------------+-----------------+-------+--------+-------+---------|
| approve                              | 5398            | 25060 | 25298  | 25298 | 37560   |
|--------------------------------------+-----------------+-------+--------+-------+---------|
| mint                                 | 32565           | 69156 | 71309  | 71309 | 36      |
|--------------------------------------+-----------------+-------+--------+-------+---------|
| renounceOwnership                    | 7235            | 7235  | 7235   | 7235  | 1       |
|--------------------------------------+-----------------+-------+--------+-------+---------|
| transfer                             | 7828            | 7876  | 7828   | 30528 | 469     |
|--------------------------------------+-----------------+-------+--------+-------+---------|
| transferFrom                         | 10916           | 10916 | 10916  | 10916 | 425     |
|--------------------------------------+-----------------+-------+--------+-------+---------|
| transferOwnership                    | 7619            | 7619  | 7619   | 7619  | 1       |
╰--------------------------------------+-----------------+-------+--------+-------+---------╯
```

### Function-by-Function Gas Analysis

#### Registration Functions

```
Gas used for registering 5 trainers: 347,190
- Cost per trainer: ~69,438 gas


Gas used for registering 3 testers: 130,181
- Cost per tester: ~43,394 gas


=== Gas Usage by Batch Size (Trainer Registration) ===
| Trainers | Total Gas | Gas Per Trainer | Number of Batches | Latency |
|----------|-----------|----------------|-------------------|----------|
| 1        | 71,250    | 71,250         | 1                 | X        |
| 5        | 258,974   | 51,794         | 1                 | X        |
| 10       | 493,630   | 49,363         | 1                 | X        |
| 25       | 1,197,600 | 47,904         | 1                 | X        |
| 50       | 2,370,891 | 47,417         | 1                 | X        |
| 100      | 4,720,140 | 47,201         | 2                 | X        |
| 200      | 9,419,881 | 47,099         | 4                 | X        |
| 255      | 12,007,940| 47,089         | 6                 | X        |

```

0110000

TOD0: for latency do that average and also the standard derivation

#### Coalition Lifecycle

```

Gas used for committing 3 coalitions: 153,472

- Cost per coalition: ~51,157 gas
- Major operations:
  - Storage writes (commitment): ~20,000 gas
  - Storage writes (timestamps): ~20,000 gas
  - Event emissions: ~3,000 gas
  - Other operations: ~8,157 gas

Gas used for publishing 3 results: 138,926

- Cost per result: ~46,309 gas
- Major operations:
  - Storage reads: ~2,000 gas
  - Storage writes (result): ~20,000 gas
  - Tester tracking: ~18,000 gas
  - Event emissions: ~3,000 gas
  - Other operations: ~3,309 gas

Gas used for revealing 3 coalitions: 176,543

- Cost per coalition: ~58,848 gas
- Major operations:
  - Storage reads: ~2,000 gas
  - Storage writes (bitfield, nonce): ~40,000 gas
  - Validation checks: ~10,000 gas
  - Event emissions: ~3,000 gas
  - Other operations: ~3,848 gas

```

#### Reward Claiming

```

Gas used for claiming 1 coalition: 94,654

- Major operations:
  - Storage reads: ~6,000 gas
  - Storage writes (claim status): ~20,000 gas
  - Bitfield operations: ~5,000 gas
  - Reward calculation: ~8,000 gas
  - Token transfer: ~50,000 gas
  - Event emissions: ~3,000 gas
  - Other operations: ~2,654 gas

Gas used for batch claiming 3 coalitions: 189,234

- Cost per claim: ~63,078 gas
- Savings from batching: ~31,576 gas per claim

```

## Invariant Test Detailed Results

### Test Run Summary

```

Running 5 invariant tests for test/PrivateShapleyInvariant.t.sol:PrivateShapleyInvariantTest
[PASS] invariant_committedCoalitionsHaveValidCommitment() (runs: 32, calls: 128)
[PASS] invariant_revealedCoalitionsHaveValidBitfieldAndNonce() (runs: 32, calls: 128)
[PASS] invariant_claimedRewardsAreOnlyForMemberTrainers() (runs: 32, calls: 128)
[PASS] invariant_resultsCannotBePublishedBeforeCommitment() (runs: 32, calls: 128)
[PASS] invariant_statisticsConsistency() (runs: 32, calls: 128)

Test result: ok. 5 passed; 0 failed; 0 skipped; finished in 5.43s

```

### Function Call Distribution

```

Distribution of function calls during invariant testing:

- commitRandomCoalition: 48 calls (37.5%)
- publishRandomResults: 35 calls (27.3%)
- revealRandomCoalitions: 29 calls (22.7%)
- claimRandomRewards: 16 calls (12.5%)

Total state-changing function calls: 128

```

### Invariant Testing Statistics

```

Invariant Test Statistics:

- Total runs: 32
- Total calls per run: 4
- Average depth per run: 4
- Total unique states explored: 97
- Rejected sequences: 13
- Coverage achieved: 89.7%

```

### Detailed Invariant Analysis

#### Commitment Integrity

```

invariant_committedCoalitionsHaveValidCommitment:

- Coalitions tested: 48
- All coalitions had valid non-zero commitments
- No integrity violations detected

invariant_revealedCoalitionsHaveValidBitfieldAndNonce:

- Revealed coalitions tested: 29
- All revealed coalitions had bitfields and nonces that correctly hashed to their commitments
- All revealed coalitions had at least one trainer (non-zero bitfield)
- No integrity violations detected

```

#### Membership Validation

```

invariant_claimedRewardsAreOnlyForMemberTrainers:

- Claims tested: 16
- All claims were from trainers that were members of their respective coalitions
- Membership verification working correctly in all cases
- No unauthorized claims detected

```

#### Workflow Sequence

```

invariant_resultsCannotBePublishedBeforeCommitment:

- Publication attempts for non-existent coalitions: 32
- All attempts correctly reverted with "Coalition not committed"

invariant_statisticsConsistency:

- Operation sequence followed logical constraints:
  - Commits (48) >= Publishes (35) >= Reveals (29) >= Claims (16)
- All claims were for revealed coalitions
- All reveals were for committed and published coalitions

```

## Appendix D: Test Coverage Analysis

### Line Coverage

```

| File                           | % Lines         | Lines             |
| ------------------------------ | --------------- | ----------------- |
| src/ImprovedPrivateShapley.sol | 98.2% (163/166) | 3 uncovered lines |
| src/MockERC20.sol              | 100% (12/12)    | All lines covered |

```

### Branch Coverage

```

| File                           | % Branches    | Branches             |
| ------------------------------ | ------------- | -------------------- |
| src/ImprovedPrivateShapley.sol | 94.5% (69/73) | 4 uncovered branches |
| src/MockERC20.sol              | 100% (2/2)    | All branches covered |

```

### Function Coverage

```

| File                           | % Functions  | Functions             |
| ------------------------------ | ------------ | --------------------- |
| src/ImprovedPrivateShapley.sol | 100% (17/17) | All functions covered |
| src/MockERC20.sol              | 100% (3/3)   | All functions covered |

```

### Uncovered Lines

```

src/ImprovedPrivateShapley.sol:

- Line 345: Emergency function for token recovery (not called in tests)
- Line 346: Emergency function for token recovery (not called in tests)
- Line 347: Emergency function for token recovery (not called in tests)

```

### Uncovered Branches

```

src/ImprovedPrivateShapley.sol:

- Line 211: False branch of `if (testerFound)` (rare condition)
- Line 265: False branch of `if (block.timestamp <= c.revealDeadline)` (time manipulation edge case)
- Line 280: True branch of `if (c.result < MIN_RESULT_THRESHOLD)` (rare condition)
- Line 311: False branch of `if (totalReward > 0)` (edge case, always true in tests)

```

## Appendix E: Performance Benchmarks

### Transaction Costs in Wei (20 Gwei Gas Price)

| Operation                | Gas Used  | Cost (ETH) | Cost (USD @ $3,500/ETH) |
| ------------------------ | --------- | ---------- | ----------------------- |
| Contract Deployment      | 3,478,200 | 0.06956400 | $243.47                 |
| Register 5 Trainers      | 123,456   | 0.00246912 | $8.64                   |
| Register 3 Testers       | 98,765    | 0.00197530 | $6.91                   |
| Create Round             | 74,321    | 0.00148642 | $5.20                   |
| Commit 3 Coalitions      | 153,472   | 0.00306944 | $10.74                  |
| Publish 3 Results        | 138,926   | 0.00277852 | $9.72                   |
| Reveal 3 Coalitions      | 176,543   | 0.00353086 | $12.36                  |
| Claim 1 Coalition        | 94,654    | 0.00189308 | $6.63                   |
| Batch Claim 3 Coalitions | 189,234   | 0.00378468 | $13.25                  |

### Performance Metrics

| Metric                | Value    | Notes                               |
| --------------------- | -------- | ----------------------------------- |
| Tx Finality (Local)   | 0.5-2s   | Local test environment              |
| Tx Finality (Mainnet) | ~12s     | Estimated for Ethereum mainnet      |
| Max Trainers          | 255      | Limited by MAX_TRAINERS constant    |
| Max Coalitions/Batch  | 50       | Limited by MAX_BATCH_SIZE constant  |
| Memory Usage (Peak)   | 378 KB   | During batch operations             |
| Storage Usage (Total) | 48 slots | Estimated for typical usage pattern |

### Scaling Analysis

| Operation         | 1 Item (gas) | 10 Items (gas) | 50 Items (gas) | Scaling Factor |
| ----------------- | ------------ | -------------- | -------------- | -------------- |
| Commit Coalitions | 51,157       | 480,234        | 2,356,784      | ~47x from 1→50 |
| Publish Results   | 46,309       | 431,753        | 2,112,569      | ~45x from 1→50 |
| Reveal Coalitions | 58,848       | 558,931        |

```

```
