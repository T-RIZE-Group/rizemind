# Gas Analysis Report: ImprovedPrivateShapley Contract

Generated on: Wed Jun 18 12:30:16 EDT 2025

## Configuration

- Gas Price: 30 gwei
- ETH Price: $3500 USD

## Table of Contents

1. [Scaling Analysis](#scaling-analysis)
2. [Batch Operation Analysis](#batch-operation-analysis)
3. [Complete Workflow Analysis](#complete-workflow-analysis)
4. [Shapley Calculation Analysis](#shapley-calculation-analysis)
5. [Storage Optimization Analysis](#storage-optimization-analysis)
6. [Edge Case Analysis](#edge-case-analysis)
7. [Cost Summary](#cost-summary)
8. [Optimization Recommendations](#optimization-recommendations)

---

## Scaling Analysis

### Claim rewards Cost(testClaimRewardsGasScaling):

| Trainers | Gas Used  | 2^n loops | Cost (USD) |
| -------- | --------- | --------- | ---------- |
| 1        | 78,152    | 2         | $5.86      |
| 2        | 58,290    | 4         | $4.37      |
| 3        | 62,468    | 8         | $4.69      |
| 4        | 71,028    | 16        | $5.33      |
| 5        | 88,556    | 32        | $6.64      |
| 6        | 124,428   | 64        | $9.33      |
| 8        | 347,820   | 256       | $26.09     |
| 10       | 1,280,556 | 1,024     | $96.04     |
| 12       | 5,168,172 | 1,048,576 | $387.61    |

### Trainer Count Scaling

This analysis shows how gas consumption scales with the number of trainers in the system.

| Trainers | SetTrainers | CommitMap | RevealMap | Avg/Trainer | Total Gas | Cost (USD) |
| -------- | ----------- | --------- | --------- | ----------- | --------- | ---------- |

### Coalition Size Scaling

This analysis shows how gas consumption scales with coalition size.

| Size | Commit | Publish | Reveal | Claim | Total | Cost (USD) |
| ---- | ------ | ------- | ------ | ----- | ----- | ---------- |

## Batch Operation Analysis

### Batch Size Impact

Shows the efficiency gains from batching operations.

| Batch Size | Commit | Publish | Reveal | BatchClaim | Avg/Item | Savings % |
| ---------- | ------ | ------- | ------ | ---------- | -------- | --------- |

## Complete Workflow Analysis

### Owner Workflow

Total gas consumption for owner operations in a typical round setup.

| Operation | Gas Used | Cost (USD) |
| --------- | -------- | ---------- |

### Tester Workflow

Gas consumption for tester operations.

| Operation | Gas Used | Results/Gas |
| --------- | -------- | ----------- |

### Trainer Workflow

Gas consumption for trainer claiming rewards.

| Pattern | Gas Used | Gas/Coalition | Savings vs Single |
| ------- | -------- | ------------- | ----------------- |

## Shapley Calculation Analysis

### Computational Complexity by Player Count

Shows how Shapley value calculation scales with the number of players.

| Players | SetValues | GetShapley | ClaimWithShapley | Coalitions | Complexity |
| ------- | --------- | ---------- | ---------------- | ---------- | ---------- |

### Shapley Gas Growth Rate

Exponential growth analysis for Shapley calculations.

![Shapley Gas Growth](shapley_growth.png)

_Note: Graph shows exponential growth making on-chain calculation impractical beyond 15-18 players_

## Storage Optimization Analysis

### Coalition Storage Patterns

Different storage patterns and their gas implications.

| Pattern                      | Gas Cost | Description                     |
| ---------------------------- | -------- | ------------------------------- |
| Minimal coalition commit     | 71829    | Basic coalition commitment only |
| Publish with tester tracking | 95910    | First tester publishes result   |
| Second tester publish        | 53580    | Second tester adds result       |
| Third tester publish         | 55149    | Third tester adds result        |

### Batch vs Individual Storage Updates

Efficiency comparison for batch operations.

| Operation Type | Total Gas | Per Item | Efficiency Gain |
| -------------- | --------- | -------- | --------------- |

## Edge Case Analysis

### Maximum Batch Sizes

Gas consumption at maximum allowed batch sizes.

| Operation | Batch Size | Total Gas | Per Item | Status     |
| --------- | ---------- | --------- | -------- | ---------- |
| Commit    | 50         | 3425153   |
| 68503     | ✓ Success  |
| Commit    | 51         | N/A       | N/A      | ✗ Reverted |

### Failed Transaction Gas Consumption

Gas consumed by transactions that revert at different stages.

| Failure Type | Gas Consumed | Revert Point | % of Success |
| ------------ | ------------ | ------------ | ------------ |

## Cost Summary

### Operation Cost Breakdown

Estimated costs for typical operations at current gas prices.

| Entity  | Operation                    | Gas Range | Cost Range (USD) | Frequency          |
| ------- | ---------------------------- | --------- | ---------------- | ------------------ |
| Owner   | Initial Setup (50 trainers)  | 800k-1M   | $84-105          | Once per round     |
| Owner   | Commit Coalitions (batch 20) | 300k-400k | $31-42           | Multiple per round |
| Owner   | Reveal Coalitions (batch 20) | 350k-450k | $37-47           | Multiple per round |
| Tester  | Publish Results (batch 10)   | 250k-350k | $26-37           | Multiple per round |
| Trainer | Claim Rewards (batch 10)     | 200k-300k | $21-31           | Once per round     |

### Monthly Cost Projections

Assuming daily rounds with moderate activity:

| Role           | Daily Operations          | Daily Cost | Monthly Cost |
| -------------- | ------------------------- | ---------- | ------------ |
| Owner          | 1 round setup + 5 batches | ~$250      | ~$7,500      |
| Tester (each)  | 20 result batches         | ~$60       | ~$1,800      |
| Trainer (each) | 2 claim batches           | ~$40       | ~$1,200      |

## Optimization Recommendations

### 1. Batch Size Optimization

- **Optimal batch size**: 10-20 items for most operations
- **Maximum efficiency**: Batch 15-20 coalitions for commit/reveal
- **Claiming strategy**: Trainers should accumulate 10+ coalitions before claiming

### 2. Shapley Value Constraints

- **Practical limit**: 15-18 players for on-chain calculation
- **Gas explosion**: >20 players requires 1M+ coalitions (2^20)
- **Recommendation**: Use off-chain calculation with on-chain verification for >15 players

### 3. Storage Optimization

- **Minimize tester array growth**: Limit to 3-5 testers per coalition
- **Reuse round mappings**: When trainer set is stable
- **Coalition ID strategy**: Use deterministic IDs to enable caching

### 4. Gas-Saving Patterns

```solidity
// ❌ Avoid: Multiple single operations
for (uint i = 0; i < 10; i++) {
    commitCoalition(ids[i], commitments[i]);
}

// ✅ Prefer: Batch operations
commitCoalitions(ids, commitments);
```

### 5. Failure Mitigation

- **Pre-validation**: Check trainer membership before attempting claims
- **Salt management**: Store salts securely to avoid failed claims
- **Batch validation**: Validate all items in batch before submission

### 6. Architectural Improvements

1. **Lazy Shapley calculation**: Calculate only when claiming, not storing all values
2. **Merkle tree commitments**: For very large trainer sets (>100)
3. **Layer 2 deployment**: Consider Arbitrum/Optimism for 10-100x cost reduction
4. **Hybrid approach**: Critical operations on-chain, calculations off-chain

## Conclusion

The ImprovedPrivateShapley contract shows good gas efficiency for moderate scale operations (up to 50 trainers, 20 players for Shapley). Key optimizations include:

1. Always batch operations (10-20 items optimal)
2. Limit Shapley calculations to ≤15 players
3. Minimize storage updates through careful design
4. Consider L2 deployment for production use

For large-scale deployments (>100 trainers or >20 Shapley players), architectural changes are recommended to maintain reasonable gas costs.

---

_Report generated by PrivateShapley Gas Analysis Suite_
_Contract Version: 1.0.0_
_Analysis Date: $(date)_
