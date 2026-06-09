# Rizemind DAML port (draft)

A draft port of the Solidity contracts in `forge/src` to
[Daml](https://docs.daml.com/), targeting Daml 2.x / Canton. It is a
**semantic** port, not a line-by-line one: Daml's ledger model (party-based
authorization, immutable contracts, no autonomous execution) makes several
EVM mechanisms unnecessary and forces a different shape for others. The
project builds with SDK 2.9.4 and the end-to-end round test
(`Rizemind.Test.RoundFlow`) passes under `daml test`.

## Module map

| Solidity (`forge/src`) | Daml (`daml/Rizemind`) | Notes |
| --- | --- | --- |
| `swarm/SwarmV1.sol`, `swarm/registry/SwarmCore.sol`, `training/RoundTraining.sol`, `training/BaseTrainingPhases.sol` | `Swarm.daml` | One `Swarm` template carries round counter, phase state, module config and per-round counters |
| `swarm/registry/RoundTrainerRegistry.sol` | `Swarm.daml` (`TrainerSubmission`) | One keyed contract per (round, trainer) |
| `swarm/registry/RoundEvaluatorRegistry.sol` | `Swarm.daml` (`EvaluatorSlot`) | One keyed contract per (round, evaluator) |
| `swarm/registry/CertificateRegistry.sol` | `Certificate.daml` | Keyed contract per certificate id |
| `access/IAccessControl.sol`, `access/BaseAccessControl.sol` | party lists in `Swarm.daml` | Roles become party sets; `msg.sender` checks become choice controllers |
| `access/FLDemocraticWhitelist.sol` | `AccessControl.daml` | Vote map + threshold, faithful to the original |
| `sampling/ISelector.sol`, `AlwaysSampled.sol`, `RandomSampling.sol` | `Selector.daml` | A pure `SelectorPolicy` value instead of a contract address |
| `scheduling/TaskAssignment.sol` | `TaskAssignment.daml` | Exact port of the affine assignment formulas |
| `contribution/EvaluationStorage.sol`, `ShapleyValueCalculator.sol`, `ContributionCalculator.sol` | `Contribution.daml` | `EvaluationResult` contracts + pure Shapley estimator (same weights, same merge-by-averaging) |
| `compensation/SimpleMintCompensation.sol` | `Compensation.daml` | ERC-20 mint becomes `RewardToken` contract creation |
| `randomness/*` | `Rng.daml` | Lehmer LCG stand-in for keccak-seeded `RandPerm` |
| `swarm/SwarmV1Factory.sol` + sub-factories | `Factory.daml` | Reduced to a workflow anchor; no proxies/CREATE2 needed |
| — | `Bits.daml`, `Types.daml` | Bitmask arithmetic (Daml-LF has no bitwise ops) and shared types |
| `test/` (Foundry) | `Test/RoundFlow.daml` | Daml Script covering one full round: start → contribute → evaluate → claim |

## Translation decisions

1. **Authorization.** `msg.sender` checks, EIP-712 signature recovery and
   ERC-165/ERC-5267 introspection all collapse into Daml's intrinsic
   authorization: a choice exercised by a party *is* the proof that party
   acted. The off-ledger `EthAccountStrategy` signature flow would be
   replaced by parties exercising choices through their participant nodes.
2. **Mutable storage.** Namespaced-storage structs become template payloads;
   every state change archives and recreates the contract. Solidity
   mappings keyed by `(roundId, address)` become contract keys.
3. **Time.** `BaseTrainingPhases` relies on `block.timestamp` evaluated on
   every call. Daml contracts cannot wake themselves, so `advance` in
   `Swarm.daml` is a pure replay of `updatePhase()` from ledger time; every
   choice applies it, and phase boundaries roll forward by exact TTLs so the
   schedule is independent of when transactions happen to arrive.
4. **Events.** All Solidity events (`PhaseTransition`, `TrainerRegistered`,
   `CompensationSent`, …) are subsumed by the ledger transaction log, which
   is the audit trail on Canton.
5. **Upgradeability.** UUPS/ERC-1967 proxies have no Daml counterpart in
   this draft; Daml handles upgrades through package versioning (and Smart
   Contract Upgrade in Daml 2.9+).
6. **Numerics.** 1e18/1e6 fixed-point `int256` math becomes `Decimal`.
   Negative contributions mint nothing (the Solidity code truncates them
   through a `uint64` cast — arguably a bug worth revisiting there).
7. **Randomness.** `RandPerm.rand(keccak256(address, roundId), i, 2^n)`
   becomes a Lehmer LCG seeded from the swarm id (`Rng.daml`). Like
   `WeakSeedProvider`, it is predictable; a production port should source
   randomness differently.
8. **Privacy and visibility.** On Canton, contracts are only visible to
   their stakeholders — sub-transaction privacy the EVM version cannot
   offer. The flip side: state that is world-readable on a public chain
   must be shared explicitly. `EvaluationResult` lists the swarm's members
   as observers because evaluators fetch it to merge results and trainers
   fetch it to compute their Shapley value at claim time.

## Known simplifications

- The selector ratio uses basis points instead of an 18-decimal ratio.
- `numSamples` is fixed per swarm rather than per round (Solidity allows a
  per-round override with fallback to the previous round).
- `RewardToken.Transfer` skips the propose-accept pattern; a production port
  should use Daml Finance holdings.
- Trainer/evaluator membership is embedded in the `Swarm` payload; the
  `DemocraticWhitelist` module is provided but not yet wired into
  `canTrain`.
- `SetCertificate` is aggregator-only; the Solidity interface leaves the
  caller policy to the implementer.

## Building and testing

```bash
cd daml
daml build
daml test   # runs Rizemind.Test.RoundFlow.roundFlow
```
