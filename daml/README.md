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

## Running

### Prerequisites

- **JDK 11+** (the sandbox and script runner are JVM-based; tested with
  OpenJDK 21).
- **Daml SDK 2.9.4**. Standard install:

  ```bash
  curl -sSL https://get.daml.com/ | sh -s 2.9.4
  export PATH="$HOME/.daml/bin:$PATH"
  ```

  If `get.daml.com` is blocked in your environment (it is in some CI
  sandboxes), the same SDK ships as a GitHub release artifact:

  ```bash
  curl -sSL -o /tmp/daml-sdk.tar.gz \
    https://github.com/digital-asset/daml/releases/download/v2.9.4/daml-sdk-2.9.4-linux.tar.gz
  tar xzf /tmp/daml-sdk.tar.gz -C /tmp && /tmp/sdk-2.9.4/install.sh
  export PATH="$HOME/.daml/bin:$PATH"
  ```

### Build and test (fast loop)

All commands run from this `daml/` directory:

```bash
daml build   # compiles to .daml/dist/rizemind-daml-0.1.0.dar
daml test    # runs all Daml Scripts under Rizemind/Test/ on an in-memory ledger
```

`daml test` compiles in memory and does **not** refresh the DAR — run
`daml build` again before uploading anywhere.

For an IDE with type-on-hover, jump-to-definition and inline script
results, open the project with `daml studio` (VS Code).

### Run against a local Canton sandbox

The demo script manipulates ledger time with `setTime`, so the sandbox must
run in **static time** mode:

```bash
# terminal 1: start a local Canton sandbox (takes ~30s to be ready)
daml sandbox --static-time --port 6865

# terminal 2: upload the package, then drive a full training round
daml build
daml ledger upload-dar --host localhost --port 6865 .daml/dist/rizemind-daml-0.1.0.dar
daml script --dar .daml/dist/rizemind-daml-0.1.0.dar \
  --script-name Rizemind.Test.RoundFlow:roundFlow \
  --ledger-host localhost --ledger-port 6865 --static-time
```

Two sharp edges, both observed in practice:

- **Ledger time is monotonic.** `setTime` cannot move backwards, so the
  round-flow script runs once per sandbox; restart the sandbox to run it
  again.
- **Stale DARs fail confusingly.** If you edit code and re-upload without
  `daml build`, the ledger keeps executing the old package. When sandbox
  behaviour contradicts `daml test`, rebuild first.

To poke at the resulting ledger state interactively, use
`daml repl .daml/dist/rizemind-daml-0.1.0.dar --ledger-host localhost --ledger-port 6865`
or point Navigator at the sandbox with `daml navigator server localhost 6865`.

## Contributing

### Ground rules

1. **`forge/src` is the source of truth.** This port tracks the Solidity
   contracts; it does not fork their semantics. If a behaviour change is
   needed, land it (or at least agree on it) on the Solidity side first,
   then mirror it here.
2. **Keep the module map honest.** Every module's doc comment names the
   Solidity file(s) it ports, and choice-level comments name the function
   they mirror (`-- | Mirrors registerRoundContribution(...)`). When you
   add or move logic, update the mapping table and the
   "Translation decisions" section above.
3. **Intentional divergences are documented, not silent.** Anything that
   deviates from the EVM behaviour (numerics, randomness, visibility,
   clamping) belongs in "Known simplifications" with a one-line rationale.

### Workflow

```bash
daml build && daml test       # must both pass before pushing
```

- Put pure logic (math, sampling, bit twiddling) in standalone modules like
  `Bits`, `Rng`, `TaskAssignment`, `Contribution` — pure functions are
  directly reusable in test scripts, as `Test/RoundFlow.daml` does when it
  recomputes coalition masks.
- Workflow state and choices live in `Swarm.daml`; new participant-facing
  operations should be choices on `Swarm` so the phase machine (`advance`)
  is applied uniformly.
- Every workflow change needs Daml Script coverage in `Rizemind/Test/`:
  the happy path plus `submitMustFail` cases for each authorization or
  phase guard you add. Scripts must stay sandbox-compatible (no
  IDE-ledger-only features) so they double as live demos.

### Daml pitfalls to watch for

Lessons already paid for while getting this port green:

- **Daml is strict.** Never pass `error ...` as a default argument
  (e.g. to `fromOptional`) — it evaluates even when unused. Pattern-match
  and `abort` instead.
- **Visibility is explicit.** On Canton a party can only fetch/exercise
  contracts it is a stakeholder or observer of, even when authorization
  would otherwise pass. If a party must read or merge a contract by key
  (as evaluators and claiming trainers do with `EvaluationResult`), add it
  to the observers and say why in a comment.
- **`daml test` is not the sandbox.** The in-memory ledger forgives
  nothing about types but differs operationally (time handling, package
  upload). Validate workflow changes against `daml sandbox` before calling
  them done.

### Versioning

Bump `version` in `daml.yaml` for any change to templates or choice
signatures — Daml packages are content-addressed, and live ledgers can only
migrate between properly versioned packages (see Smart Contract Upgrade in
Daml 2.9+).

## Status / roadmap

Open items, roughly in priority order:

- Wire `DemocraticWhitelist` into `canTrain` as an alternative membership
  policy (mirroring how swarms choose an `IAccessControl` implementation).
- Replace `RewardToken`'s direct `Transfer` with propose-accept or Daml
  Finance holdings.
- Per-round `numSamples` override with previous-round fallback, as in
  `ContributionCalculator`.
- A CI job running `daml build && daml test`, alongside `forge-test.yml`.
