# Rizemind — Tech Lead Handoff

**Audience:** Engineering & product leadership · **Purpose:** understand what Rizemind
is, how it works, and what work is required to reach a production-ready release.
**Status of the codebase:** advanced research prototype / proof-of-concept.
**Date:** 2026-07-13

---

## 1. Executive summary

Rizemind is a Python library for **collaborative, privacy-preserving machine learning**.
It lets multiple parties who do not trust each other train a shared model *without
sharing their raw data*, and it uses a blockchain to decide **who is allowed to
participate**, to **measure how much each participant actually contributed**, and to
**pay them** a token reward proportional to that contribution.

Technically it is a **thin trust-and-incentive layer wrapping the
[Flower](https://flower.ai) federated-learning framework**, plus a suite of
**Solidity smart contracts** that hold the source of truth for identity, round
state, contribution scores and rewards.

**What is genuinely built and working:**

- A complete federated-learning round loop driven by on-chain state (Flower +
  smart contracts working end-to-end in simulation and against a testnet).
- Cryptographic identity and authorization for every participant using Ethereum
  keys and EIP-712 signatures — no passwords, no shared secrets, no raw data
  leaving a client.
- **Shapley-value contribution scoring** (both a server-side and a fully
  decentralized/client-side variant) — the core differentiator: a mathematically
  principled, game-theoretic measure of each trainer's marginal value.
- An ERC-20 reward token that mints compensation proportional to contribution.
- Optional **differential privacy** (Opacus) integration, including an adaptive
  privacy-budget example that tightens privacy for high contributors.
- A modular smart-contract architecture (pluggable access control, sampling,
  scoring, compensation) deployed via a CREATE2 factory.
- A real automated test suite: 19 Solidity test files + ~31 Python unit test
  files + on-chain integration tests against a local Anvil node, all wired into CI.

**What makes it a prototype, not a product** (detail in §7):

- Several load-bearing mechanisms carry explicit `TODO`s — most importantly
  **on-chain randomness is predictable** (no VRF) and there is a **reward-minting
  integer-cast bug** that can over-mint on a negative contribution score.
- **No third-party smart-contract security audit**, and the static analyzer
  (Aderyn) is run by hand, not in CI.
- The examples — the primary onboarding surface — are **undocumented and commit
  test secrets**; deployed swarm contracts are **not upgradeable**.
- Differential-privacy examples run in `secure_mode=False` (not a real privacy
  guarantee).

**The ask:** approve a hardening program (§8) to take this from a working research
prototype to an auditable, operable production system. The heaviest items are a
**smart-contract security audit + fixes**, **verifiable randomness (VRF)**, and
**operational maturity** (docs, key management, monitoring, upgrade strategy).

---

## 2. What it does (functionalities)

| Capability | What it means | Where it lives |
|---|---|---|
| **Federated training** | Many clients train locally; only model weights (never data) are aggregated. Built on Flower's `FedAvg`. | `workflow/`, `strategies/` |
| **Cryptographic identity** | Each participant is an Ethereum account. Every message is EIP-712 signed; the server recovers the signer instead of trusting a claimed name. | `authentication/` |
| **On-chain authorization** | A smart contract decides per round who may train (`canTrain`) and who may evaluate (`canEvaluate`). | `contracts/access_control/`, `forge/src/access/` |
| **Mutual verification** | Clients verify the server is a legitimate aggregator (signed global model) before training; the server verifies each client is authorized before accepting its update. | `authentication/notary/`, `eth_account_strategy.py` |
| **Contribution scoring** | Shapley values quantify each trainer's marginal contribution to model quality — the fair-attribution engine. Centralized or decentralized. | `strategies/contribution/`, `forge/src/contribution/` |
| **Reward distribution** | An ERC-20 token mints rewards proportional to contribution, on-chain, per round. | `contracts/compensation/`, `forge/src/compensation/` |
| **Differential privacy** | Optional Opacus integration adds calibrated noise to gradients; one example adapts each client's privacy budget to their past contribution. | `examples/torch_diff_privacy*`, `ml` dependency group |
| **Provenance / certificates** | Models and TLS certificates can be stored/attested on-chain. | `swarm/certificate/`, `CertificateRegistry` |
| **Metrics & experiment tracking** | Per-round and per-client metrics to local disk or MLflow; best-model tracking. | `logging/` |
| **CLI & key management** | `rzmnd` CLI to generate/deploy swarms and manage an encrypted (AES-256-GCM/Scrypt) mnemonic keystore. | `cli/`, `mnemonic/` |

**Two operating modes** ship today:

- **Centralized Shapley** — the aggregator holds a validation set and scores each
  coalition itself. Simpler; the server is trusted to evaluate honestly.
- **Decentralized Shapley** — evaluation work is assigned to clients on-chain and
  results are recorded on-chain. Removes the server as sole evaluator; this is the
  more novel, more complete architecture (the `workflow` example).

---

## 3. Architecture

Rizemind is a **two-plane system**. The off-chain plane does the machine learning;
the on-chain plane is the source of truth for trust, state, scoring and money.

```
┌──────────────────────────────── OFF-CHAIN (Python / Flower) ────────────────────────────────┐
│                                                                                              │
│   ServerApp (Aggregator)                                   ClientApp (Trainer / Evaluator)   │
│   ┌──────────────────────────────────────┐                ┌───────────────────────────────┐ │
│   │ RizemindWorkflow                      │                │ NumPyClient (local train/eval)│ │
│   │  └─ AggregatorLifecycle (phase graph) │                │  wrapped by mods:             │ │
│   │      Idle → Train → EvalReg → Eval    │                │   • authentication_mod        │ │
│   │ Strategy stack (decorators):          │   Flower gRPC  │   • model_notary_mod          │ │
│   │   MetricStorageStrategy               │◄──────────────►│   • register_contribution_mod │ │
│   │    └─ (Decentral)ShapleyValueStrategy │   weights +     │   • mlflow_mod                │ │
│   │        └─ EthAccountStrategy          │   signatures    │ RizemindClient (async chain   │ │
│   │            └─ FedAvg (base)            │                │   phase watcher/indexer)      │ │
│   └──────────────────┬───────────────────┘                └───────────────┬───────────────┘ │
│                      │  web3.py (contract wrappers, EIP-712, revert decoding)                │
└──────────────────────┼───────────────────────────────────────────────────┼─────────────────┘
                       │                                                     │
┌──────────────────────┼──────────────── ON-CHAIN (Solidity / EVM) ─────────┼─────────────────┐
│                      ▼                                                     ▼                 │
│   SwarmV1 (one proxy per project) — orchestrator + phase state machine                       │
│   ├─ AccessControl ...... who can train / evaluate / aggregate (roles)                       │
│   ├─ Selector (sampling) . which participants are chosen this round                          │
│   ├─ ContributionCalc .... on-chain Shapley scoring from evaluation results                  │
│   ├─ Compensation ........ ERC-20 that mints rewards ∝ contribution                          │
│   ├─ Registries .......... per-round trainers, evaluators, certificates                      │
│   └─ TaskAssignment ...... routes evaluation tasks to evaluators                             │
│   Deployed by SwarmV1Factory (CREATE2) which wires 5 pluggable sub-contracts per swarm.      │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Off-chain plane (Python, ~10.7k LOC)

The design is **decorator-heavy**: every Rizemind `Strategy` wraps an inner Flower
strategy and layers behavior on top. A typical production stack (from the `workflow`
example) is:

```
MetricStorageStrategy( DecentralShapleyValueStrategy( … EthAccountStrategy( FedAvg() ) ) )
```

- **`EthAccountStrategy`** (`authentication/eth_account_strategy.py`) — the trust
  gate. Challenges each client to sign a per-round nonce, recovers the address,
  checks `canTrain`/`canEvaluate` on-chain, signs the global model so clients can
  verify the aggregator, and rejects unauthorized updates.
- **`ShapleyValueStrategy`** and its `Central`/`Decentral` subclasses
  (`strategies/contribution/shapley/`) — forms coalitions (subsets of trainers),
  aggregates each, evaluates each, and computes each trainer's Shapley value.
- **`RizemindWorkflow` + `AggregatorLifecycle`** (`workflow/`, `swarm/lifecycle/`)
  — replaces Flower's fixed "N rounds" loop with a **phase graph that follows the
  on-chain state machine**: Idle → Training → EvaluatorRegistration → Evaluation →
  reset. Each phase's `can_execute()` reads the current on-chain phase.
- **Client mods** (Flower middleware) — `authentication_mod` (answers the auth
  challenge), `model_notary_mod` (verifies the aggregator, signs its own update),
  `register_contribution_mod` (records its contribution on-chain), `mlflow_mod`
  (logging).
- **`Swarm` facade** (`swarm/swarm.py`) — one Python object that resolves a swarm
  address into ~13 typed contract wrappers and exposes `can_train`, `distribute`,
  `start_training_round`, phase transitions, etc.
- **web3 layer** (`web3/`) — connection management, POA/error-decoding middleware,
  and an async event **indexer** (block watcher → log poller → topic-indexed event
  bus) that lets clients react to on-chain phase changes.

### 3.2 On-chain plane (Solidity, ~3.4k LOC, Foundry)

One **`SwarmV1`** proxy per federated-learning project is the entry point. It is a
state machine (idle → training → evaluator-registration → evaluation) that delegates
five concerns to pluggable satellite contracts, each resolved through an interface +
ERC-165 support check:

| Concern | Interface | Default implementation |
|---|---|---|
| Trainer / evaluator selection | `ISelector` | `RandomSampling` / `AlwaysSampled` |
| Contribution scoring | `IContributionCalculator` | `ContributionCalculator` (Shapley) |
| Roles / whitelist | `IAccessControl` | `BaseAccessControl` |
| Reward token | `ICompensation` | `SimpleMintCompensation` (ERC-20) |

**Deployment** uses a factory tree: `SwarmV1Factory.createSwarm()` deterministically
deploys the swarm proxy (CREATE2), then calls four sub-factories to spin up the
selectors, calculator, access-control and compensation contracts, and wires them
together in one transaction. Address pre-computation lets the not-yet-deployed swarm
be granted the roles it needs on its own calculator and token.

Standards in play: **ERC-20** (rewards), **EIP-712 / ERC-5267** (domain-versioned
signing; the EIP-712 domain is also the factory's implementation-registry key),
**ERC-1967** (proxies), **ERC-165** (interface probing), OpenZeppelin
AccessControl/UUPS.

---

## 4. Runtime — one training round, end to end

1. **Round start.** The aggregator lifecycle sees the swarm is `IDLE` and calls
   `startTrainingRound()` on-chain (`onlyAggregator`), which increments the round
   and enters the `TRAINING` phase.
2. **Authentication.** The server sends every client a `GET_PROPERTIES` message
   containing a fresh 32-byte nonce. Each client's `authentication_mod` verifies the
   swarm's EIP-712 domain matches its own config, signs `(round, nonce)`, and
   returns the signature. The server recovers the address and **tags** the client.
3. **Authorization + notarization.** The server filters clients through
   `CanTrainCriterion` (on-chain `canTrain`), then signs the global model
   parameters (acting as notary) and attaches the signature so clients can verify
   the sender.
4. **Local training.** Each client's `model_notary_mod` checks `isAggregator(signer)`
   on-chain before training, trains locally on its private data, then signs its
   updated weights into the response metrics. `register_contribution_mod` records
   the contribution on-chain (`registerRoundContribution`).
5. **Aggregation.** The server recovers each update's signer, cross-checks it against
   the auth-tagged address and the model hash, re-checks `canTrain`, and aggregates
   only whitelisted updates (base `FedAvg`). Unauthorized updates become failures.
6. **Coalition formation & evaluation.** The Shapley strategy forms coalitions
   (subsets of trainers), aggregates each one, and evaluates them — either on the
   server (centralized) or by assigning evaluation tasks to clients on-chain and
   collecting their results (decentralized).
7. **Scoring & payout.** `close_round()` computes each trainer's Shapley value,
   clamps negatives to zero (logging "free rider detected" at zero), and calls
   `swarm.distribute()` — which mints ERC-20 rewards proportional to contribution.
8. **Advance.** Phases auto-advance by wall-clock TTL; the lifecycle resets and the
   next round begins when the swarm returns to `IDLE`.

**A subtle but important property:** the Python lifecycle *follows* the smart-contract
state machine rather than driving it. On-chain phase transitions are time-driven
(TTLs), so the two planes stay loosely synchronized via polling.

---

## 5. The contribution engine — Shapley values

This is the intellectual core and the main differentiator, so it warrants detail.

**The problem it solves:** in collaborative ML, how do you fairly reward each
participant when their data is private and their individual effect on the final model
is entangled with everyone else's? Naive "everyone gets an equal share" invites free
riders; "reward by data volume" is gameable.

**The approach:** the *Shapley value* from cooperative game theory — the unique fair
way to split value among players. For each trainer, measure their **marginal
contribution** (how much model quality improves when they join a coalition) averaged
over all possible coalitions, weighted by the classic `|S|! · (n−|S|−1)!` factor.

**How it is implemented:**

- **Coalitions** are subsets of trainers. Each is a bitmask over participants
  (`ParticipantMapping`), the *same representation used on-chain*, so server and
  chain agree on which set is which.
- **Sampling** decides which coalitions to evaluate. `AllSets` enumerates the full
  power set (exact Shapley, exponential — fine for small swarms); `RandomDeterministic`
  samples coalitions deterministically from an **on-chain source** so the server and
  all evaluators agree on the sample without coordination (scales to large swarms).
- **Scoring** normalizes by the *used* weight total rather than `n!`, which makes it
  robust when only a sampled subset of coalitions is evaluated.
- **Decentralized mode** routes coalition-evaluation tasks to clients via an on-chain
  `TaskAssignment` contract (affine assignment, zero per-pair storage) and records
  results on-chain in `EvaluationStorage`, so contribution scoring does not depend on
  a single trusted evaluator.

A duplicate on-chain implementation (`ShapleyValueCalculator.sol`) computes the same
value in fixed-point arithmetic for trustless, verifiable scoring.

---

## 6. Trust & security model

**Identity = key.** A participant is whoever controls an Ethereum private key. There
are no passwords and no raw data on the wire — only signed model weights.

**Every interaction is signed and domain-bound.** All signatures are EIP-712,
bound to a specific `(verifyingContract, chainId)` fetched on-chain via ERC-5267, so
a signature for one swarm/chain cannot be replayed against another. A fresh per-round
nonce blocks replay within a swarm. Bad/absent signatures fail *closed* (the
participant is rejected, not crashed).

**On-chain access control is the root of trust.** The `BaseAccessControl` contract
decides roles. The **aggregator role is the dominant trust anchor**: an aggregator can
add trainers, evaluators, and *other aggregators*. Compromise of one aggregator key
compromises the swarm's membership.

**Mutual verification** closes the loop: clients verify the aggregator signed the
global model (`isAggregator`) before training; the server verifies clients are
authorized (`canTrain`) before aggregating.

This is a solid, well-constructed trust design. The **gaps are in the details** —
see §7.

---

## 7. Production-readiness assessment

### Engineering maturity already present

- Clean separation: framework-agnostic library core; PyTorch only in an optional
  dependency group.
- Real test coverage on both planes: 19 Solidity test files, ~31 Python unit test
  files, and on-chain integration tests that deploy the actual contracts against a
  local Anvil node using the same mechanism the examples use.
- CI runs unit tests on every push, integration + Solidity tests on PRs, lint,
  versioned doc deploys, and OIDC "trusted publisher" PyPI releases.
- Static analysis (Aderyn) is *used* — contracts carry triaged `aderyn-ignore`
  annotations — and OpenZeppelin 5.2 is the contract base.
- Thoughtful crypto: AES-256-GCM + Scrypt keystore; EIP-712 domain binding.

### Gaps to close before production

Severity: 🔴 critical · 🟠 high · 🟡 medium.

| # | Sev | Gap | Detail |
|---|:--:|---|---|
| 1 | 🔴 | **Reward over-mint on negative score** | `SwarmV1.claimReward` casts a signed Shapley `int256` to `uint64` with no `≤ 0` guard. A negative contribution becomes a huge positive mint. Correctness/economic bug — must fix + add invariant tests. |
| 2 | 🔴 | **Predictable randomness (no VRF)** | All seeds are `keccak256(address(this), roundId)`. Participant sampling and coalition/task selection are fully predictable in advance; a participant can know if they'll be selected. `// TODO: Use VRF` is acknowledged but unimplemented. |
| 3 | 🔴 | **No external security audit** | No third-party smart-contract audit. Aderyn runs manually, not in CI. Contracts hold funds (mint rewards) → an audit is table stakes. |
| 4 | 🟠 | **Unprotected `TaskAssignment.setConfig`** | Inherited onto the swarm with no access modifier; anyone can overwrite a round's evaluation-task config and grief/redirect evaluation authorization. |
| 5 | 🟠 | **Deployed swarms are not upgradeable** | `SwarmV1` runs behind a proxy but has no UUPS upgrade path. A bug in a live swarm cannot be patched in place — only mitigated by deploying a fresh swarm and migrating. Decide the upgrade strategy deliberately. |
| 6 | 🟠 | **Differential privacy is not production-grade** | Both DP examples use Opacus `secure_mode=False` (non-cryptographic RNG) and globally suppress warnings. The privacy guarantee is illustrative, not real. |
| 7 | 🟠 | **Hardening gaps on logic contracts** | `SwarmV1` and `SimpleMintCompensation` don't call `_disableInitializers()`; evaluation-result averaging (`_mergeResults`) is manipulable by a duplicate submitter. |
| 8 | 🟠 | **Secrets committed in examples** | Anvil test mnemonic/private key committed across examples; the testnet example commits `passphrase = "secret"`. No key-management story for real deployments. |
| 9 | 🟡 | **Examples undocumented / untested** | All six example READMEs are empty; the compatibility table lists non-existent examples and omits the real `workflow` one. No example is exercised in CI. This is the primary onboarding surface. |
| 10 | 🟡 | **No type-checking, coverage, or Solidity scanner in CI** | Heavy `cast(...)` use with no mypy/pyright gate; no coverage thresholds; Aderyn/Slither not enforced. Integration tests run only on PRs. |
| 11 | 🟡 | **Known correctness/operability TODOs** | Reward-rounding TODO; loss/selected-model mismatch documented in the Shapley strategy; busy-wait polling loops in the lifecycle; a latent event-bus attribute-name bug in the indexer; lifecycle stop-signal not propagated. |
| 12 | 🟡 | **Reorg / finality handling** | The chain indexer checkpoints block height only, with no reorg/rollback handling and a defined-but-unused `CONFIRMATIONS` constant. |

---

## 8. Recommended roadmap (the work to approve)

Phased so that value and de-risking come early. Rough sizing is indicative and to be
confirmed by the team.

### Phase 1 — Correctness & security hardening (highest priority)
- Fix the reward-minting cast bug (#1) and add economic-invariant tests.
- Lock down `TaskAssignment.setConfig` and other access-control gaps (#4, #7).
- Add `_disableInitializers()` and remove debug artifacts (leftover `console` import).
- Wire Aderyn (and ideally Slither) into CI; add mypy/pyright and coverage gates
  (#3, #10).
- **Commission a third-party smart-contract security audit** (#3) — start this early
  since it gates any mainnet/value-bearing deployment.

### Phase 2 — Trust-critical mechanisms
- Integrate **verifiable randomness (VRF)** for sampling and coalition/task selection
  (#2) — the single most important cryptographic gap.
- Decide and implement the **upgrade strategy** for deployed swarms (#5).
- Harden differential privacy: `secure_mode=True`, formal budget accounting, stop
  suppressing warnings (#6).
- Robust evaluation-result handling (per-evaluator dedup, tamper resistance) (#7).

### Phase 3 — Operability & onboarding
- Real key management for aggregators and clients; remove committed secrets;
  document the keystore/HSM story (#8).
- Complete example documentation and add at least one example to CI as an
  end-to-end smoke test (#9).
- Indexer reorg/finality handling and confirmation depth (#12).
- Monitoring/observability for the aggregator lifecycle and on-chain state;
  resolve the busy-wait and stop-signal TODOs (#11).

### Phase 4 — Scale & product readiness
- Performance and scale testing of decentralized Shapley with realistic swarm sizes.
- Gas-cost analysis and optimization for on-chain scoring/rewards.
- Threat model review of the aggregator-as-trust-anchor design; consider
  multi-sig / decentralized aggregation if the product requires it.

---

## 9. Deployment architecture

How Rizemind is intended to run in a real, multi-organization deployment — as
opposed to the single-process Flower *simulation* the examples use by default.

### 9.1 Topology

A production federation has three kinds of participants, each in its own trust and
network boundary:

- **Aggregator** — runs the Flower `ServerApp` (the `EthAccountStrategy` +
  Shapley + `RizemindWorkflow` stack) and holds the aggregator Ethereum key. It
  deploys/owns the swarm contracts and drives the round lifecycle. One per
  federation.
- **Trainer orchestration node** — a *small, always-on* node inside each trainer's
  infrastructure. It runs the Flower `ClientApp`, holds that trainer's Ethereum key,
  and speaks to the aggregator. **It does not hold the data or the GPUs.**
- **Trainer ETL / compute plane** — the trainer's *existing* infrastructure: their
  ETL that produces training datasets, and the compute (K8s, spot nodes, a GPU box)
  that runs the actual training. The orchestration node **delegates** training to it.

The **blockchain** (a local chain, the Rizenet testnet, or any EVM network) sits
underneath all three as the shared source of truth for identity, round state,
contribution scores, and rewards.

### 9.2 Delegated training — keep data and compute on the trainer's side

The key production pattern (implemented in `examples/prefect_delegation/`, using
Prefect as the workflow engine) is that **a data owner never ships raw data and
never exposes their compute to the federation**. Instead:

1. The trainer's **ETL exports a training dataset to a bucket** in a format
   compatible with the containerized training step. That bucket's access is scoped
   so it is **readable only by the training step**.
2. When the aggregator calls the trainer for a round, the orchestration node
   publishes the global model weights to a second scoped bucket and **triggers a
   training workflow** (a Prefect flow) on the trainer's own ETL/compute plane.
3. The workflow **provisions the appropriate resources** and runs the
   **containerized training step**, which can access *only* the training artifacts
   (dataset + input weights) and writes the resulting **model weights to a bucket
   readable by the trainer**. The container holds no swarm key and no chain access —
   least privilege.
4. The orchestration node **waits for the workflow to complete**, reads the produced
   model artifact back, and **shares it with the aggregator** — where Rizemind's
   notary mod signs the update and the normal authorization/aggregation applies.

```
Aggregator ──global weights──► Trainer node ──put──► [weights-in bucket] ──┐
     ▲                              │                                      │ read-only
     │                              │ trigger flow (Prefect)               ▼
     │                     ┌────────┴─────────┐                    Training container
     │                     │  Trainer ETL /   │  ETL export ──► [dataset bucket] ──►(reads)
     │                     │  compute plane   │                                      │
  signed weights ◄──read── │  (their infra)   │ ◄──[weights-out bucket]◄──(writes)───┘
                           └──────────────────┘        (trainer-readable)
```

**Access scoping** is enforced by the object store's IAM (three buckets/prefixes
with different policies) plus short-lived, tightly-scoped credentials handed to the
container at run time. The example ships a swappable `ArtifactStore` abstraction
(local filesystem for the demo; S3/GCS/MinIO in production).

**Additional security step — training-image attestation.** *(Inferred: the request
describing this was cut off — please confirm.)* The workflow refuses to run any
container whose **image digest** is not the approved one. The sanctioned digest is
published by the aggregator/DAO — for example registered on-chain via the swarm's
`CertificateRegistry` — so a trainer cannot silently substitute a tampered training
step, and the aggregator can audit which image produced each update.

### 9.3 Experiment tracking with MLflow

Rizemind ships first-class MLflow support, used on both sides of the delegation:

- **Aggregator** wraps its strategy in `MetricStorageStrategy(strategy,
  MLFLowMetricStorage(...))` (`logging/mlflow/metric_storage.py`) to log per-round
  aggregated metrics and to save the best global model as an MLflow artifact.
- **Trainer / training step** logs local training curves (loss, accuracy, epsilon
  for DP, training time) to the same MLflow tracking server, either via the
  `mlflow_mod` client mod or directly from the containerized step.

Point every participant at one MLflow tracking server (`mlflow-uri`) to get a single
federated view of the run. The `prefect_delegation` example logs both the
aggregator run and each trainer's delegated step.

### 9.4 What "deploy" concretely involves

| Step | Action |
|---|---|
| **Chain** | Choose an EVM network; deploy the `SwarmV1Factory` (or reuse the Rizenet testnet factory); whitelist the aggregator for contract deployment. |
| **Swarm** | Aggregator deploys a swarm via the factory (access control, sampling, calculator, compensation, phase TTLs) — one transaction. |
| **Keys** | Each participant generates/loads an Ethereum account (encrypted `rzmnd` keystore, or an HSM/KMS in production). Aggregator whitelists trainers/evaluators on-chain. |
| **Aggregator** | Run the `ServerApp` (Flower deployment engine, not simulation) pointed at the chain + MLflow. |
| **Trainers** | Run the orchestration node (`ClientApp`) on a small node; register a Prefect deployment + work pool on the ETL/compute plane; build and pin the training-step image digest. |
| **Buckets** | Provision the three scoped buckets/prefixes; wire the ETL export schedule; configure the `ArtifactStore` backend. |
| **Observe** | One MLflow tracking server for all participants; monitor the aggregator lifecycle and on-chain phase state. |

> Note: the delegation example is **reference architecture**. Its storage,
> flow-trigger, and container-runtime seams are marked in the code and README as the
> three things to replace with real infrastructure. Production hardening of the
> broader system still depends on the items in §7–§8 (audit, VRF, key management).

## 10. How to run it (for the evaluating engineer)

```bash
# Install (Python 3.12, uv package manager)
uv sync --group ml

# Local blockchain examples: start a chain and deploy contracts
cd forge && forge soldeer install && forge build
anvil                       # local EVM on 127.0.0.1:8545
forge script script/deployments/SwarmV1Factory.s.sol \
  --rpc-url http://127.0.0.1:8545 --private-key <anvil-key> --broadcast

# Run an example (Flower simulation)
cd examples/torch_shapley && uv run -- flwr run .

# Delegated-training example (trainer offloads compute to Prefect + MLflow)
cd examples/prefect_delegation && uv run -- flwr run .
#   optional live metrics:  uv run -- mlflow ui --backend-store-uri ./mlruns

# Tests
uv run pytest tests/unit           # in-memory, no chain
uv run pytest tests/integration    # requires Anvil + Foundry
cd forge && forge test -vvv        # Solidity
```

Recommended reading order in the code: `swarm/swarm.py` → `swarm/lifecycle/` →
`authentication/eth_account_strategy.py` → `strategies/contribution/shapley/` →
`forge/src/swarm/SwarmV1.sol` → `forge/src/contribution/`.

---

*Prepared from a full read of the codebase (Python orchestration, authentication/web3,
Solidity contracts, and examples/tests/CI). File-level references are available in the
supporting notes for any section on request.*
