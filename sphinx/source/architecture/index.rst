============
Architecture
============

This page is an initial, developer-facing overview of the Rizemind
architecture. Rizemind is a complementary `Flower <https://flower.ai/>`_
library that moves federated-learning (FL) coordination onto a distributed
ledger, so that partially trusted participants can train a shared model
while keeping coordination state **verifiable, auditable, and accountable**.

The system is split in two halves that mirror this split of concerns:

* **Off-ledger components** (Python, ``src/py/rizemind``) run the actual
  federated learning on top of Flower: local training, secure model-update
  exchange, aggregation, contribution scoring, and the orchestration glue.
* **On-ledger components** (Solidity, ``forge/src``) hold the *coordination
  state of record*: who may participate, which round is active, what each
  participant submitted, how contributions scored, and how rewards were paid.

.. contents::
   :local:
   :depth: 2

High-level topology
===================

.. code-block:: text

   ┌────────────────────────── Off-ledger (Python / Flower) ──────────────────────────┐
   │                                                                                    │
   │   Trainer nodes            Aggregator (ServerApp)              Evaluator nodes      │
   │   ┌───────────┐            ┌────────────────────────┐         ┌───────────┐         │
   │   │ ClientApp │            │ RizemindWorkflow       │         │ ClientApp │         │
   │   │ local fit │◀──Flower──▶│  ├─ EthAccountStrategy │◀─Flower▶│ evaluate  │         │
   │   │ EIP-712   │  Grid/gRPC │  ├─ Shapley strategy   │         │           │         │
   │   │ signing   │            │  ├─ Compensation strat │         └───────────┘         │
   │   └───────────┘            │  └─ SwarmIndexer       │                               │
   │                            └───────────┬────────────┘                               │
   │                                        │ web3 (read state / send txs / index logs)  │
   └────────────────────────────────────────┼───────────────────────────────────────────┘
                                            │
   ┌─────────────────────────── On-ledger (EVM / Solidity) ─┼───────────────────────────┐
   │                                  SwarmV1 (ERC-1967 proxy, the swarm entry point)     │
   │   ┌──────────────┬──────────────┬─────────────────┬──────────────┬───────────────┐   │
   │   │ Access       │ Selectors    │ Training phases │ Registries   │ Contribution  │   │
   │   │ control      │ (sampling)   │ + scheduling    │ (trainer/    │ calculator    │   │
   │   │ (roles)      │              │ (TaskAssignment)│ evaluator/   │ (Shapley) +   │   │
   │   │              │              │                 │ certificate) │ Compensation  │   │
   │   └──────────────┴──────────────┴─────────────────┴──────────────┴───────────────┘   │
   └──────────────────────────────────────────────────────────────────────────────────────┘

On-ledger components
====================

The on-ledger side is a set of EVM smart contracts under ``forge/src`` built
with `Foundry <https://book.getfoundry.sh/>`_ and OpenZeppelin 5.2. They use
the **namespaced-storage / upgradeable** pattern (ERC-1967 proxies, UUPS where
applicable) and advertise their capabilities through **ERC-165** interface
detection and **ERC-5267** EIP-712 domains.

``SwarmV1`` is the entry point for a single swarm (one federated-learning
project). It is a proxy that composes the coordination modules through
inheritance:

* ``RoundTraining`` / ``BaseTrainingPhases`` — round and phase lifecycle.
* ``CertificateRegistry`` — model/identity certificate storage.
* ``RoundTrainerRegistry`` / ``RoundEvaluatorRegistry`` — per-round
  participant bookkeeping.
* ``TaskAssignment`` — deterministic distribution of evaluation tasks.
* ``SwarmCore`` — pointers to the pluggable strategy contracts
  (trainer selector, evaluator selector, contribution calculator, access
  control, compensation).

``SwarmCore`` is the configuration hub: it stores the addresses of the five
pluggable modules and validates (via ERC-165) that each address actually
implements the expected interface before accepting it.

Smart contracts
===============

The contracts are organized by responsibility:

Swarm coordination (``forge/src/swarm``)
----------------------------------------

* ``SwarmV1`` — orchestrates a training round end to end. The documented
  round flow is:

  1. The **aggregator** calls ``startTrainingRound()``.
  2. **Trainers** call ``registerRoundContribution()`` to record their model
     update (by hash) for the round.
  3. **Evaluators** call ``registerForRoundEvaluations()`` to register.
  4. Evaluators call ``registerEvaluation()`` to submit evaluation results.
  5. The aggregator calls ``nextRound()`` to close the round.
  6. Trainers call ``claimReward()`` to mint/collect their compensation.

* ``SwarmV1Factory`` — deploys new swarms. It wires together the per-swarm
  modules produced by the sub-factories and deploys the swarm behind an
  ERC-1967 proxy using ``CREATE2`` for deterministic addresses.
* ``registry/SwarmCore`` — module address registry described above.
* ``registry/RoundTrainerRegistry`` — per-round trainer records: assigned
  trainer ``id``, submitted ``modelHash``, and a ``rewardsClaimed`` flag.
* ``registry/RoundEvaluatorRegistry`` — per-round evaluator records and IDs.
* ``registry/CertificateRegistry`` — generic ``bytes32 id → bytes``
  certificate store, emitting ``NewCertificate``.

Access control (``forge/src/access``)
--------------------------------------

* ``IAccessControl`` — the permissioning interface (``isTrainer``,
  ``isAggregator``, ``isEvaluator``), itself an ERC-5267 domain.
* ``BaseAccessControl`` — the default implementation, built on OpenZeppelin
  ``AccessControlUpgradeable`` with three roles (``AGGREGATOR_ROLE``,
  ``TRAINER_ROLE``, ``EVALUATOR_ROLE``). The aggregator administers the
  trainer/evaluator sets.
* ``FLDemocraticWhitelist`` — an alternative, vote-based whitelist where
  participants reach a configurable approval threshold (governance variant,
  see below).
* ``AccessControlFactory`` — deploys access-control instances.

Sampling / selection (``forge/src/sampling``)
---------------------------------------------

* ``ISelector`` — ``isSelected(addr, roundId)`` interface used to decide who
  participates in a given round.
* ``AlwaysSampled`` — selects everyone (useful for small/dev swarms).
* ``RandomSampling`` — probabilistic selection against a target ratio, driven
  by on-chain randomness (``RNG`` / seed providers in ``forge/src/randomness``).
* ``SelectorFactory`` — deploys selector instances.

Contribution & compensation (``forge/src/contribution``, ``forge/src/compensation``)
------------------------------------------------------------------------------------

* ``IContributionCalculator`` / ``ContributionCalculator`` —
  records evaluation results and computes contributions. The default math is
  ``ShapleyValueCalculator`` plus ``EvaluationStorage``; the calculator decides
  how many evaluation tasks a round needs (``getEvaluationsRequired``) and
  exposes ``calculateContribution(...)``. It is UUPS-upgradeable.
* ``ICompensation`` / ``SimpleMintCompensation`` — distributes rewards. The
  default implementation is an ``ERC20Upgradeable`` token that **mints**
  rewards proportional to contribution scores via ``distribute(roundId,
  recipients, contributions)``, gated by a ``MINTER_ROLE``.
* ``CalculatorFactory`` / ``CompensationFactory`` — deploy the respective
  instances.

Training lifecycle & scheduling (``forge/src/training``, ``forge/src/scheduling``)
----------------------------------------------------------------------------------

* ``BaseTrainingPhases`` — a time-based state machine with four phases:
  ``IDLE`` → ``TRAINING`` → ``EVALUATOR_REGISTRATION`` → ``EVALUATION``,
  transitioning automatically once each phase's time-to-live (TTL) expires.
  ``updatePhase()`` advances the machine and emits ``PhaseTransition`` events.
* ``RoundTraining`` — round counter and per-round summary state.
* ``TaskAssignment`` — a zero-per-pair-storage affine scheme that assigns
  ``T`` evaluation tasks across ``N`` evaluators with ``R`` tasks each,
  computed on the fly from a small per-round ``Config``.

Off-ledger components
=====================

The Python package ``rizemind`` (under ``src/py/rizemind``) is the runtime
that participants actually execute. Key subsystems:

* ``swarm/`` — the ``Swarm`` facade wraps a deployed ``SwarmV1`` and all its
  modules behind a single Python object (``access_control``, ``training``,
  ``contribution``, ``compensation``, ``certificates``, registries,
  ``task_assignement``), backed by a ``web3`` connection and an optional
  signing account.
* ``swarm/lifecycle/`` and ``workflow/`` — the aggregator-side orchestration
  (``RizemindWorkflow``, ``AggregatorLifecycle``) that drives Flower rounds in
  step with the on-chain phase machine.
* ``swarm/indexer/`` and ``web3/indexer/`` — log/block indexers
  (``SwarmIndexer``, ``PhaseWatcher``, event buses) that watch chain state
  (e.g. phase transitions, ``TrainerContributed`` events) and surface them to
  the runtime.
* ``authentication/`` — Web3 identity and EIP-712 signing (see *Permissioning*
  and *Model registry*).
* ``strategies/`` — Flower ``Strategy`` decorators for contribution scoring
  (Shapley, centralized and decentralized variants) and compensation.
* ``contracts/`` — typed Python wrappers + ABIs for every on-chain contract.
* ``configuration/`` — ``pyproject.toml``-driven configuration
  (``TomlConfig``, Web3 / account / swarm configs).
* ``logging/`` — pluggable metric storage (local disk and MLflow backends).
* ``mnemonic/`` and ``cli/`` — key/account management and CLI helpers.

Flower integration
==================

Rizemind is designed to drop into an existing Flower app with minimal code
changes. The integration happens at three Flower extension points:

* **Server workflow.** ``RizemindWorkflow`` replaces Flower's default
  workflow. It still uses Flower's ``default_fit_workflow`` /
  ``default_evaluate_workflow`` internally, but wraps them in an
  ``AggregatorLifecycle`` that keeps each Flower round aligned with the swarm's
  on-chain phases.
* **Strategy decorators.** Rizemind strategies wrap any base Flower
  ``Strategy`` (e.g. ``FedAvg``):

  * ``EthAccountStrategy`` enforces authentication — it verifies each client's
    EIP-712 signature against the on-chain access control before accepting a
    ``FitRes``, and drops unauthorized clients into ``failures``.
  * The Shapley strategies (``strategies/contribution/shapley/...``) form
    coalitions of submitted updates and score each trainer's marginal
    contribution.
  * ``SimpleCompensationStrategy`` distributes rewards after aggregation.

* **Client side.** Trainer/evaluator ``ClientApp`` s answer Flower
  ``GetProperties`` authentication challenges (``train_auth``) and sign their
  model parameters; an authorized client manager
  (``ClientManagerWithCriterion`` with ``CanTrainCriterion`` /
  ``CanEvaluateCriterion``) filters participation.

See ``examples/workflow`` for a complete end-to-end server/client setup.

Model registry
=============

Rizemind does not store model *weights* on chain; it stores **commitments**
to them. Each model update is reduced to a keccak256 hash of its tensors
(``authentication/notary/model/model_signature.hash_parameters``) and signed
by the trainer's Ethereum account under an EIP-712 ``Model`` type
(``round``, ``hash``). The "notary" modules produce and verify these
signatures.

On chain, ``RoundTrainerRegistry`` records the ``modelHash`` each trainer
submitted for a round (emitting ``ModelHashUpdated``), and
``CertificateRegistry`` provides a generic certificate store keyed by
``bytes32`` id. Together these give a tamper-evident, replayable record of
*which signer committed which model at which round* — without exposing the
underlying data or weights.

Audit trail
==========

Auditability is a first-class design principle: the ledger is the canonical,
append-only record of coordination, and almost every state change emits an
event. Notable event streams that an auditor (or the ``SwarmIndexer``) can
replay include:

* ``PhaseTransition`` — round/phase progression.
* ``TrainerRegistered`` / ``ModelHashUpdated`` — participation and model
  commitments per round.
* ``EvaluatorRegistered`` and evaluation results — who evaluated what.
* ``ContributionResultRegistered`` / ``TrainerContributed`` — contribution
  scoring inputs and outputs.
* ``CompensationSent`` — reward payouts.
* ``NewCertificate`` — certificate issuance.
* ``SwarmCore`` ``*Updated`` events — changes to the pluggable modules.

Because these are emitted by the contracts (not by an off-ledger service),
any participant can independently verify protocol state and cross-check the
information they receive over Flower against the chain.

Contribution tracking
====================

Contribution accounting is the link between *work done* and *rewards paid*,
and it is split across both halves of the system:

* **Off-ledger (scoring).** The Flower Shapley strategies evaluate coalitions
  of trainer updates and compute each trainer's marginal contribution. A
  ``ContributionCalculator`` interface and sampling strategies
  (``AllSets``, deterministic random sampling) bound the number of coalition
  evaluations.
* **On-ledger (settlement).** The ``ContributionCalculator`` contract records
  evaluation results keyed by ``(roundId, setId, modelHash)`` and, for the
  Shapley case, accepts results within Hamming distance 1 of the target set
  (the "with" and "without player" coalitions the Shapley value needs). The
  ``TaskAssignment`` contract spreads evaluation tasks uniformly across
  registered evaluators. Final contributions feed ``ICompensation.distribute``,
  which mints rewards proportionally.

Permissioning
============

Permissioning is enforced on chain and proven off chain via signatures:

* **Roles.** ``BaseAccessControl`` defines ``AGGREGATOR``, ``TRAINER``, and
  ``EVALUATOR`` roles using OpenZeppelin's role-based access control. The
  aggregator manages the trainer/evaluator sets; ``SwarmV1`` gates sensitive
  operations with ``onlyAggregator`` / ``onlyTrainer`` / ``onlyEvaluator``
  modifiers that consult the configured access-control contract.
* **Per-round selection.** Being whitelisted is necessary but not sufficient
  — an ``ISelector`` (e.g. ``RandomSampling``) decides who is actually
  sampled into a given round.
* **Off-ledger proof.** Over Flower, a client proves it controls a permitted
  address by signing an EIP-712 challenge (nonce + round + domain). The
  aggregator's ``EthAccountStrategy`` recovers the signer and checks it
  against access control before accepting the update, so authorization is
  cryptographic rather than trust-on-first-use.

Deployment model
==============

Deployment is **factory-based and proxy-based**:

* A one-time platform deployment publishes the implementation/logic contracts
  and the sub-factories (``SelectorFactory``, ``CalculatorFactory``,
  ``AccessControlFactory``, ``CompensationFactory``) plus the top-level
  ``SwarmV1Factory``. ``forge/deploy.sh`` performs this in order and is the
  reference for the dependency graph between factories.
* Creating a swarm is then a single ``SwarmV1Factory`` call that selects an
  implementation for each pluggable module (by ``id`` + init data), deploys the
  swarm behind an ERC-1967 proxy via ``CREATE2`` (deterministic address), and
  initializes its phase configuration.
* Contracts are **upgradeable** (UUPS / ERC-1967 + namespaced storage), so
  logic can evolve without migrating state; the factory also exposes proxy
  upgrade paths (``ProxyUpgraded``).
* Off-ledger, participants run standard Flower ``ServerApp`` / ``ClientApp``
  processes configured (via ``pyproject.toml``) with the swarm address and a
  Web3 endpoint. Examples target a local Anvil node and the Rizenet testnet
  (see ``examples/``).

Governance
=========

Governance covers *who controls the swarm's parameters and membership* over
its lifetime. The current building blocks are:

* **Administered model (default).** With ``BaseAccessControl``, the aggregator
  holds the administrative role: it manages membership and, through
  ``SwarmCore``'s ``*Updated`` paths, can repoint pluggable modules (selector,
  calculator, compensation, access control). This is simple but centralized,
  and the relevant factory paths are explicitly annotated as
  centralization-risk in the Solidity source.
* **Democratic membership (alternative).** ``FLDemocraticWhitelist`` replaces
  unilateral admission with a vote: participants approve trainers and a trainer
  becomes whitelisted once it crosses a configurable ``approvalThreshold``.
  This is the first step toward decentralizing membership decisions.
* **Verifiable change log.** Regardless of model, every governance-relevant
  change is an on-chain event (module updates, role grants, threshold votes),
  so policy changes are themselves auditable.

.. note::

   Governance beyond these primitives (e.g. full DAO-style control of upgrades
   and treasury) is a direction rather than a finished subsystem. This section
   should be revisited as those mechanisms land.
