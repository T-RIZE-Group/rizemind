.. rizemind documentation master file, created by
   sphinx-quickstart on Wed Feb 19 10:00:38 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=================
Meet Rizemind
=================

.. toctree::
   :maxdepth: 2
   :caption: Sections

   install
   quickstarts/index
   how-tos/index
   references/index
   developers/index

**Rizemind** is a cooperative, privacy‑preserving framework developed by **T‑RIZE** and **T‑RIZE Labs** a Canadian industrial research chair. Built on **Federated Learning (FL)**, Rizemind uses **distributed ledgers** to strengthen coordination, robustness, and security across untrusted participants.

* Participants **do not share raw data**. They train **on‑premise** and share **model updates/outputs** only.
* Local training extracts generalizable knowledge that is **aggregated** into a collective **“supermodel.”**
* The framework provides **transparent, verifiable contribution scoring**, which powers an **incentives module** to align collaborators.

This decentralized approach protects sensitive information while improving model accuracy via broad data diversity. By lowering data‑sharing barriers—even among competitors—Rizemind unlocks collaboration where assessments are typically siloed.

Flower × T‑RIZE
===============

`Flower <https://flower.ai/>`_ is a widely adopted open‑source framework for federated AI across research and production. It offers a unified approach to federated learning, analytics, and evaluation with an excellent developer experience.

**Rizemind** combines T‑RIZE’s applied expertise in distributed ledgers and AI with the research capacity of **T‑RIZE Labs (Prof. Kaiwen Zhang)** to bring Flower to the next level in **decentralized coordination**. Rizemind is designed as a **complementary Flower library**, easing the transition from centralized FL orchestration to decentralized setups with **minimal code changes**.

Why cooperation?
================

The concept of cooperation between multiple data owners is attractive because it enables:

* **Collective intelligence:** co‑training models across organizations.
* **Geographically distributed compute:** better latency/cost profiles and resilience.
* **Heterogeneous data integration:** reduce biases via broader coverage.
* **Breaking down silos:** unlock previously infeasible use cases.
* **Collective model ownership:** align incentives to maintain and improve models.
* **New data monetization:** attribute and reward valuable contributions.

There’s an adage that fits well here: *the whole is greater than the sum of its parts*.

Design principles
=================

To enable cooperation among partially trusted parties, Rizemind embraces the following principles:

**Neutrality**

Collaboration among untrusted participants requires a **neutral coordination layer**. Rizemind leverages **blockchains** to provide a permissionless environment for training coordination with high availability, scale, and verifiability.

**Auditability**

A distributed ledger records **training metadata** (e.g., round progress, participants, artifacts), enabling peers to **cross‑check** the information they receive and independently verify protocol state.

**Accountability**

Whether deterring **model poisoning** or **rewarding productive trainers**, accountability safeguards long‑term system health. Rizemind provides attribution and traceability needed for **policy enforcement** and **incentive payouts**.

**Robustness**

Distributed systems must tolerate **faulty** or **Byzantine** nodes and intermittent networks. Drawing on blockchain patterns and FL research, Rizemind emphasizes **availability** and **adversarial resilience**.

How it works
==========================

1. **Register & discover** participants and training jobs on a neutral ledger.
2. **Initialize** a task (model, rounds, metrics, incentives, policies).
3. **Train locally** at each participant; **share updates** (not raw data).
4. **Validate & aggregate** contributions (secure aggregation and/or robust aggregation strategies).
5. **Score contributions** with transparent metrics tied to incentives.
6. **Settle incentives** and **publish artifacts/metadata** to the ledger for auditability.

Key capabilities
================

* **Ledger‑backed coordination:** verifiable state, replayability, and accountability.
* **Contribution accounting:** clear, transparent contribution scores for incentives.
* **Minimal adoption friction:** Flower‑native ergonomics; plug‑in orchestration.
* **Privacy‑preserving defaults:** on‑prem training; configurable sharing policies.
* **Enterprise‑ready controls:** governance hooks, policy enforcement, and audit trails.


Get involved
============

* **Source code:** `GitHub <https://github.com/t‑rize-group/rizemind>`_
* **Community:** Join our `Slack Channel <https://join.slack.com/t/rizemind/shared_invite/zt-3dufpugzb-znhIxQcO8sCAKY6V6JrhCg>`_
* **Security:** security\@t‑rize.io