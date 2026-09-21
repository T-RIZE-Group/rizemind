========================
Rizemind on Arc Mainnet
========================

Rizemind is deployed on `Arc <https://www.arc.io>`_, Circle's USDC-gas L1. This
page documents the current deployment and walks through ``examples/arc``, which
runs a federation against it end to end.

.. caution::

   On Arc Mainnet, gas is **real USDC**. Every transaction in this walkthrough
   spends money. Rehearse on Arc Testnet first — the same example runs there
   with one word changed.

Deployment record
=================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Item
     - Value
   * - Network
     - Arc Mainnet, chain ID ``5042``
   * - ``SwarmV1Factory``
     - `0x721a4ebA8747eF5db299E8Eec67A2FbAe7866353 <https://explorer.arc.io/address/0x721a4ebA8747eF5db299E8Eec67A2FbAe7866353>`_

**Deployed components.** Deployed once per chain, by
``forge/script/deployments/``:

* ``SelectorFactory`` — registry and ERC-1967 proxy factory for trainer and
  evaluator selectors.
* ``AlwaysSampled`` — selector implementation, ``always-sampled-v1.0.0``.
* ``RandomSampling`` — selector implementation, ``random-sampling-v1.0.0``.
* ``SwarmV1`` — swarm logic contract; proxies delegate to it.
* ``SwarmV1Factory`` — what the Python library calls. ``createSwarm()`` deploys
  a swarm proxy plus its two selector proxies in one transaction.

Each *federation* then deploys its own ``SwarmV1`` ERC-1967 proxy through the
factory. Only the factory address is needed to use the deployment; it is
registered in the library, so no configuration is required to reach it.

**Supported release.** The chain registrations landed on ``main`` and are not
yet in a tagged release. Use a checkout that contains
``ARC_MAINNET_CHAINID`` in ``rizemind/web3/chains.py``.

**Arc Testnet.** The same components are deployed on chain ID ``5042002`` and
registered alongside mainnet, so the walkthrough below runs unchanged against
either network.

Two Arc properties worth knowing
================================

**USDC has two views.** The *native* view — ``eth_getBalance``, gas prices,
``msg.value`` — uses **18 decimals**, exactly like ether, so web3.py's and
Foundry's wei arithmetic needs no adjustment. The *ERC-20* view (precompile
``0x3600…0000``) uses **6 decimals**. They are the same balance a factor of
10\ :sup:`12` apart; mixing them is the usual explanation for a balance that
looks absurd.

**Deployment is permissionless.** Unlike Rizenet, Arc has no deployer
allowlist, so there is no address to whitelist before you begin.

Walkthrough: ``examples/arc``
=============================

A minimal federation: three trainers, one round, decentralized Shapley value
contribution scoring, CPU-only simulation. It is deliberately small — the
decentralized Shapley strategy evaluates every non-empty coalition, so three
trainers means seven evaluations per round and ten would mean 1023.

Step 1 — Install
----------------

.. code:: shell

   uv sync
   cd examples/arc

Torch and the dataset dependencies are pulled in on first use; no extra sync
flags are needed.

Step 2 — Choose the network
---------------------------

One setting picks the chain, in ``[tool.flwr.app.config]``:

.. code-block:: toml

   arc-network = "mainnet"   # or "testnet", or "local" for Anvil

It selects the chain ID and the default RPC endpoint *together*, so the chain
guard and the endpoint cannot drift apart. Override it for a single run instead
of editing the file:

.. code:: shell

   uv run -- flwr run . --run-config arc-network=testnet

``$ARC_RPC_URL`` overrides only the endpoint, for a private or rate-limited
node. The chain guard still holds it to the network's chain ID.

Step 3 — Create the aggregator account
--------------------------------------

Only the **aggregator** sends transactions. Trainers sign EIP-712 messages
off-chain and never touch the chain, so their accounts need no USDC.

.. code:: shell

   uv run -- rzmnd account generate arc-aggregator --words 24
   export RZMND_PASSPHRASE='the passphrase you just chose'

This writes an encrypted keystore to ``~/.rzmnd/keystore/``. Account 0 of that
mnemonic is the aggregator; accounts 1..N are the trainers.

.. caution::

   Do not use ``rzmnd account load`` to read the addresses back — it prints the
   mnemonic *and* every derived private key to your terminal, where they remain
   in scrollback and shell logs.

Step 4 — Fund the aggregator
----------------------------

Print the addresses. This touches no network, so it works before anything else
is configured:

.. code:: shell

   uv run -- python addresses.py

Send USDC to the aggregator address. Budget one ``createSwarm`` (roughly 1M gas)
plus two transactions per round. On testnet, ``https://faucet.circle.com``
issues test USDC.

.. note::

   The aggregator is the **same address** on every network, because the same
   mnemonic derives the same accounts everywhere. Funding it on testnet does
   nothing on mainnet, and the address itself is no evidence of which chain you
   are on — the chain guard is.

Step 5 — Run the pre-flight checks
----------------------------------

.. code:: shell

   uv run -- python preflight.py

It reads the same ``pyproject.toml`` the run does and verifies, **without
sending a transaction**, that:

* the RPC is reachable and reports the chain ID ``arc-network`` selected;
* web3's Proof-of-Authority middleware is configured as this chain needs;
* the ``SwarmV1Factory`` has code at the registered address;
* the aggregator can pay, priced at the current gas price;
* ``createSwarm`` would succeed — a dry ``eth_call`` that resolves both
  selectors through the ``SelectorFactory`` and reverts if either version is
  unregistered on this chain.

Fix every ``FAIL`` before continuing. Use ``--network testnet`` to check the
other network without editing anything. Neither the passphrase nor any key is
printed.

Step 6 — Run the federation
---------------------------

.. code:: shell

   uv run -- flwr run .

What happens, in order:

1. ``Arc mainnet (chain 5042) via https://rpc.mainnet.arc.io`` — the resolved
   network. The server aborts here if the RPC disagrees.
2. ``Web3 swarm contract address: 0x721a…6353`` — the factory.
3. One ``createSwarm`` transaction; the new swarm proxy address follows.
4. Per round: trainers sign, coalitions are evaluated, then ``distribute``
   followed by ``nextRound`` — two aggregator transactions.
5. Metrics are written under ``logs/``.

First run also downloads CIFAR-10 (roughly 170MB), the trainers' data.

Step 7 — Verify on-chain
------------------------

.. code:: shell

   export ARC_RPC=https://rpc.mainnet.arc.io
   export SWARM=0x...   # the proxy address from the run

   cast call $SWARM "currentRound()(uint256)" --rpc-url $ARC_RPC

Then open ``https://explorer.arc.io/address/<swarm>`` and confirm the
``TrainerContributed`` events, one per trainer per round.

Reusing a swarm
---------------

While ``factory_v1`` is set, every run deploys a **new** swarm — on mainnet,
that means paying for one each time. To keep training against an existing swarm,
replace the factory block with its address:

.. code-block:: toml

   [tool.web3.swarm]
   address = "0xYourSwarmProxyAddress"

``SwarmConfig`` accepts exactly one of ``address`` or ``factory_v1``, so remove
the ``[tool.web3.swarm.factory_v1]`` block when you add ``address``.

Cost model
==========

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Action
     - Frequency
     - Gas
   * - Contract deployment
     - once per chain, already done
     - ≈ 15.25M (measured)
   * - ``createSwarm``
     - once per federation
     - ≈ 1M
   * - ``distribute`` + ``nextRound``
     - per round
     - grows with trainer count
   * - Trainer signing
     - per round
     - 0 — off-chain EIP-712

``distribute`` batches every trainer into one transaction, so per-round cost
grows with the number of trainers while the transaction *count* stays at two.
``preflight.py`` turns these into a USDC figure at the current gas price.

Further reading
===============

* ``examples/arc/README.md`` — the example's own documentation, including
  rehearsing against a local Anvil chain.
* ``forge/DEPLOYING.md`` — bringing the contracts up on a chain that does not
  have them yet.
