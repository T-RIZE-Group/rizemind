==========================================
Deploying Rizemind on Arc Mainnet
==========================================

This runbook takes you from an empty chain to a finished federated-learning run on
`Arc <https://www.arc.io>`_, Circle's USDC-gas L1:

1. **Part 1** deploys the one-time infrastructure (the selector and swarm factories).
2. **Part 2** points the Python library at your freshly deployed factory.
3. **Part 3** runs a small example end-to-end against it.

.. caution::
   Every transaction in Part 1 and Part 3 spends **real USDC**. Rehearse the whole
   runbook on Arc testnet first — the steps are identical, only the RPC URL changes.

What gets deployed
==================

Deployed **once per chain** (Part 1, via Foundry):

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Contract
     - Role
   * - ``SelectorFactory``
     - Registry + ERC-1967 proxy factory for trainer/evaluator selectors. Owned by
       the address in ``SELECTOR_FACTORY_OWNER``; only that owner can register
       new selector implementations.
   * - ``AlwaysSampled``
     - Selector implementation, version ``always-sampled-v1.0.0``. Registered in
       ``SelectorFactory``.
   * - ``RandomSampling``
     - Selector implementation, version ``random-sampling-v1.0.0``. Registered in
       ``SelectorFactory``.
   * - ``SwarmV1``
     - Swarm logic contract. Never used directly; proxies delegate to it.
   * - ``SwarmV1Factory``
     - What the Python library calls. ``createSwarm()`` deploys a swarm proxy plus
       its two selector proxies in a single transaction.

Deployed **once per federation** (Part 3, by the aggregator): a ``SwarmV1``
ERC-1967 proxy, created through ``SwarmV1Factory.createSwarm()``.

Network facts
=============

.. list-table::
   :header-rows: 1
   :widths: 25 37 37

   * -
     - Arc mainnet
     - Arc testnet
   * - Chain ID
     - ``5042`` (``0x13b2``)
     - ``5042002`` (``0x4cef52``)
   * - RPC
     - ``https://rpc.mainnet.arc.io``
     - ``https://rpc.testnet.arc.io``
   * - Explorer
     - ``https://explorer.arc.io``
     - ``https://explorer.testnet.arc.io``
   * - Gas token
     - USDC
     - test USDC (``https://faucet.circle.com``)

Two Arc properties matter for this repo:

**USDC has two views.** The *native* view (``eth_getBalance``, gas prices,
``msg.value``) uses **18 decimals**, so Foundry's and web3.py's wei math needs no
adjustment: what ``forge`` prints as ``1 ETH`` is ``1 USDC``. The *ERC-20* view
(precompile ``0x3600000000000000000000000000000000000000``) uses **6 decimals**.
The two differ by a factor of 10\ :sup:`12` and are the same balance, so native
amounts finer than 10\ :sup:`12` wei are not representable.

**Deployment is permissionless.** Unlike Rizenet, Arc has no deployer allowlist —
block production is permissioned, application deployment is not. There is no
whitelisting step.

.. important::
   Confirm the chain ID and RPC against the official Arc docs before you spend
   anything, then prove it from your shell:

   .. code:: shell

      cast chain-id --rpc-url https://rpc.mainnet.arc.io   # expect 5042

   If that does not return ``5042``, stop and fix your endpoint. Do not take an
   RPC URL from a search result, a chain-list aggregator, or this page alone.

Prerequisites
=============

.. code:: shell

   # Foundry (forge, cast)
   curl -L https://foundry.paradigm.xyz | bash && foundryup

   # Python toolchain and this repo's dependencies
   uv sync

You also need a funded deployer account and, for Part 3, an aggregator account.
They may be the same account, but separating them is cleaner: the deployer holds
``SelectorFactory`` ownership and can stay cold afterwards.

Part 1 — Deploy the factories
=============================

Step 1.1: Add the Arc endpoints
-------------------------------

Add Arc to ``forge/foundry.toml`` so you can use short RPC aliases:

.. code-block:: toml

   [rpc_endpoints]
   local = "http://127.0.0.1:8545"
   rizenet-testnet = "https://testnet.rizenet.io"
   arc-mainnet = "https://rpc.mainnet.arc.io"
   arc-testnet = "https://rpc.testnet.arc.io"

Step 1.2: Build
---------------

.. code:: shell

   cd forge
   forge soldeer install
   forge build

Step 1.3: Prepare the deployer key
----------------------------------

Never pass a mainnet private key on the command line — it lands in your shell
history and in the process table. Import it into Foundry's encrypted keystore
instead:

.. code:: shell

   cast wallet import arc-deployer --interactive   # paste the key, set a password
   export DEPLOYER=$(cast wallet address --account arc-deployer)
   echo $DEPLOYER

A hardware wallet works too: replace ``--account arc-deployer`` with ``--ledger``
or ``--trezor`` in every ``forge script`` command below.

.. note::
   ``forge/deploy.sh`` exists but only accepts a raw private key, so it is fine for
   Anvil and wrong for mainnet. Part 1 runs the same four scripts by hand.

Step 1.4: Budget and fund
-------------------------

Gas used by a first-time deployment, measured from the broadcast artifacts of the
Rizenet testnet deployment in ``forge/broadcast/`` (Arc is EVM-equivalent, so
expect the same numbers):

.. list-table::
   :header-rows: 1
   :widths: 45 20 35

   * - Script
     - Gas
     - Contents
   * - ``selectors/SelectorFactory.s.sol``
     - 3,083,562
     - ``DevOpsTools`` 1,581,190 + ``SelectorFactory`` 1,502,372
   * - ``selectors/AlwaysSampled.s.sol``
     - 2,480,681
     - ``DevOpsTools`` + impl 834,748 + register 64,743
   * - ``selectors/RandomSampling.s.sol``
     - 2,781,042
     - ``DevOpsTools`` + impl 1,135,097 + register 64,755
   * - ``SwarmV1Factory.s.sol``
     - 6,895,731
     - ``DevOpsTools`` + ``SwarmV1`` 3,602,529 + factory 1,712,012
   * - **Total**
     - **≈ 15.25 M**
     -

Convert that to USDC at the current Arc gas price:

.. code:: shell

   GAS_PRICE=$(cast gas-price --rpc-url arc-mainnet)
   python3 -c "print(f'{15_241_016 * $GAS_PRICE / 1e18:.6f} USDC')"

Fund ``$DEPLOYER`` with at least twice that, then confirm the balance — remember
that ``cast balance`` reports the 18-decimal native view, so ``cast to-unit … ether``
gives you USDC:

.. code:: shell

   cast to-unit $(cast balance $DEPLOYER --rpc-url arc-mainnet) ether

.. note::
   ``DevOpsTools`` is an external library that Foundry redeploys on each of the four
   scripts, costing about 6.3 M gas in total. It is only used to look up previous
   deployments from ``broadcast/``. Converting its functions to ``internal`` would
   inline them and remove that cost; that is a code change, not part of this runbook.

Step 1.5: Dry-run, then broadcast — in order
--------------------------------------------

The scripts are ordered by dependency: everything else needs ``SelectorFactory``
to already exist on-chain, because the selector scripts *call* it. Dry-running all
four up front therefore fails; dry-run each one right before you broadcast it.

**1. SelectorFactory**

``SELECTOR_FACTORY_OWNER`` becomes the owner that is allowed to register selector
implementations. Steps 2 and 3 call ``registerSelectorImplementation()``, which is
``onlyOwner``, so the broadcasting account must *be* that owner — keep them the
same unless you are ready to sign the registrations from the owner separately.

.. code:: shell

   export SELECTOR_FACTORY_OWNER=$DEPLOYER

   # dry run (simulation only, no transactions)
   forge script script/deployments/selectors/SelectorFactory.s.sol \
       --rpc-url arc-mainnet --sender $DEPLOYER

   # broadcast
   forge script script/deployments/selectors/SelectorFactory.s.sol \
       --rpc-url arc-mainnet --account arc-deployer --sender $DEPLOYER \
       --broadcast --slow

Copy the ``SelectorFactory deployed at:`` address from the output and pin it, so the
remaining scripts resolve it explicitly instead of scanning ``broadcast/``:

.. code:: shell

   export SELECTOR_FACTORY=0x...   # from the log above

**2. AlwaysSampled** and **3. RandomSampling**

.. code:: shell

   for s in AlwaysSampled RandomSampling; do
     forge script script/deployments/selectors/$s.s.sol \
         --rpc-url arc-mainnet --account arc-deployer --sender $DEPLOYER \
         --broadcast --slow
   done

Both scripts are idempotent: if the version is already registered they log
``Already Registered`` and send nothing.

**4. SwarmV1Factory**

.. code:: shell

   forge script script/deployments/SwarmV1Factory.s.sol \
       --rpc-url arc-mainnet --account arc-deployer --sender $DEPLOYER \
       --broadcast --slow

Record ``Swarm Implementation deployed at:`` and ``Factory deployed at:``.

Step 1.6: Record the addresses
------------------------------

Foundry writes them to ``forge/broadcast/<script>.s.sol/5042/run-latest.json``.
Commit those artifacts — Part 2 can read the factory address straight out of them.
Keep a copy in your own notes as well:

.. code-block:: text

   Chain:                 Arc mainnet (5042)
   SelectorFactory:       0x...    owner: 0x...
   AlwaysSampled impl:    0x...    always-sampled-v1.0.0
   RandomSampling impl:   0x...    random-sampling-v1.0.0
   SwarmV1 implementation:0x...
   SwarmV1Factory:        0x...    <- this is the address the library needs

Step 1.7: Verify the deployment
-------------------------------

On-chain sanity checks — all four must be true before you go on:

.. code:: shell

   cast call $SELECTOR_FACTORY "owner()(address)" --rpc-url arc-mainnet
   cast call $SELECTOR_FACTORY "isSelectorVersionRegistered(string)(bool)" \
       "always-sampled-v1.0.0" --rpc-url arc-mainnet          # true
   cast call $SELECTOR_FACTORY "isSelectorVersionRegistered(string)(bool)" \
       "random-sampling-v1.0.0" --rpc-url arc-mainnet         # true
   cast call $SWARM_FACTORY "getImplementation()(address)" --rpc-url arc-mainnet

Those version strings are not decorative: the Python client derives the selector ID
as ``keccak256("<name>-v<version>")`` (``AlwaysSamplesSelectorConfig``), and the
Solidity side derives it from the implementation's EIP-712 domain version. If they
ever diverge, ``createSwarm`` reverts with ``SelectorImplementationNotFound``.

Arc runs Blockscout-style explorers, so source verification is:

.. code:: shell

   forge verify-contract $SELECTOR_FACTORY \
       src/sampling/SelectorFactory.sol:SelectorFactory \
       --chain-id 5042 --verifier blockscout \
       --verifier-url https://explorer.arc.io/api \
       --constructor-args $(cast abi-encode "constructor(address)" $SELECTOR_FACTORY_OWNER)

Repeat for each contract (``SwarmV1Factory`` takes
``constructor(address,address)`` — logic contract, then selector factory). Confirm
the API path against the explorer's own documentation if the call 404s.

Part 2 — Point the library at Arc
=================================

``SwarmV1FactoryConfig`` resolves the factory address from a chain-ID map that
currently only knows Rizenet testnet, so an unmodified checkout fails on Arc with
``Chain ID#5042 is unsupported, provide a local_deployment_path``. Pick one of
three ways to fix that.

Option A — TOML only (no code changes)
--------------------------------------

Add the address to the example's ``pyproject.toml``:

.. code-block:: toml

   [tool.web3.swarm.factory_v1]
   name = "arc_smoke_test"

   [tool.web3.swarm.factory_v1.factory_deployments.5042]
   address = "0xYourSwarmV1FactoryAddress"

.. warning::
   Setting ``factory_deployments`` **replaces** the default map, so the Rizenet
   testnet entry disappears from that config. That is fine for an Arc-only project.

Option B — read the Foundry artifact
------------------------------------

Skip addresses entirely and let the config read what Part 1 broadcast. The path is
relative to the directory you run ``flwr`` from:

.. code-block:: toml

   [tool.web3.swarm.factory_v1]
   name = "arc_smoke_test"
   local_factory_deployment_path = "../../forge/broadcast/SwarmV1Factory.s.sol/5042/run-latest.json"

Handy while iterating; brittle once anyone re-runs the deployment script.

Option C — register the chain in the library (recommended once it is permanent)
-------------------------------------------------------------------------------

This is the only option that also makes the ``rzmnd swarm new`` CLI work against
Arc, since the CLI builds its ``SwarmV1FactoryConfig`` in code with no TOML
override.

``src/py/rizemind/web3/chains.py``:

.. code-block:: python

   RIZENET_TESTNET_CHAINID = 79123
   ARC_MAINNET_CHAINID = 5042
   ARC_TESTNET_CHAINID = 5042002

``src/py/rizemind/contracts/swarm/swarm_v1/swarm_v1_factory.py``:

.. code-block:: python

   from rizemind.web3.chains import ARC_MAINNET_CHAINID, RIZENET_TESTNET_CHAINID

   factory_deployments: dict[int, DeployedContract] = {
       RIZENET_TESTNET_CHAINID: DeployedContract(
           address=Web3.to_checksum_address("0xd66c7c89fb97ea5c06b0b7caf2086df1e82b9e88")
       ),
       ARC_MAINNET_CHAINID: DeployedContract(
           address=Web3.to_checksum_address("0xYourSwarmV1FactoryAddress")
       ),
   }

Check whether Arc needs PoA middleware
--------------------------------------

``Web3Config.get_web3()`` injects ``ExtraDataToPOAMiddleware`` only for chains
listed in ``poaChains``. Chains whose block header ``extraData`` exceeds 32 bytes
break web3.py without it. Probe Arc before you rely on it:

.. code-block:: python

   from web3 import Web3

   w3 = Web3(Web3.HTTPProvider("https://rpc.mainnet.arc.io"))
   print("chain id:", w3.eth.chain_id)
   block = w3.eth.get_block("latest")      # raises ExtraDataLengthError if PoA
   print("extraData bytes:", len(block["extraData"]))

If that raises ``web3.exceptions.ExtraDataLengthError``, add the chain to
``poaChains`` in ``src/py/rizemind/web3/config.py``:

.. code-block:: python

   poaChains = [RIZENET_TESTNET_CHAINID, ARC_MAINNET_CHAINID]

Part 3 — Test run with an example
=================================

We copy ``examples/rizenet_testnet`` rather than editing it, so nothing that says
"testnet" ends up pointing at mainnet.

Step 3.1: Create the example
----------------------------

.. code:: shell

   cd /path/to/rizemind
   cp -r examples/rizenet_testnet examples/arc_mainnet

Register it as a uv workspace member in the root ``pyproject.toml``:

.. code-block:: toml

   [tool.uv.workspace]
   members = [
       "examples/arc_mainnet",
       "examples/rizenet_testnet",
       # …
   ]

Step 3.2: Create the aggregator account
---------------------------------------

Only the **aggregator** sends transactions. Trainers sign EIP-712 messages
off-chain and never touch the chain, so they need no USDC.

.. code:: shell

   uv run rzmnd account generate arc-aggregator --words 24
   export RZMND_PASSPHRASE='the passphrase you just chose'

Print the addresses you need to fund (account 0 is the aggregator; 1..N are the
simulated trainers):

.. code-block:: python

   import os
   from rizemind.authentication.config import AccountConfig, MnemonicStoreConfig

   cfg = AccountConfig(
       mnemonic_store=MnemonicStoreConfig(
           account_name="arc-aggregator",
           passphrase=os.environ["RZMND_PASSPHRASE"],
       )
   )
   print("aggregator:", cfg.get_account(0).address)
   for i in range(1, 4):
       print(f"trainer {i}:", cfg.get_account(i).address)

.. caution::
   Do not use ``rzmnd account load`` for this — it prints the mnemonic *and* every
   private key to your terminal, where they stay in scrollback and shell logs.

Fund the aggregator address with USDC on Arc. Budget one ``createSwarm``
(roughly 1 M gas) plus two transactions per round (``distribute`` +
``nextRound``); measure the real numbers during your testnet rehearsal and scale.

Step 3.3: Configure the example
-------------------------------

Replace ``examples/arc_mainnet/pyproject.toml`` with this. It is the Rizenet
example trimmed to the smallest thing that still exercises the full path:
1 round, 3 trainers, CPU-only simulation.

.. code-block:: toml

   [project]
   name = "arc_mainnet"
   version = "0.1.0"
   description = "Rizemind smoke test on Arc mainnet"
   readme = "README.md"
   requires-python = ">=3.12"
   dependencies = [
       "flwr-datasets[vision]>=0.5.0",
       "flwr[simulation]>=1.18.0",
       "torch>=2.7.0",
       "torchvision>=0.22.0",
   ]

   [tool.hatch.build.targets.wheel]
   packages = ["."]

   [tool.flwr.app]
   publisher = "trizelabs"

   [tool.flwr.app.components]
   serverapp = "src.server:app"
   clientapp = "src.client:app"

   [tool.flwr.app.config]
   num-server-rounds = 1
   min-available-clients = 3
   num-supernodes = 3
   local-epochs = 1
   batch-size = 32
   learning-rate = 0.005
   fraction-fit = 1.0
   fraction-evaluate = 1.0
   verbose = false
   metrics-storage-path = 'logs'

   [tool.flwr.federations]
   default = "local-simulation"

   [tool.flwr.federations.local-simulation]
   options.num-supernodes = 3

   [tool.eth.account.mnemonic_store]
   account_name = "arc-aggregator"
   passphrase = "$RZMND_PASSPHRASE"

   [tool.web3]
   url = "$ARC_RPC_URL"

   [tool.web3.swarm.factory_v1]
   name = "arc_smoke_test"
   ticker = "ARCTEST"

   [tool.web3.swarm.factory_v1.factory_deployments.5042]
   address = "0xYourSwarmV1FactoryAddress"

Notes on this config:

- ``TomlConfig`` expands ``$VAR`` in any string, so the passphrase and RPC URL stay
  out of the file. Export both before running; an unset variable surfaces as a
  URL-validation error rather than a silent fallback.
- ``num-supernodes`` appears twice on purpose. The server derives one trainer
  address per supernode from ``[tool.flwr.app.config]``; the federation block sizes
  the simulation. Keep them equal or the swarm's member list will not match the
  clients that actually connect.
- The example uses ``DecentralShapleyValueStrategy``, which evaluates every
  non-empty coalition — 2\ :sup:`n`-1 per round. Three trainers means 7 coalitions.
  Ten would mean 1023, which is why this config is deliberately small.
- Drop the ``factory_deployments`` block if you chose Option B or C in Part 2.

Step 3.4: Run it
----------------

.. code:: shell

   export ARC_RPC_URL=https://rpc.mainnet.arc.io
   export RZMND_PASSPHRASE='…'
   cd examples/arc_mainnet
   uv run -- flwr run .

What you should see, in order:

1. ``Web3 swarm contract address: 0x…`` — the ``SwarmV1Factory`` from Part 1. If this
   is not your address, the config did not take effect.
2. One ``createSwarm`` transaction; the new swarm proxy address follows.
3. Per round: trainers sign, coalitions evaluate, then ``distribute`` followed by
   ``nextRound`` — two aggregator transactions.
4. Metrics written under ``logs/``.

Step 3.5: Verify on-chain
-------------------------

.. code:: shell

   export SWARM=0x...   # the proxy address from the run

   cast call $SWARM "currentRound()(uint256)" --rpc-url arc-mainnet
   cast call $SWARM "canTrain(address,uint256)(bool)" $TRAINER_1 1 --rpc-url arc-mainnet

Then open ``https://explorer.arc.io/address/<swarm>`` and confirm the
``TrainerContributed`` events, one per trainer per round.

Step 3.6: Reuse the swarm
-------------------------

``get_or_deploy`` deploys a **new** swarm on every run when ``factory_v1`` is set,
which on mainnet means paying for a fresh swarm each time. To keep training against
the swarm you just created, swap the factory block for its address:

.. code-block:: toml

   [tool.web3.swarm]
   address = "0xYourSwarmProxyAddress"

``SwarmConfig`` accepts exactly one of ``address`` or ``factory_v1`` and rejects
both together, so delete the ``factory_v1`` block when you add ``address``.

Cost model
==========

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Action
     - Frequency
     - Gas
   * - Factory deployment (Part 1)
     - once per chain
     - ≈ 15.25 M (measured)
   * - ``createSwarm``
     - once per federation
     - ≈ 1 M (measure on testnet)
   * - ``distribute`` + ``nextRound``
     - per round
     - measure on testnet
   * - Trainer signing
     - per round
     - 0 — off-chain EIP-712

``distribute`` batches every trainer into one transaction, so per-round cost grows
with the number of trainers but the transaction *count* stays at two.

Troubleshooting
===============

**Chain ID#5042 is unsupported, provide a local_deployment_path**
   Part 2 is not applied, or the TOML override is under the wrong table. It must be
   ``[tool.web3.swarm.factory_v1.factory_deployments.5042]``.

**SelectorFactory not found. Deploy SelectorFactory first or set SELECTOR_FACTORY**
   You ran a later script before ``SelectorFactory`` existed on Arc, or you started a
   new shell and lost ``SELECTOR_FACTORY``. Re-export it.

**Reverts with ``OwnableUnauthorizedAccount`` during registration**
   The broadcasting account is not ``SELECTOR_FACTORY_OWNER``. Re-run steps 2 and 3
   with the owner's key.

**``SelectorImplementationNotFound`` on createSwarm**
   The selector version the Python config asks for is not registered on this chain.
   Check with ``isSelectorVersionRegistered("always-sampled-v1.0.0")``.

**``web3.exceptions.ExtraDataLengthError``**
   Arc needs the PoA middleware — see the probe at the end of Part 2.

**``assert tx_receipt["status"] != 0``**
   An aggregator transaction reverted. Re-run it with
   ``cast run <tx-hash> --rpc-url arc-mainnet`` for a decoded trace.

**Balances look 10**\ :sup:`12` **off**
   You mixed the native (18-decimal) and ERC-20 (6-decimal) views of USDC.
   ``cast balance`` and gas prices are native; the ERC-20 precompile is not.

Security checklist
==================

- Deployer key in a Foundry keystore or a hardware wallet; never ``--private-key``
  on a mainnet command line.
- ``SELECTOR_FACTORY_OWNER`` set to an address you intend to keep — it is a
  standing privilege to register new selector implementations for every future
  swarm on this chain. A multisig is the right answer for a shared deployment.
- Aggregator mnemonic in the ``rzmnd`` keystore, passphrase from the environment.
  Never commit ``.env``, never paste a mnemonic into ``pyproject.toml``.
- The factory addresses in ``broadcast/`` and in your notes should match what the
  explorer shows before you send the first ``createSwarm``.
