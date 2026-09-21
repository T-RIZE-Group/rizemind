# Rizemind on Arc mainnet

A minimal end-to-end run against [Arc](https://www.arc.io), Circle's USDC-gas L1:
three trainers, one round, decentralized Shapley value, CPU-only simulation.

> [!CAUTION]
> Every transaction this example sends spends **real USDC**. Rehearse on Arc
> testnet first — the steps are identical, only the RPC URL and the chain ID
> change.

## Network facts

|          | Arc mainnet                 | Arc testnet                                              |
| -------- | --------------------------- | -------------------------------------------------------- |
| Chain ID | `5042`                      | `5042002`                                                 |
| RPC      | `https://rpc.mainnet.arc.io` | `https://rpc.testnet.arc.io`                             |
| Explorer | `https://explorer.arc.io`   | `https://explorer.testnet.arc.io`                         |
| Gas      | USDC                        | test USDC (<https://faucet.circle.com>)                   |

Two Arc properties matter here:

- **USDC has two views.** The _native_ view (`eth_getBalance`, gas prices,
  `msg.value`) uses **18 decimals**, so web3.py's and Foundry's wei math needs no
  adjustment — what `forge` prints as `1 ETH` is `1 USDC`. The _ERC-20_ view
  (precompile `0x3600…0000`) uses **6 decimals**. Same balance, factor of
  10<sup>12</sup> apart. `preflight.py` reports the native view.
- **Deployment is permissionless.** Unlike Rizenet, Arc has no deployer
  allowlist, so there is no whitelisting step.

> [!IMPORTANT]
> Confirm the chain ID and RPC against the official Arc docs before spending
> anything, then prove it from your shell:
>
> ```shell
> cast chain-id --rpc-url https://rpc.mainnet.arc.io   # expect 5042
> ```
>
> Do not take an RPC URL from a search result or a chain-list aggregator. The
> value this example is configured with is `expected-chain-id` in
> `pyproject.toml`; both `preflight.py` and the server abort if the live RPC
> disagrees with it.

## What is already deployed

The `SwarmV1Factory` is deployed once per chain. Both Arc networks are
registered in the library, so this example needs no address of its own:

| Contract         | Network     | Address                                      |
| ---------------- | ----------- | -------------------------------------------- |
| `SwarmV1Factory` | Arc mainnet | `0x721a4ebA8747eF5db299E8Eec67A2FbAe7866353` |
| `SwarmV1Factory` | Arc testnet | `0x721a4ebA8747eF5db299E8Eec67A2FbAe7866353` |

The two are the same address. `CREATE` derives it from deployer and nonce alone,
so the same deployer running the same scripts from a fresh account on both chains
lands on the same one — it is not a copy-paste slip.

See `src/py/rizemind/contracts/swarm/swarm_v1/swarm_v1_factory.py`. If you deploy
your own factory instead (`forge/script/deployments/`), override it from this
example's `pyproject.toml` without touching the library:

```toml
[tool.web3.swarm.factory_v1.factory_deployments.5042]
address = "0xYourSwarmV1FactoryAddress"
```

> [!WARNING]
> Setting `factory_deployments` **replaces** the default map, so the other chains
> disappear from that config. That is fine for an Arc-only project.

Each _run_ then deploys one `SwarmV1` ERC-1967 proxy through the factory.

## Setup

### 1. Create the aggregator account

Only the **aggregator** sends transactions. Trainers sign EIP-712 messages
off-chain and never touch the chain, so their accounts need no USDC.

```shell
uv run -- rzmnd account generate arc-aggregator --words 24
export RZMND_PASSPHRASE='the passphrase you just chose'
```

Account 0 of that mnemonic is the aggregator; accounts 1..N are the simulated
trainers. Print them — this touches no network, so you can do it before anything
else:

```shell
uv run -- python addresses.py
```

> [!CAUTION]
> Do not use `rzmnd account load` to look them up — it prints the mnemonic _and_
> every private key to your terminal, where they stay in scrollback and shell
> logs.

### 2. Fund the aggregator

Send USDC to the aggregator address that `addresses.py` printed. Budget one
`createSwarm` (roughly 1M gas) plus two transactions per round (`distribute` +
`nextRound`); `preflight.py` turns that into a number at the current gas price.

The trainer addresses need nothing.

### 3. Run the pre-flight checks

```shell
export ARC_RPC_URL=https://rpc.mainnet.arc.io
export RZMND_PASSPHRASE='…'
cd examples/arc_mainnet
uv run -- python preflight.py
```

It reads the same `pyproject.toml` the run does and verifies, **without sending a
transaction**, that:

- the RPC is reachable and reports `expected-chain-id`;
- web3's PoA middleware is configured the way this chain needs (see below);
- the `SwarmV1Factory` has code at the registered address;
- the aggregator can pay, at the current gas price;
- `createSwarm` would succeed — a dry `eth_call` that resolves both selectors
  through the `SelectorFactory` and reverts if either version is unregistered.

Fix every `FAIL` before going further. Neither the passphrase nor any key is
printed.

### 4. Run it

```shell
uv run -- flwr run .
```

What you should see, in order:

1. `Web3 swarm contract address: 0x721a…6353` — the factory. If this is not the
   address you expect, your config did not take effect.
2. One `createSwarm` transaction; the new swarm proxy address follows.
3. Per round: trainers sign, coalitions evaluate, then `distribute` followed by
   `nextRound` — two aggregator transactions.
4. Metrics under `logs/`.

### 5. Verify on-chain

```shell
export SWARM=0x...   # the proxy address from the run

cast call $SWARM "currentRound()(uint256)" --rpc-url https://rpc.mainnet.arc.io
```

Then open `https://explorer.arc.io/address/<swarm>` and confirm the
`TrainerContributed` events, one per trainer per round.

## Reusing the swarm

`get_or_deploy` deploys a **new** swarm on every run while `factory_v1` is set,
which on mainnet means paying for a fresh swarm each time. To keep training
against the swarm you just created, replace the factory block with its address:

```toml
[tool.web3.swarm]
address = "0xYourSwarmProxyAddress"
```

`SwarmConfig` accepts exactly one of `address` or `factory_v1`, so delete the
`[tool.web3.swarm.factory_v1]` block when you add `address`.

## Configuration notes

- `TomlConfig` expands `$VAR` in any string, so the passphrase and RPC URL stay
  out of the file. Export both before running; an unset variable is caught by
  `preflight.py` rather than silently falling back.
- `num-supernodes` appears twice on purpose. The server derives one trainer
  address per supernode from `[tool.flwr.app.config]`; the federation block sizes
  the simulation. `preflight.py` fails if they disagree, because the swarm's
  member list would not match the clients that connect.
- `DecentralShapleyValueStrategy` evaluates every non-empty coalition —
  2<sup>n</sup>-1 per round. Three trainers means 7 coalitions; ten would mean
  1023. That is why this config is deliberately small.
- The server redacts `tool.eth.account` before writing the run config to `logs/`,
  so the expanded passphrase does not land on disk.

## Does Arc need the PoA middleware?

`Web3Config.get_web3()` injects `ExtraDataToPOAMiddleware` only for chains listed
in `poaChains` (`src/py/rizemind/web3/config.py`). Chains whose block header
`extraData` exceeds 32 bytes break web3.py without it — and web3 fetches a block
while building any EIP-1559 transaction, so the failure would land on
`createSwarm`.

Arc is **not** in `poaChains`, because that could not be verified from a network
that can reach the Arc RPC. `preflight.py` probes it and tells you which way it
went. If it reports that the middleware is needed, add the chain:

```python
from rizemind.web3.chains import ARC_MAINNET_CHAINID, RIZENET_TESTNET_CHAINID

poaChains = [RIZENET_TESTNET_CHAINID, ARC_MAINNET_CHAINID]
```

## Running on Arc testnet instead

Arc testnet is registered in the library too, so switching networks is two
overrides and no file edit:

```shell
export ARC_RPC_URL=https://rpc.testnet.arc.io

uv run -- python preflight.py --chain-id 5042002
uv run -- flwr run . --run-config expected-chain-id=5042002
```

Both refuse to run if the RPC does not report 5042002, so the pair keeps mainnet
and testnet from being confused for one another.

Test USDC comes from <https://faucet.circle.com>. The aggregator is a different
account per network only if you make it one — the same mnemonic derives the same
addresses everywhere, so a funded testnet aggregator is the same address on
mainnet.

To point at a factory you deployed yourself instead, override it:

```toml
[tool.web3.swarm.factory_v1.factory_deployments.5042002]
address = "0xYourTestnetFactoryAddress"
```

That **replaces** the default map rather than extending it, so mainnet
disappears from that config.

## Rehearsing on a local chain

Before spending anything, run the exact same code against Anvil. This exercises
the whole path — `createSwarm`, the selectors, per-round `distribute` and
`nextRound` — on a throwaway chain, so the only thing left untested on Arc is Arc
itself.

Deploy the contracts to a local node, following the same order Part 1 uses on a
real chain (`forge/deploy.sh` runs all four scripts and is fine here, because
Anvil's keys are public):

```shell
cd forge
forge soldeer install
forge build
anvil &                  # chain id 31337, pre-funded accounts
./deploy.sh              # SelectorFactory, both selectors, then SwarmV1Factory
```

Then point this example at it. Replace the `[tool.eth.account.mnemonic_store]`,
`[tool.web3]` and `[tool.web3.swarm.factory_v1]` blocks in `pyproject.toml` with:

```toml
[tool.eth.account]
# Anvil's default mnemonic. Account 0 is pre-funded and is also deploy.sh's
# deployer. Public knowledge — never use it on a chain that holds value.
mnemonic = "test test test test test test test test test test test junk"

[tool.web3]
url = "http://127.0.0.1:8545"

[tool.web3.swarm.factory_v1]
name = "arc_smoke_test"
ticker = "ARCTEST"
local_factory_deployment_path = "../../forge/broadcast/SwarmV1Factory.s.sol/31337/run-latest.json"
```

and set the guard to Anvil's chain:

```toml
[tool.flwr.app.config]
expected-chain-id = 31337
```

`local_factory_deployment_path` takes precedence over the chain-ID map, so the
factory address comes straight out of what you just deployed. `preflight.py` and
`flwr run .` work unchanged from there.

Revert those four blocks before running against Arc. `expected-chain-id` is the
backstop: with it left at `31337`, the server refuses to touch mainnet.

## Troubleshooting

**`Chain ID#5042 is unsupported, provide a local_deployment_path`**
The library does not know a factory for this chain. Either you are on a chain
other than Arc mainnet, or a `factory_deployments` override is under the wrong
table — it must be `[tool.web3.swarm.factory_v1.factory_deployments.<chain id>]`.

**`RPC reports chain ID …, expected 5042`**
The endpoint is not the chain in `expected-chain-id`. Fix one or the other; do
not "fix" it by loosening the check.

**`SelectorImplementationNotFound` on createSwarm**
The selector version this config asks for is not registered on this chain. The
Python side derives the selector ID as `keccak256("<name>-v<version>")`; the
Solidity side derives it from the implementation's EIP-712 domain version. Check
with `cast call $SELECTOR_FACTORY "isSelectorVersionRegistered(string)(bool)"
"always-sampled-v1.0.0"`.

**`web3.exceptions.ExtraDataLengthError`**
Arc needs the PoA middleware — see the section above.

**`assert tx_receipt["status"] != 0`**
An aggregator transaction reverted. Re-run it with
`cast run <tx-hash> --rpc-url https://rpc.mainnet.arc.io` for a decoded trace.

**Balances look 10<sup>12</sup> off**
You mixed the native (18-decimal) and ERC-20 (6-decimal) views of USDC.
`cast balance` and gas prices are native; the ERC-20 precompile is not.

## Security checklist

- Aggregator mnemonic in the `rzmnd` keystore, passphrase from the environment.
  Never paste a mnemonic into `pyproject.toml`, never commit `.env`.
- The factory address in this README should match what the explorer shows before
  you send the first `createSwarm`.
- Deploying your own factory: keep the deployer key in a Foundry keystore or a
  hardware wallet, never `--private-key` on a mainnet command line. Whoever owns
  the `SelectorFactory` holds a standing privilege to register selector
  implementations for every future swarm on that chain — a multisig is the right
  answer for a shared deployment.
