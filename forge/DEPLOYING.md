# Deploying the contracts to a chain

The infrastructure below is deployed **once per chain**. After that, each
federation deploys only its own `SwarmV1` proxy through `SwarmV1Factory`, which
is what the Python library calls.

| Contract         | Role                                                                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------------------- |
| `SelectorFactory` | Registry + ERC-1967 proxy factory for trainer/evaluator selectors. Only its owner may register implementations. |
| `AlwaysSampled`   | Selector implementation, version `always-sampled-v1.0.0`. Registered in `SelectorFactory`.                      |
| `RandomSampling`  | Selector implementation, version `random-sampling-v1.0.0`. Registered in `SelectorFactory`.                     |
| `SwarmV1`         | Swarm logic contract. Never called directly; proxies delegate to it.                                            |
| `SwarmV1Factory`  | `createSwarm()` deploys a swarm proxy plus its two selector proxies in one transaction.                         |

## Networks

RPC aliases are defined in `foundry.toml`, so `--rpc-url arc-testnet` works from
this directory.

| Network      | Chain ID  | Alias             | Gas                                          |
| ------------ | --------- | ----------------- | -------------------------------------------- |
| Arc mainnet  | `5042`    | `arc-mainnet`     | USDC — **real money**                        |
| Arc testnet  | `5042002` | `arc-testnet`     | test USDC (<https://faucet.circle.com>)      |
| Rizenet testnet | `79123` | `rizenet-testnet` | free, but the deployer must be whitelisted    |
| Anvil        | `31337`   | `local`           | free                                          |

Arc deployment is permissionless — unlike Rizenet, there is no deployer
allowlist. Confirm the chain before spending anything:

```shell
cast chain-id --rpc-url arc-testnet    # expect 5042002
```

## Prepare

```shell
cd forge
forge soldeer install
forge build
```

Import the deployer key into Foundry's encrypted keystore rather than passing it
on the command line, where it lands in shell history and the process table:

```shell
cast wallet import arc-deployer --interactive
export DEPLOYER=$(cast wallet address --account arc-deployer)
```

A hardware wallet works too — replace `--account arc-deployer` with `--ledger`
or `--trezor` throughout.

> [!NOTE]
> `deploy.sh` runs all four scripts but only accepts a raw private key, so it is
> fine for Anvil and wrong for any chain that holds value. The steps below run
> the same scripts by hand.

## Deploy, in order

The scripts are ordered by dependency: steps 2–4 **call** `SelectorFactory`, so
it must already exist on-chain. Dry-running all four up front therefore fails —
dry-run each one immediately before broadcasting it.

### 1. SelectorFactory

`SELECTOR_FACTORY_OWNER` becomes the only address allowed to register selector
implementations. Steps 2 and 3 call `registerSelectorImplementation()`, which is
`onlyOwner`, so the broadcasting account must *be* that owner unless you plan to
sign the registrations separately.

```shell
export SELECTOR_FACTORY_OWNER=$DEPLOYER

# dry run — simulation only, no transactions
forge script script/deployments/selectors/SelectorFactory.s.sol \
    --rpc-url arc-testnet --sender $DEPLOYER

# broadcast
forge script script/deployments/selectors/SelectorFactory.s.sol \
    --rpc-url arc-testnet --account arc-deployer --sender $DEPLOYER \
    --broadcast --slow
```

Copy the `SelectorFactory deployed at:` address and pin it:

```shell
export SELECTOR_FACTORY=0x...
```

This matters more than it looks. Steps 2–4 resolve the selector factory from
`$SELECTOR_FACTORY` first, and otherwise fall back to scanning `broadcast/` for
this chain ID. Exporting it makes the remaining scripts explicit instead of
dependent on local files — and it is the only way they work from a checkout that
does not carry your broadcast artifacts.

### 2 and 3. The selectors

```shell
for s in AlwaysSampled RandomSampling; do
  forge script script/deployments/selectors/$s.s.sol \
      --rpc-url arc-testnet --account arc-deployer --sender $DEPLOYER \
      --broadcast --slow
done
```

Both are idempotent: if the version is already registered they log
`Already Registered` and send nothing.

### 4. SwarmV1Factory

```shell
forge script script/deployments/SwarmV1Factory.s.sol \
    --rpc-url arc-testnet --account arc-deployer --sender $DEPLOYER \
    --broadcast --slow
```

Record `Swarm Implementation deployed at:` and `Factory deployed at:`. The
factory address is the one the Python library needs.

## Record the addresses

Foundry writes them to `broadcast/<script>.s.sol/<chain id>/run-latest.json`.
**Commit those artifacts.** They are how `DevOpsTools` finds earlier deployments
on a later run, and how anyone else reproduces what is on-chain. Keep a copy in
your own notes too:

```text
Chain:                  Arc testnet (5042002)
SelectorFactory:        0x...    owner: 0x...
AlwaysSampled impl:     0x...    always-sampled-v1.0.0
RandomSampling impl:    0x...    random-sampling-v1.0.0
SwarmV1 implementation: 0x...
SwarmV1Factory:         0x...    <- the library needs this one
```

## Verify

All four must hold before you go further:

```shell
cast call $SELECTOR_FACTORY "owner()(address)" --rpc-url arc-testnet
cast call $SELECTOR_FACTORY "isSelectorVersionRegistered(string)(bool)" \
    "always-sampled-v1.0.0" --rpc-url arc-testnet          # true
cast call $SELECTOR_FACTORY "isSelectorVersionRegistered(string)(bool)" \
    "random-sampling-v1.0.0" --rpc-url arc-testnet         # true
cast call $SWARM_FACTORY "getImplementation()(address)" --rpc-url arc-testnet
```

Those version strings are not decorative. The Python client derives the selector
ID as `keccak256("<name>-v<version>")` (`SelectorConfig.get_selector_id`); the
Solidity side derives it from the implementation's EIP-712 domain version. If
they diverge, `createSwarm` reverts with `SelectorImplementationNotFound`.

Arc runs Blockscout-style explorers, so source verification is:

```shell
forge verify-contract $SELECTOR_FACTORY \
    src/sampling/SelectorFactory.sol:SelectorFactory \
    --chain-id 5042002 --verifier blockscout \
    --verifier-url https://explorer.testnet.arc.io/api \
    --constructor-args $(cast abi-encode "constructor(address)" $SELECTOR_FACTORY_OWNER)
```

Repeat per contract; `SwarmV1Factory` takes `constructor(address,address)` — the
logic contract, then the selector factory. Confirm the API path against the
explorer's own docs if the call 404s.

## Tell the library about it

Register the chain so every example and the `rzmnd swarm new` CLI find the
factory without a per-project override.

Both Arc networks and Rizenet testnet are already registered in
`SwarmV1FactoryConfig.factory_deployments`
(`src/py/rizemind/contracts/swarm/swarm_v1/swarm_v1_factory.py`). For a new chain,
add its ID to `src/py/rizemind/web3/chains.py` and an entry to that map:

```python
factory_deployments: dict[int, DeployedContract] = {
    ...,
    MY_CHAIN_ID: DeployedContract(
        address=Web3.to_checksum_address("0xYourFactoryAddress")
    ),
}
```

A project can also point at its own deployment without a library change:

```toml
[tool.web3.swarm.factory_v1.factory_deployments.5042002]
address = "0xYourFactoryAddress"
```

Note that this **replaces** the default map rather than extending it, so the
other chains disappear from that config.

Addresses repeating across chains is normal, not a mistake: `CREATE` derives the
address from deployer and nonce alone, so the same deployer running these scripts
from a fresh account on two chains produces the same addresses on both. Verify
against the chain before "correcting" one.

Also check whether the chain needs web3's PoA middleware — see "Does Arc need the
PoA middleware?" in `examples/arc_mainnet/README.md`.

## Cost

First-time gas, measured from the committed Rizenet testnet artifacts in
`broadcast/` (the EVM is equivalent, so expect the same numbers on Arc):

| Script                              | Gas            | Contents                                            |
| ----------------------------------- | -------------- | --------------------------------------------------- |
| `selectors/SelectorFactory.s.sol`   | 3,083,562      | `DevOpsTools` 1,581,190 + `SelectorFactory`          |
| `selectors/AlwaysSampled.s.sol`     | 2,480,681      | `DevOpsTools` + impl + register                      |
| `selectors/RandomSampling.s.sol`    | 2,781,042      | `DevOpsTools` + impl + register                      |
| `SwarmV1Factory.s.sol`              | 6,895,731      | `DevOpsTools` + `SwarmV1` + factory                  |
| **Total**                           | **15,241,016** |                                                      |

Convert at the current gas price, and fund the deployer with at least twice it:

```shell
GAS_PRICE=$(cast gas-price --rpc-url arc-testnet)
python3 -c "print(f'{15_241_016 * $GAS_PRICE / 1e18:.6f}')"
```

On Arc that figure is USDC: the native view (`cast balance`, gas prices,
`msg.value`) uses 18 decimals like ether, while the ERC-20 precompile uses 6.
Same balance, factor of 10^12 apart.

> [!NOTE]
> `DevOpsTools` is an external library that Foundry redeploys for each script —
> about 6.3M gas of the total, purely to look up previous deployments from
> `broadcast/`. Making its functions `internal` would inline them and remove that
> cost. That is a code change, not a deployment step.

## Troubleshooting

**`SelectorFactory not found. Deploy SelectorFactory first or set SELECTOR_FACTORY`**
A later script ran before `SelectorFactory` existed on this chain, or a new shell
lost `$SELECTOR_FACTORY`. Re-export it.

**Reverts with `OwnableUnauthorizedAccount` during registration**
The broadcasting account is not `SELECTOR_FACTORY_OWNER`. Re-run steps 2 and 3
with the owner's key.

**`SelectorFactory codehash mismatch`**
Step 1 found an existing deployment whose bytecode differs from what this
checkout builds. The chain has an older `SelectorFactory`; deploy a fresh one
against a clean `broadcast/` and `SELECTOR_FACTORY` unset, or use the matching
revision.

**A dry run of step 2, 3 or 4 fails on a fresh chain**
Expected: they call `SelectorFactory`, which does not exist yet. Broadcast step 1
first, then dry-run the next one.
