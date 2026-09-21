"""Pre-flight checks for the Arc mainnet example.

Everything this example does on-chain costs real USDC, so run this first:

    uv run -- python preflight.py

It reads the very same ``pyproject.toml`` the run does, then verifies — without
sending a transaction — that the RPC is the chain you think it is, that the
``SwarmV1Factory`` is deployed where the library expects it, that the aggregator
can pay, and that ``createSwarm`` would actually succeed.

``--chain-id`` overrides ``expected-chain-id`` for one invocation, mirroring
``flwr run . --run-config expected-chain-id=…``, so pointing at another Arc
network is an argument rather than an edit.

Nothing here prints a mnemonic, a private key, or the keystore passphrase.
"""

import argparse
import os
import sys
from typing import Any

from eth_account import Account
from rizemind.authentication.config import AccountConfig
from rizemind.configuration.toml_config import TomlConfig
from rizemind.contracts.swarm.swarm_v1.swarm_v1_factory import (
    SwarmV1FactoryConfig,
)
from rizemind.contracts.swarm.swarm_v1.swarm_v1_factory import (
    abi as factory_abi,
)
from rizemind.web3 import Web3Config
from rizemind.web3.config import poaChains
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

# `createSwarm` is roughly this expensive; used only to size the balance warning
# when gas estimation is unavailable.
CREATE_SWARM_GAS_HINT = 1_000_000

failures: list[str] = []
warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  \033[32mok\033[0m    {msg}")


def warn(msg: str) -> None:
    warnings.append(msg)
    print(f"  \033[33mwarn\033[0m  {msg}")


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"  \033[31mFAIL\033[0m  {msg}")


def section(title: str) -> None:
    print(f"\n{title}")


def usdc(wei: int) -> str:
    """Format a native-view balance.

    Arc's *native* view of USDC — ``eth_getBalance``, gas prices, ``msg.value`` —
    uses 18 decimals, exactly like ether. The ERC-20 precompile uses 6 and is a
    different view of the same balance; don't mix the two.
    """
    return f"{Web3.from_wei(wei, 'ether'):.6f} USDC"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chain-id",
        type=int,
        default=None,
        help="override `expected-chain-id` from pyproject.toml for this run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    Account.enable_unaudited_hdwallet_features()

    section("Config")
    config = TomlConfig("./pyproject.toml")
    run_config = config.get("tool.flwr.app.config")
    expected_chain_id = args.chain_id or int(run_config["expected-chain-id"])
    url = config.get("tool.web3.url")
    if isinstance(url, str) and url.startswith("$"):
        fail(f"{url} is not set in the environment; export it and re-run")
        return report()
    source = "--chain-id" if args.chain_id else "pyproject.toml"
    ok(f"expected chain id: {expected_chain_id} (from {source})")

    try:
        web3_config = Web3Config(**config.get("tool.web3"))
    except Exception as exc:  # noqa: BLE001 - surfaced to the operator verbatim
        fail(f"tool.web3 is invalid: {exc}")
        return report()

    section("RPC")
    w3 = Web3(web3_config.web3_provider())
    try:
        chain_id = w3.eth.chain_id
    except Exception as exc:  # noqa: BLE001
        fail(f"cannot reach {url}: {exc}")
        return report()

    if chain_id == expected_chain_id:
        ok(f"{url} reports chain id {chain_id}")
    else:
        fail(
            f"{url} reports chain id {chain_id}, expected {expected_chain_id} — "
            "stop and fix the endpoint before spending anything"
        )
        return report()

    # PoA probe. Ask the provider directly so web3's own block validation cannot
    # raise before we get to look at the field.
    raw: dict[str, Any] = w3.provider.make_request(
        "eth_getBlockByNumber", ["latest", False]
    )
    extra_data = raw.get("result", {}).get("extraData", "0x")
    extra_len = len(bytes.fromhex(extra_data[2:]))
    needs_poa = extra_len > 32
    registered_poa = chain_id in poaChains
    if needs_poa:
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        if registered_poa:
            ok(f"extraData is {extra_len} bytes; chain is in `poaChains`")
        else:
            fail(
                f"extraData is {extra_len} bytes, so web3 needs the PoA middleware, "
                f"but chain {chain_id} is not in `poaChains` "
                "(src/py/rizemind/web3/config.py) — the run will raise "
                "ExtraDataLengthError"
            )
    else:
        if registered_poa:
            warn(
                f"extraData is {extra_len} bytes (no PoA middleware needed), but "
                f"chain {chain_id} is listed in `poaChains`"
            )
        else:
            ok(f"extraData is {extra_len} bytes; no PoA middleware needed")

    section("Factory")
    factory_config = SwarmV1FactoryConfig(**config.get("tool.web3.swarm.factory_v1"))
    try:
        deployment = factory_config.get_factory_deployment(chain_id)
    except Exception as exc:  # noqa: BLE001
        fail(str(exc))
        return report()

    factory_address = Web3.to_checksum_address(deployment.address)
    code = w3.eth.get_code(factory_address)
    if len(code) == 0:
        fail(f"no contract code at {factory_address} on chain {chain_id}")
        return report()
    ok(f"SwarmV1Factory at {factory_address} ({len(code)} bytes of code)")

    factory = w3.eth.contract(address=factory_address, abi=factory_abi)
    implementation = factory.functions.getImplementation().call()
    ok(f"swarm implementation: {implementation}")

    section("Accounts")
    try:
        account_config = AccountConfig(**config.get("tool.eth.account"))
    except Exception as exc:  # noqa: BLE001
        fail(f"cannot load the account: {exc}")
        return report()

    aggregator = account_config.get_account(0)
    num_supernodes = int(run_config["num-supernodes"])
    trainers = [
        account_config.get_account(i).address for i in range(1, num_supernodes + 1)
    ]
    ok(f"aggregator: {aggregator.address}")
    for i, trainer in enumerate(trainers, start=1):
        ok(f"trainer {i}:  {trainer} (signs off-chain, needs no USDC)")

    federation_supernodes = int(
        config.get("tool.flwr.federations.local-simulation.options.num-supernodes")
    )
    if federation_supernodes != num_supernodes:
        fail(
            f"num-supernodes is {num_supernodes} in [tool.flwr.app.config] but "
            f"{federation_supernodes} in the federation: the swarm's member list "
            "would not match the clients that connect"
        )

    section("Cost")
    balance = w3.eth.get_balance(aggregator.address)
    gas_price = w3.eth.gas_price
    if balance == 0:
        fail(f"aggregator {aggregator.address} has no balance; fund it before running")
    else:
        ok(f"aggregator balance: {usdc(balance)}")
    ok(f"gas price: {gas_price} wei")

    # Dry-run the real thing. This is the check that matters: it resolves both
    # selectors through the SelectorFactory and reverts if either version is
    # unregistered on this chain.
    salt = os.urandom(32)
    trainer_params = factory_config.trainer_selector.get_selector_params()
    evaluator_params = factory_config.evaluator_selector.get_selector_params()
    swarm_params = {
        "swarm": {
            "name": factory_config.name,
            "symbol": factory_config.ticker,
            "aggregator": aggregator.address,
            "trainers": trainers,
        },
        "trainerSelector": {
            "id": trainer_params.id,
            "initData": trainer_params.init_data,
        },
        "evaluatorSelector": {
            "id": evaluator_params.id,
            "initData": evaluator_params.init_data,
        },
    }
    call = factory.functions.createSwarm(salt, swarm_params)
    gas = CREATE_SWARM_GAS_HINT
    dry_run_params = {"from": aggregator.address}
    try:
        predicted = call.call(dry_run_params)
        ok(f"createSwarm dry run succeeds; swarm would be {predicted}")
    except Exception as exc:  # noqa: BLE001
        fail(
            f"createSwarm would not succeed: {exc}\n"
            "        A revert means the selector version this config asks for is "
            "not registered on this chain (`SelectorImplementationNotFound`); some "
            "nodes also reject the call outright when the sender cannot pay."
        )

    if balance == 0:
        # Most nodes refuse to estimate for an account that cannot pay.
        warn(f"skipping gas estimation; using a {CREATE_SWARM_GAS_HINT:,} gas hint")
    else:
        try:
            gas = call.estimate_gas(dry_run_params)
            ok(f"createSwarm gas estimate: {gas:,} ({usdc(gas * gas_price)})")
        except Exception as exc:  # noqa: BLE001
            warn(f"cannot estimate createSwarm gas ({exc}); using a hint instead")

    # One createSwarm plus two aggregator transactions per round.
    rounds = int(run_config["num-server-rounds"])
    projected = (gas + rounds * 2 * gas) * gas_price
    if balance == 0:
        pass  # already reported above
    elif balance < projected:
        warn(
            f"balance {usdc(balance)} is below the rough projection "
            f"{usdc(projected)} for {rounds} round(s)"
        )
    else:
        ok(f"balance covers the rough projection of {usdc(projected)}")

    return report()


def report() -> int:
    print()
    if failures:
        print(f"\033[31m{len(failures)} check(s) failed — do not run yet.\033[0m")
        return 1
    if warnings:
        print(f"\033[33mAll checks passed with {len(warnings)} warning(s).\033[0m")
        return 0
    print("\033[32mAll checks passed.\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
