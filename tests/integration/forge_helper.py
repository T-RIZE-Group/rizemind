"""forge_helper.py
Utility helpers for integration-testing Solidity contracts from Python while keeping
Foundry (forge + anvil) as the single source-of-truth for compilation and
deployment.

start_anvil(...)
    Launch a local Anvil JSON-RPC node in the background.

deploy_contract(...)
    Deploy a single contract with ``forge create`` and return the on-chain address.

run_script(...)
    Execute a Forge script (e.g. `script/Deploy.s.sol`) and parse the JSON
    summary so you can wire the deployed addresses into your tests.
"""

import json
import os
import shutil
import subprocess
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

from eth_typing import ChecksumAddress
from rizemind.contracts.deployment import DeployedContract
from web3 import Web3

_CMD_TIMEOUT = 60  # seconds for forge/anvil invocations

FORGE_DIR = Path(os.path.dirname(__file__)).parent.parent / "forge"


def _check_tool_installed(cmd: str) -> None:
    if shutil.which(cmd) is None:
        raise RuntimeError(
            f"Required tool '{cmd}' is not on PATH; install Foundry: https://book.getfoundry.sh/"
        )


def _run(
    cmd: Sequence[str], *, cwd: str | None = None, env: dict[str, str] | None = None
) -> str:
    """Run *cmd* and return *stdout*.

    Parameters
    ----------
    cmd:
        Command and arguments to execute.
    cwd:
        Working directory for the command.
    env:
        Environment variables to pass to the command. If None, inherits from parent process.

    Raises
    ------
    RuntimeError
        If the command exits with a non-zero status.
    """
    # Merge provided env with current environment

    os_env = os.environ.copy()
    if env is not None:
        os_env.update(env)

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd,
        env=os_env,
        timeout=_CMD_TIMEOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}\n{result.stdout}"
        )
    return result.stdout


def start_anvil(
    *,
    mnemonic: str = ("test test test test test test test test test test test junk"),
    chain_id: int = 31337,
    port: int = 8545,
    extra_args: Sequence[str] | None = None,
    wait: int = 2,
) -> subprocess.Popen[str]:
    """Launch Anvil in the background and return the *Popen* handle.

    The caller is responsible for terminating the process (``proc.terminate()``).

    Parameters
    ----------
    mnemonic:
        Seed phrase used to generate deterministic accounts. Change this if your
        tests rely on bespoke private keys.
    chain_id:
        EVM chainId exposed by the node.
    port:
        TCP port for the JSON-RPC endpoint.
    extra_args:
        Additional CLI flags forwarded to ``anvil`` (e.g. ``["--code-size-limit","infinite"]``).
    wait:
        Seconds to sleep after spawning so Anvil finishes booting.
    """
    _check_tool_installed("anvil")
    cmd = [
        "anvil",
        "--mnemonic",
        mnemonic,
        "--chain-id",
        str(chain_id),
        "--port",
        str(port),
        "--block-time",
        "1",
    ]
    if extra_args:
        cmd.extend(extra_args)
    proc: subprocess.Popen[str] = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    w3 = Web3(Web3.HTTPProvider(f"http://127.0.0.1:{port}"))

    for _ in range(wait * 10):
        if w3.is_connected():
            break
        time.sleep(0.1)
    else:
        proc.terminate()
        raise RuntimeError("Anvil didn't start")
    if proc.poll() is not None:
        raise RuntimeError(
            f"Anvil failed to start. Output:\n{proc.stdout.read() if proc.stdout else ''}"
        )
    return proc


def deploy_contract(
    contract_path: str | Path,
    contract_name: str,
    *,
    account: ChecksumAddress,
    rpc_url: str = "http://127.0.0.1:8545",
    constructor_args: Iterable[str] | None = None,
    forge_project_root: str | Path | None = FORGE_DIR,
) -> DeployedContract:
    """Deploy *contract_name* found at *contract_path* using ``forge create``.

    Returns a dictionary with at least ``address`` and ``txHash`` as reported by Forge.

    Examples
    --------
    >>> deploy_contract(
    ...     "src/MyToken.sol", "MyToken", constructor_args=["My Token", "MT"]
    ... )
    {'address': '0x1234...abcd', 'txHash': '0xabcd...'}
    """
    _check_tool_installed("forge")

    cmd: list[str] = [
        "forge",
        "create",
        "--rpc-url",
        rpc_url,
        "--json",  # machine-readable summary
        "--broadcast",
        "--unlocked",
        "--from",
        f"{account}",
        f"{contract_path}:{contract_name}",
    ]
    if constructor_args:
        cmd.extend(["--constructor-args", *map(str, constructor_args)])

    stdout = _run(cmd, cwd=str(forge_project_root) if forge_project_root else None)
    print(" ".join(cmd))
    # ``forge create`` may emit multiple JSON blobs; the last is the summary we need.
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Unable to parse forge output as JSON. Full output:\n{stdout}"
        ) from e

    return DeployedContract(address=data["deployedTo"])


def run_script(
    script_path: str | Path,
    *,
    rpc_url: str = "http://127.0.0.1:8545",
    account: ChecksumAddress,
    forge_project_root: str | Path = FORGE_DIR,
    env: dict[str, str] | None = None,
) -> Path:
    """Execute a Forge script and return a mapping of *contractName → deployed address*.

    Requires the script to emit a JSON summary (use ``forge script --json``).

    Examples
    --------
    >>> run_script("script/Deploy.s.sol", rpc_url="http://127.0.0.1:8545")

    """
    _check_tool_installed("forge")
    cmd: list[str] = [
        "forge",
        "script",
        str(script_path),
        "--rpc-url",
        rpc_url,
        "--broadcast",
        "--quiet",
        "--unlocked",
        "--sender",
        f"{account}",
    ]

    try:
        _run(cmd, cwd=str(forge_project_root) if forge_project_root else None, env=env)
    except Exception as e:
        raise ValueError(f"Failed to run{' '.join(cmd)}") from e

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    chain_id = w3.eth.chain_id

    filename = Path(script_path).name.split(":")[0]
    return (
        Path(forge_project_root)
        / "broadcast"
        / filename
        / str(chain_id)
        / "run-latest.json"
    )


__all__ = [
    "start_anvil",
    "deploy_contract",
    "run_script",
]
