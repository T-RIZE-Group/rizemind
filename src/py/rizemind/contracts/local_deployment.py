import json
from pathlib import Path

from web3 import Web3

from rizemind.contracts.deployment import DeployedContract


def load_forge_artifact(path: Path, contract_name: str) -> DeployedContract:
    """
    Load a Forge broadcast artifact and return the contract address for the given contract name.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path) as f:
        data = json.load(f)
    for tx in data.get("transactions", []):
        if tx.get("contractName") == contract_name:
            return DeployedContract(
                address=Web3.to_checksum_address(tx.get("contractAddress"))
            )
    raise ValueError(f"Contract '{contract_name}' not found in artifact: {path}")
