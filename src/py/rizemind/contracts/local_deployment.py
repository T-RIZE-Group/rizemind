from pathlib import Path
import json

from rizemind.contracts.deployment import DeployedContract


def load_local_deployment(path="../smart_contracts") -> DeployedContract:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r") as f:
        file_data = json.load(f)

    return DeployedContract.model_validate(file_data)
