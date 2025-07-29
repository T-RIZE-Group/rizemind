from pathlib import Path

import pytest
from rizemind.contracts.local_deployment import load_forge_artifact

ARTIFACT_PATH = Path(__file__).parent / "forge_artifact.json"


def test_load_forge_artifact_success():
    address = load_forge_artifact(ARTIFACT_PATH, "SwarmV1")
    assert address.address == "0x5FbDB2315678afecb367f032d93F642f64180aa3"
    address_factory = load_forge_artifact(ARTIFACT_PATH, "SwarmV1Factory")
    assert address_factory.address == "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"


def test_load_forge_artifact_not_found():
    with pytest.raises(ValueError, match="Contract 'NonExistent' not found"):
        load_forge_artifact(ARTIFACT_PATH, "NonExistent")


def test_load_forge_artifact_file_not_found():
    path = Path("/tmp/nonexistent_file.json")
    with pytest.raises(FileNotFoundError):
        load_forge_artifact(path, "SwarmV1")
