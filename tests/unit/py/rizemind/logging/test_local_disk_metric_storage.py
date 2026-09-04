import json
from pathlib import Path

import pytest
from rizemind.configuration.secrets import REDACTED
from rizemind.logging.local_disk_metric_storage import LocalDiskMetricStorage

MNEMONIC = "test test test test test test test test test test test junk"
PASSPHRASE = "open sesame"


@pytest.fixture
def storage(tmp_path: Path) -> LocalDiskMetricStorage:
    return LocalDiskMetricStorage(tmp_path, "test-app")


def config_text(storage: LocalDiskMetricStorage) -> str:
    return storage.config_file.read_text()


def test_writes_ordinary_config(storage: LocalDiskMetricStorage):
    storage.write_config({"num-server-rounds": 3, "metrics-storage-path": "logs"})

    written = json.loads(config_text(storage))

    assert written[0]["num-server-rounds"] == 3
    assert written[0]["metrics-storage-path"] == "logs"


def test_flat_mnemonic_never_reaches_disk(storage: LocalDiskMetricStorage):
    storage.write_config({"eth.account.mnemonic": MNEMONIC, "web3.url": "http://rpc"})

    text = config_text(storage)

    assert MNEMONIC not in text
    assert REDACTED in text
    assert "http://rpc" in text


def test_nested_credentials_never_reach_disk(storage: LocalDiskMetricStorage):
    storage.write_config(
        {
            "eth": {
                "account": {
                    "mnemonic": MNEMONIC,
                    "mnemonic_store": {
                        "account_name": "alice",
                        "passphrase": PASSPHRASE,
                    },
                }
            },
            "web3": {"url": "http://127.0.0.1:8545"},
        }
    )

    text = config_text(storage)

    assert MNEMONIC not in text
    assert PASSPHRASE not in text
    # The non-secret siblings are still recorded.
    assert "alice" in text
    assert "http://127.0.0.1:8545" in text


def test_merges_successive_writes_and_redacts_each(storage: LocalDiskMetricStorage):
    storage.write_config({"num-server-rounds": 2})
    storage.write_config({"web3": {"url": "http://rpc"}, "mnemonic": MNEMONIC})

    text = config_text(storage)
    written = json.loads(text)

    assert MNEMONIC not in text
    assert len(written) == 2
    assert written[0]["num-server-rounds"] == 2


def test_caller_config_is_not_mutated(storage: LocalDiskMetricStorage):
    config = {"eth": {"account": {"mnemonic": MNEMONIC}}}

    storage.write_config(config)

    assert config["eth"]["account"]["mnemonic"] == MNEMONIC
