import pytest
from rizemind.configuration.secrets import (
    REDACTED,
    is_secret_key,
    redact,
)

MNEMONIC = "test test test test test test test test test test test junk"


@pytest.mark.parametrize(
    "key",
    [
        "mnemonic",
        "passphrase",
        "password",
        "private_key",
        "secret",
        "seed_phrase",
        "token",
        "eth.account.mnemonic",
        "tool.eth.account.mnemonic_store.passphrase",
        "MNEMONIC",
        "Passphrase",
        "private-key",
        "rizenet_mnemonic",
        "aggregator_private_key",
    ],
)
def test_secret_keys_are_recognized(key):
    assert is_secret_key(key)


@pytest.mark.parametrize(
    "key",
    [
        "url",
        "mnemonic_store",
        "tool.eth.account.mnemonic_store",
        "account_name",
        "num-server-rounds",
        "web3.swarm.factory_v1.name",
        # `token` matches, but a key merely containing it must not.
        "tokenizer",
        "n_tokens",
    ],
)
def test_ordinary_keys_are_not_recognized(key):
    assert not is_secret_key(key)


def test_redacts_nested_layout():
    config = {
        "tool": {
            "eth": {"account": {"mnemonic": MNEMONIC}},
            "web3": {"url": "http://127.0.0.1:8545"},
        }
    }

    assert redact(config) == {
        "tool": {
            "eth": {"account": {"mnemonic": REDACTED}},
            "web3": {"url": "http://127.0.0.1:8545"},
        }
    }


def test_redacts_flat_dotted_layout():
    config = {"eth.account.mnemonic": MNEMONIC, "web3.url": "http://127.0.0.1:8545"}

    assert redact(config) == {
        "eth.account.mnemonic": REDACTED,
        "web3.url": "http://127.0.0.1:8545",
    }


def test_keystore_subtree_survives_but_passphrase_does_not():
    config = {"mnemonic_store": {"account_name": "alice", "passphrase": "open sesame"}}

    assert redact(config) == {
        "mnemonic_store": {"account_name": "alice", "passphrase": REDACTED}
    }


def test_secret_key_holding_a_mapping_loses_the_whole_subtree():
    config = {"secret": {"mnemonic": MNEMONIC, "note": "keep me"}}

    assert redact(config) == {"secret": REDACTED}


def test_redacts_inside_lists():
    config = {"accounts": [{"mnemonic": MNEMONIC}, {"url": "http://rpc"}]}

    assert redact(config) == {
        "accounts": [{"mnemonic": REDACTED}, {"url": "http://rpc"}]
    }


def test_non_secret_values_are_untouched():
    config = {
        "num-server-rounds": 3,
        "fraction-fit": 0.5,
        "verbose": False,
        "missing": None,
        "tags": ["a", "b"],
    }

    assert redact(config) == config


def test_input_is_not_modified():
    config = {"eth": {"account": {"mnemonic": MNEMONIC}}}

    redact(config)

    assert config["eth"]["account"]["mnemonic"] == MNEMONIC


def test_placeholder_is_configurable():
    config = {"mnemonic": MNEMONIC}

    assert redact(config, placeholder="<hidden>") == {"mnemonic": "<hidden>"}
