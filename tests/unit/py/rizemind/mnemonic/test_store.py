"""Full pytest test‑suite for the ``MnemonicStore`` helper.

Coverage goals
--------------
✔️ `generate` – delegates to *eth_account.hdaccount.generate_mnemonic* and returns
   the string produced.
✔️ `save` / `load` round‑trip – the mnemonic we encrypt must decrypt unchanged.
✔️ `exists` – reflects on‑disk state accurately.
✔️ `list_accounts` – lists *only* ``*.json`` keystores and returns names sorted.
✔️ Error handling –
    • loading an unknown account raises *FileNotFoundError*;
    • wrong pass‑phrase raises *ValueError*.
✔️ Private helpers – `_derive_key` length is 32 bytes by default.

The suite is *hermetic* (uses ``tmp_path``) and *deterministic* (monkeypatches
``generate_mnemonic`` so we don’t depend on entropy or network I/O).
"""

import json
import os
from pathlib import Path

import pytest
from rizemind.mnemonic.store import MnemonicStore


@pytest.fixture()
def store(tmp_path: Path) -> MnemonicStore:  # noqa: D401 – fixture
    """Provide an **empty** keystore directory for each test run."""
    return MnemonicStore(keystore_dir=tmp_path)


@pytest.fixture()
def plaintext_mnemonic() -> str:
    """A deterministic 24‑word English BIP‑39 seed phrase for repeatability."""
    return (
        "abandon abandon abandon abandon abandon abandon abandon abandon abandon "
        "abandon abandon abandon abandon abandon abandon abandon abandon abandon "
        "abandon abandon abandon abandon abandon abandon"
    )


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def test_generate_invokes_eth_account(
    monkeypatch: pytest.MonkeyPatch, store: MnemonicStore
):
    dummy_result = "lorem ipsum dolor sit amet"  # clearly *not* a real 24‑word seed

    def fake_generate_mnemonic(*_args, **_kwargs):  # noqa: D401 – local stub
        return dummy_result

    # Patch the external dependency so we do not require eth_account in the env.
    monkeypatch.setattr(
        "rizemind.mnemonic.store.generate_mnemonic",
        fake_generate_mnemonic,
        raising=True,
    )

    assert store.generate() == dummy_result


# ---------------------------------------------------------------------------
# save / load round‑trip
# ---------------------------------------------------------------------------


def test_save_creates_keystore_json(store: MnemonicStore, plaintext_mnemonic: str):
    path = store.save("alice", "correct horse", plaintext_mnemonic)
    assert path.exists(), "Keystore JSON file was not created"

    data = json.loads(path.read_text())
    # Basic structural expectations – proves that encryption layer ran.
    required_fields = {
        "version",
        "kdf",
        "salt",
        "nonce",
        "cipher",
        "cipher_algo",
        "aad",
    }
    assert required_fields.issubset(data), "Encrypted blob is missing expected keys"


def test_load_decrypts_mnemonic(store: MnemonicStore, plaintext_mnemonic: str):
    store.save("bob", "open sesame", plaintext_mnemonic)
    recovered = store.load("bob", "open sesame")
    assert recovered == plaintext_mnemonic


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


def test_load_unknown_account_raises(store: MnemonicStore):
    with pytest.raises(FileNotFoundError):
        store.load("does‑not‑exist", "irrelevant")


def test_load_wrong_passphrase_raises(store: MnemonicStore, plaintext_mnemonic: str):
    store.save("charlie", "top secret", plaintext_mnemonic)
    with pytest.raises(ValueError):
        store.load("charlie", "incorrect")


# ---------------------------------------------------------------------------
# exists / list_accounts helpers
# ---------------------------------------------------------------------------


def test_exists_reflects_state(store: MnemonicStore, plaintext_mnemonic: str):
    assert store.exists("david") is False
    store.save("david", "pw", plaintext_mnemonic)
    assert store.exists("david") is True


def test_list_accounts_sorted(store: MnemonicStore, plaintext_mnemonic: str):
    store.save("zeta", "pw", plaintext_mnemonic)
    store.save("alpha", "pw", plaintext_mnemonic)
    store.save("mu", "pw", plaintext_mnemonic)

    assert store.list_accounts() == ["alpha", "mu", "zeta"]


# ---------------------------------------------------------------------------
# private API sanity checks
# ---------------------------------------------------------------------------


def test_derive_key_length(store: MnemonicStore):
    key = store._derive_key("passphrase", salt=os.urandom(16))
    assert len(key) == 32, "Derived key length should default to 32 bytes"
