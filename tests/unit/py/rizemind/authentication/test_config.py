import pytest
from eth_account import Account
from eth_account.signers.local import LocalAccount
from pydantic import ValidationError
from rizemind.authentication.config import (
    AccountConfig,
    MnemonicStoreConfig,
)

VALID_MNEMONIC = "test test test test test test test test test test test junk"
INVALID_MNEMONIC = "foo bar baz qux"


def _derive_address(mnemonic: str, index: int) -> str:
    """Utility that derives an address directly via eth-account.

    This replicates the logic in `AccountConfig.get_account` so that we can
    assert the method is deterministic and correct without relying on the
    implementation under test.
    """
    Account.enable_unaudited_hdwallet_features()
    hd_path = f"m/44'/60'/0'/0/{index}"
    return Account.from_mnemonic(mnemonic, account_path=hd_path).address


class TestAccountConfig:
    """Unit-tests for :class:`AccountConfig`."""

    # --- Validators ---------------------------------------------------------
    def test_accepts_valid_mnemonic(self):
        cfg = AccountConfig(mnemonic=VALID_MNEMONIC)
        assert cfg.mnemonic == VALID_MNEMONIC

    def test_rejects_invalid_mnemonic(self):
        with pytest.raises(ValidationError):
            AccountConfig(mnemonic=INVALID_MNEMONIC)

    # --- get_account() ------------------------------------------------------
    @pytest.mark.parametrize("index", [0, 1, 2, 10])
    def test_get_account_returns_correct_account(self, index):
        cfg = AccountConfig(mnemonic=VALID_MNEMONIC)
        acct = cfg.get_account(index)

        # Returned type must be eth_account.signers.local.LocalAccount
        assert isinstance(acct, LocalAccount)

        # Address must match independent derivation for the same index
        expected_address = _derive_address(VALID_MNEMONIC, index)
        assert acct.address == expected_address

    def test_get_account_is_deterministic(self):
        cfg = AccountConfig(mnemonic=VALID_MNEMONIC)
        first = cfg.get_account(0)
        second = cfg.get_account(0)
        assert first.address == second.address

    def test_get_account_unique_indices_yield_unique_addresses(self):
        cfg = AccountConfig(mnemonic=VALID_MNEMONIC)
        addresses = {cfg.get_account(i).address for i in range(5)}
        # All five indices should map to five distinct addresses
        assert len(addresses) == 5

    def test_get_account_with_default_index(self):
        cfg = AccountConfig(mnemonic=VALID_MNEMONIC, default_account_index=2)
        account = cfg.get_account()
        expected_address = _derive_address(VALID_MNEMONIC, 2)
        assert account.address == expected_address

    def test_get_account_explicit_index_overrides_default(self):
        cfg = AccountConfig(mnemonic=VALID_MNEMONIC, default_account_index=2)
        account = cfg.get_account(1)  # Explicit index should override default
        expected_address = _derive_address(VALID_MNEMONIC, 1)
        assert account.address == expected_address

    def test_get_account_no_index_raises_error(self):
        cfg = AccountConfig(mnemonic=VALID_MNEMONIC)  # No default_account_index
        with pytest.raises(ValueError, match="no default_account_index specified"):
            cfg.get_account()  # Should raise when no index provided


class TestAccountConfigWithMnemonicStore:
    """Additional unit-tests for the [tool.eth.account.mnemonic_store] variant."""

    # -- fixtures ------------------------------------------------------------

    @pytest.fixture(autouse=True)
    def _patch_store_load(self, monkeypatch):
        """
        Autouse fixture patches MnemonicStore.load so the tests:
        • stay purely in-memory (no keystore files needed)
        • can assert the loader is called with the right args
        """

        def _fake_load(self, account_name: str, passphrase: str) -> str:
            assert account_name == "bob"
            assert passphrase == "open sesame"
            return VALID_MNEMONIC

        monkeypatch.setattr(
            "rizemind.mnemonic.store.MnemonicStore.load",
            _fake_load,
            raising=True,
        )

    # -- positive path -------------------------------------------------------

    def test_accepts_mnemonic_store(self):
        cfg = AccountConfig(
            mnemonic_store=MnemonicStoreConfig(
                account_name="bob",
                passphrase="open sesame",
            ),
        )
        # The mnemonic field should now hold the decrypted phrase
        assert cfg.mnemonic == VALID_MNEMONIC

    @pytest.mark.parametrize("index", [0, 3, 7])
    def test_get_account_matches_direct_derivation(self, index):
        cfg = AccountConfig(
            mnemonic_store=MnemonicStoreConfig(
                account_name="bob",
                passphrase="open sesame",
            ),
        )
        acct = cfg.get_account(index)
        assert isinstance(acct, LocalAccount)
        expected = _derive_address(VALID_MNEMONIC, index)
        assert acct.address == expected

    def test_both_sources_provides_same_mnemonic(self):
        cfg = AccountConfig(
            mnemonic=VALID_MNEMONIC,
            mnemonic_store=MnemonicStoreConfig(
                account_name="bob",
                passphrase="open sesame",
            ),
        )
        assert cfg.mnemonic == VALID_MNEMONIC

    # -- negative / edge cases ----------------------------------------------

    def test_both_sources_provided_is_rejected(self, monkeypatch):
        def _fake_load(self, account_name: str, passphrase: str) -> str:
            assert account_name == "bob"
            assert passphrase == "open sesame"
            return "junk test test test test test test test test test test junk"

        monkeypatch.setattr(
            "rizemind.mnemonic.store.MnemonicStore.load",
            _fake_load,
            raising=True,
        )

        with pytest.raises(ValidationError):
            AccountConfig(
                mnemonic=VALID_MNEMONIC,
                mnemonic_store=MnemonicStoreConfig(
                    account_name="bob",
                    passphrase="open sesame",
                ),
            )

    def test_neither_source_provided_is_rejected(self):
        with pytest.raises(ValidationError):
            AccountConfig()

    def test_store_load_failure_bubbles_up(self, monkeypatch):
        """Simulate a bad pass-phrase / corrupted keystore."""

        def _failing_load(self, account_name: str, passphrase: str):
            raise ValueError("Decryption failed")

        monkeypatch.setattr(
            "rizemind.mnemonic.store.MnemonicStore.load",
            _failing_load,
            raising=True,
        )

        with pytest.raises(ValidationError):
            AccountConfig(
                mnemonic_store=MnemonicStoreConfig(
                    account_name="bob",
                    passphrase="open sesame",
                ),
            )
