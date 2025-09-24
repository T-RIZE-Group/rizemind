from typing import Any

from eth_account import Account
from eth_account.signers.base import BaseAccount
from flwr.common.context import Context
from mnemonic import Mnemonic
from pydantic import BaseModel, Field, field_validator, model_validator

from rizemind.configuration.base_config import BaseConfig
from rizemind.configuration.transform import unflatten
from rizemind.mnemonic.store import MnemonicStore

ACCOUNT_CONFIG_STATE_KEY = "rizemind.account"


class MnemonicStoreConfig(BaseModel):
    """A Pydantic model to store mnemonics."""

    account_name: str = Field(..., description="account name")
    passphrase: str = Field(..., description="Pass-phrase that unlocks the keystore")


class AccountConfig(BaseConfig):
    """Ethereum account configuration.

    Accepts **one** of two authentication sources: direct mnemonic string or a
    keystore reference. For example:

    1.  Direct mnemonic string
        ```toml
        [tool.eth.account]
        mnemonic = "test test … junk"
        ```

    2.  Keystore reference
        ```toml
        [tool.eth.account.mnemonic_store]
        account_name = "bob"
        passphrase   = "open sesame"
        ```

    Attributes:
        mnemonic: BIP-39 seed phrase. Leave empty if using mnemonic_store.
        mnemonic_store: Keystore configuration.
        default_account_index: The default HD wallet account index to use.
    """

    mnemonic: str | None = Field(
        default=None,
        description="BIP-39 seed phrase (leave empty if using mnemonic_store)",
    )

    mnemonic_store: MnemonicStoreConfig | None = None
    default_account_index: int | None = Field(default=None)

    @field_validator("mnemonic")
    @classmethod
    def _validate_mnemonic(cls, value: str) -> str:
        """Validate a mnemonic phrase."""
        mnemo = Mnemonic("english")
        if not mnemo.check(value):
            raise ValueError("Invalid mnemonic phrase")
        return value

    @model_validator(mode="after")
    def _populate_and_sanity_check(self) -> "AccountConfig":
        """Ensure exactly one variant is supplied.

        If the keystore path is chosen, decrypt it and populate ``self.mnemonic``.

        Raises:
            ValueError: If both or neither of ``mnemonic`` and ``mnemonic_store``
                are provided, or if the provided mnemonic does not match the one
                in the keystore.
        """

        if self.mnemonic_store is not None:
            store_conf = self.mnemonic_store
            store = MnemonicStore()
            mnemonic_stored = store.load(
                account_name=store_conf.account_name,
                passphrase=store_conf.passphrase,
            )
            if self.mnemonic is None:
                self.mnemonic = mnemonic_stored
            elif self.mnemonic != mnemonic_stored:
                raise ValueError(
                    "The mnemonic stored in the keystore does not match the mnemonic provided."
                )

        if self.mnemonic is None:
            raise ValueError(
                "You must supply either 'mnemonic' **or** 'mnemonic_store', but not both."
            )

        return self

    def get_account(self, i: int | None = None) -> BaseAccount:
        """Get the i-th HD wallet account.

        Args:
            i: The account index. If not supplied, the ``default_account_index``
                from the config will be used.

        Returns:
            The account.

        Raises:
            ValueError: If no account index is provided and no default is configured.
        """
        if i is None:
            i = self.default_account_index

        if i is None:
            raise ValueError(
                "no default_account_index specified, provide the index as an argument"
            )

        hd_path = f"m/44'/60'/0'/0/{i}"
        Account.enable_unaudited_hdwallet_features()
        return Account.from_mnemonic(self.mnemonic, account_path=hd_path)

    @staticmethod
    def from_context(context: Context) -> "AccountConfig | None":
        """Get the ``AccountConfig`` from the context if it exists.

        Args:
            context: The Flower context.

        Returns:
            The config if it exists, otherwise ``None``.
        """
        if ACCOUNT_CONFIG_STATE_KEY in context.state.config_records:
            records: Any = context.state.config_records[ACCOUNT_CONFIG_STATE_KEY]
            return AccountConfig(**unflatten(records))
        return None
