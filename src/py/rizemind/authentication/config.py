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
    account_name: str = Field(..., description="account name")
    passphrase: str = Field(..., description="Pass-phrase that unlocks the keystore")


class AccountConfig(BaseConfig):
    """
    Accept **one** of the two authentication sources:

    1.  Direct mnemonic string

        [tool.eth.account]
        mnemonic = "test test … junk"

    2.  Keystore reference

        [tool.eth.account.mnemonic_store]
        account_name = "bob"
        passphrase   = "open sesame"
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
        mnemo = Mnemonic("english")
        if not mnemo.check(value):
            raise ValueError("Invalid mnemonic phrase")
        return value

    @model_validator(mode="after")
    def _populate_and_sanity_check(self) -> "AccountConfig":
        """
        • Ensure *exactly one* variant is supplied
        • If the keystore path is chosen, decrypt it and populate ``self.mnemonic``
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
        if ACCOUNT_CONFIG_STATE_KEY in context.state.config_records:
            records: Any = context.state.config_records[ACCOUNT_CONFIG_STATE_KEY]
            return AccountConfig(**unflatten(records))
        return None
