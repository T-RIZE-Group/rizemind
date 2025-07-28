from eth_account import Account
from eth_account.signers.local import LocalAccount
from mnemonic import Mnemonic
from pydantic import BaseModel, Field, field_validator, model_validator

from rizemind.mnemonic.store import MnemonicStore


class MnemonicStoreConfig(BaseModel):
    account_name: str = Field(..., description="account name")
    passphrase: str = Field(..., description="Pass-phrase that unlocks the keystore")


class AccountConfig(BaseModel):
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
        has_mnemonic = self.mnemonic is not None
        has_mnemonic_store = self.mnemonic_store is not None

        if has_mnemonic == has_mnemonic_store:
            raise ValueError(
                "You must supply either 'mnemonic' **or** 'mnemonic_store', but not both."
            )

        if self.mnemonic_store is not None:
            store_conf = self.mnemonic_store
            store = MnemonicStore()
            self.mnemonic = store.load(
                account_name=store_conf.account_name,
                passphrase=store_conf.passphrase,
            )

        return self

    def get_account(self, i: int) -> LocalAccount:
        hd_path = f"m/44'/60'/0'/0/{i}"
        Account.enable_unaudited_hdwallet_features()
        return Account.from_mnemonic(self.mnemonic, account_path=hd_path)
