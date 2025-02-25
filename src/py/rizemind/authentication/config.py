from mnemonic import Mnemonic
from eth_account import Account
from eth_account.signers.local import LocalAccount

from pydantic import BaseModel, Field, field_validator


class AccountConfig(BaseModel):
    mnemonic: str = Field(..., description="A valid BIP-39 mnemonic phrase")

    @field_validator("mnemonic")
    @classmethod
    def validate_mnemonic(cls, value: str) -> str:
        mnemo = Mnemonic("english")
        if not mnemo.check(value):
            raise ValueError("Invalid mnemonic phrase")
        return value

    def get_account(self, i: int) -> LocalAccount:
        hd_path = f"m/44'/60'/{i}'/0/0"
        Account.enable_unaudited_hdwallet_features()
        return Account.from_mnemonic(self.mnemonic, account_path=hd_path)
