import tomli
from mnemonic import Mnemonic
from eth_account import Account
from eth_account.signers.local import LocalAccount


def load_auth_config(config_path):
    mnemo = Mnemonic("english")
    with open(config_path, "rb") as f:
        toml_dict = tomli.load(f)
        web3_config = toml_dict.get("tool", {}).get("web3", {})

    mnemonic_phrase = web3_config.get("mnemonic")
    if not mnemo.check(mnemonic_phrase):
        raise ValueError("Invalid mnemonic phrase")

    return SimulationConfig(
        web3_config.get("mnemonic"),
        web3_config.get("chainid"),
        web3_config.get("contract"),
        web3_config.get("name"),
    )


class SimulationConfig:
    mnemonic: str
    chainid: int
    contract: str
    name: str

    def __init__(self, mnemonic: str, chainid: int, contract: str, name: str):
        self.mnemonic = mnemonic
        self.chainid = chainid
        self.contract = contract
        self.name = name

    def get_account(self, i: int) -> LocalAccount:
        hd_path = f"m/44'/60'/{i}'/0/0"
        Account.enable_unaudited_hdwallet_features()
        return Account.from_mnemonic(self.mnemonic, account_path=hd_path)
