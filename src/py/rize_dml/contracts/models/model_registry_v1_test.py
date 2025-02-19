from eth_account import Account

from rize_dml.contracts.models.model_registry_v1 import ModelV1Config
from web3 import Web3

mnemonic = "test test test test test test test test test test test junk"


def test_model_v1_config_deploy():
    hd_path = "m/44'/60'/0'/0/0"
    Account.enable_unaudited_hdwallet_features()
    account = Account.from_mnemonic(mnemonic, account_path=hd_path)
    w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
    model_config = ModelV1Config(name="test", ticker="tst")
    model = model_config.deploy(account, [account.address], w3)
    assert model.is_aggregator(account.address) is True
