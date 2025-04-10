from eth_account import Account
from rizemind.contracts.models.model_factory_v1 import (
    ModelFactoryV1,
    ModelFactoryV1Config,
)
from web3 import Web3

mnemonic = "test test test test test test test test test test test junk"


def test_model_v1_config_deploy():
    hd_path = "m/44'/60'/0'/0/0"
    Account.enable_unaudited_hdwallet_features()
    account = Account.from_mnemonic(mnemonic, account_path=hd_path)
    w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
    model_config = ModelFactoryV1Config(
        name="test",
        ticker="tst",
        local_factory_deployment_path="forge/output/31337/ModelRegistryFactory.json",
    )
    model_factory = ModelFactoryV1(model_config)
    model = model_factory.deploy(account, [account.address], w3)
    assert model.is_aggregator(account.address) is True
