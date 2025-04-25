from eth_account import Account
from rizemind.contracts.models.model_factory_v1 import (
    ModelFactoryV1,
    ModelFactoryV1Config,
)
from web3 import Web3

mnemonic = "test test test test test test test test test test test junk"


def test_model_v1_config_deploy():
    Account.enable_unaudited_hdwallet_features()
    aggregator = Account.from_mnemonic(mnemonic, account_path="m/44'/60'/0'/0/0")
    w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
    model_config = ModelFactoryV1Config(
        name="test",
        ticker="tst",
        local_factory_deployment_path="forge/output/31337/ModelRegistryFactory.json",
    )
    model_factory = ModelFactoryV1(model_config)
    model = model_factory.deploy(aggregator, [aggregator.address], w3)
    trainer1 = Account.from_mnemonic(mnemonic, account_path="m/44'/60'/1'/0/0")
    trainer2 = Account.from_mnemonic(mnemonic, account_path="m/44'/60'/2'/0/0")
    trainer3 = Account.from_mnemonic(mnemonic, account_path="m/44'/60'/3'/0/0")
    model.distribute([(trainer1.address, 1.0), (trainer2.address, 2.0)])
    model.distribute([(trainer2.address, 3.0), (trainer3.address, 4.0)])

    trainer1_latest_contribution = model.get_latest_contribution(trainer1.address)
    assert trainer1_latest_contribution == 1.0

    trainer2_latest_contribution = model.get_latest_contribution(trainer2.address)
    assert trainer2_latest_contribution == 3.0

    trainer_3_latest_contritbution = model.get_latest_contribution(trainer3.address)
    assert trainer_3_latest_contritbution == 4.0

    trainer4 = Account.from_mnemonic(mnemonic, account_path="m/44'/60'/4'/0/0")
    trainer4_latest_contribution = model.get_latest_contribution(trainer4.address)
    assert trainer4_latest_contribution is None
