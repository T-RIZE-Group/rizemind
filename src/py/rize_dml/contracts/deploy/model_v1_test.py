from rize_dml.contracts.deploy.model_v1 import deploy_new_model_v1
from eth_account import Account

mnemonic = "test test test test test test test test test test test junk"
def test_deploy_new_model_v1_test():
    hd_path = f"m/44'/60'/0'/0/0"
    Account.enable_unaudited_hdwallet_features()
    account = Account.from_mnemonic(mnemonic, account_path=hd_path)
    model = deploy_new_model_v1(account, "Test", [account.address])
    assert model.functions.isAggregator(account.address).call() is True