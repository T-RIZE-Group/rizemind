from typing import Any
from eth_account import Account
from eth_typing import Address
from rize_dml.contracts.access_control.FlAccessControl import FlAccessControl
from rize_dml.contracts.deployed_contracts import load_contract_data
from rize_dml.contracts.models.model_registry import ModelRegistry
from web3 import Web3
from web3.contract import Contract


class ModelRegistryV1(FlAccessControl, ModelRegistry):
    account: Account

    def __init__(self, model: Contract, account: Account):
        FlAccessControl.__init__(self, model)
        ModelRegistry.__init__(self, model)
        self.account = account

    def distribute(self, trainers: list[Address], contributions: list[Any]) -> bool:
        w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
        tx = self.model.functions.distribute(trainers, contributions).build_transaction(
            {
                "from": self.account.address,
                "nonce": w3.eth.get_transaction_count(self.account.address),
                "gas": 2000000,
                "gasPrice": w3.to_wei("20", "gwei"),
            }
        )
        signed_tx = self.account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        return tx_receipt.status == 0  # type: ignore

    @staticmethod
    def from_address(address: str, account: Account) -> "ModelRegistryV1":
        model = load_contract_data("ModelRegistryV1", "smart_contracts/output/local")
        checksum_address = Web3.to_checksum_address(address)
        w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
        return ModelRegistryV1(
            w3.eth.contract(address=checksum_address, abi=model.abi), account
        )
