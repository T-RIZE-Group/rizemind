from typing import Any, cast
from eth_typing import Address
from pydantic import BaseModel, Field
from rizemind.contracts.access_control.FlAccessControl import FlAccessControl
from rizemind.contracts.deployed_contracts import load_contract_data
from rizemind.contracts.models.model_registry import ModelRegistry
from web3 import Web3
from eth_account.signers.base import BaseAccount
from web3.contract import Contract
from eth_account.types import TransactionDictType


class ModelRegistryV1(FlAccessControl, ModelRegistry):
    account: BaseAccount
    w3: Web3

    def __init__(self, model: Contract, account: BaseAccount, w3: Web3):
        FlAccessControl.__init__(self, model)
        ModelRegistry.__init__(self, model)
        self.account = account
        self.w3 = w3

    def distribute(self, trainers: list[Address], contributions: list[Any]) -> bool:
        tx = self.model.functions.distribute(trainers, contributions).build_transaction(
            {
                "from": self.account.address,
                "nonce": self.w3.eth.get_transaction_count(self.account.address),
                "gas": 2000000,
                "gasPrice": self.w3.to_wei("20", "gwei"),
            }
        )
        signed_tx = self.account.sign_transaction(cast(TransactionDictType, tx))
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return tx_receipt["status"] == 0

    @staticmethod
    def from_address(address: str, account: BaseAccount, w3: Web3) -> "ModelRegistryV1":
        model = load_contract_data("ModelRegistryV1", "smart_contracts/output/local")
        checksum_address = Web3.to_checksum_address(address)
        return ModelRegistryV1(
            w3.eth.contract(address=checksum_address, abi=model.abi), account, w3
        )


class ModelV1Config(BaseModel):
    name: str = Field(..., description="The model name")
    ticker: str | None = Field(None, description="The ticker symbol of the model")

    def __init__(self, **data):
        super().__init__(**data)
        if self.ticker is None:
            self.ticker = self.name  # Default to name if ticker is not provided

    def deploy(self, deployer: BaseAccount, member_address: list[str], w3: Web3):
        factory_meta = load_contract_data(
            "ModelRegistryFactory", f"smart_contracts/output/{w3.eth.chain_id}"
        )
        factory = w3.eth.contract(
            abi=factory_meta.abi, address=cast(Address, factory_meta.address)
        )

        tx = factory.functions.createModel(
            self.name, self.ticker, deployer.address, member_address
        ).build_transaction(
            {
                "from": deployer.address,
                "nonce": w3.eth.get_transaction_count(deployer.address),
                "gas": 2000000,
                "gasPrice": w3.to_wei("20", "gwei"),
            }
        )

        signed_tx = deployer.sign_transaction(cast(TransactionDictType, tx))

        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)

        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        assert tx_receipt["status"] != 0, "Deployment transaction failed or reverted."

        event_signature = w3.keccak(
            text="ContractCreated(address,address,address)"
        ).hex()
        event_filter = factory.events.ContractCreated.create_filter(
            from_block=tx_receipt["blockNumber"],
            to_block=tx_receipt["blockNumber"],
            topics=[event_signature, Web3.to_hex(deployer.address.encode("utf-8"))],
        )
        logs = event_filter.get_all_entries()
        assert len(logs) == 1, "multiple instance started in the same block?"
        contract_created = logs[0]

        event_args = contract_created["args"]
        proxy_address = event_args["proxyAddress"]

        model = load_contract_data("ModelRegistryV1", "smart_contracts/output/local")

        return ModelRegistryV1(
            w3.eth.contract(address=proxy_address, abi=model.abi), deployer, w3
        )
