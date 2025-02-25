from typing import Optional, cast
from pydantic import BaseModel, Field
from eth_account.signers.base import BaseAccount
from rizemind.contracts.deployment import DeployedContract
from rizemind.contracts.local_deployment import load_local_deployment
from rizemind.contracts.models.model_registry_v1 import ModelRegistryV1
from rizemind.contracts.abi.model_factory_v1 import model_factory_v1_abi
from rizemind.web3.chains import RIZENET_TESTNET_CHAINID
from web3 import Web3
from eth_account.types import TransactionDictType


class ModelFactoryV1Config(BaseModel):
    name: str = Field(..., description="The model name")
    ticker: Optional[str] = Field(None, description="The ticker symbol of the model")
    local_factory_deployment_path: Optional[str] = Field(
        None, description="path to local deployments"
    )

    factory_deployments: dict[int, DeployedContract] = {
        RIZENET_TESTNET_CHAINID: DeployedContract(
            address=Web3.to_checksum_address(
                "0xB88D434B10f0bB783A826bC346396AbB19B6C6F7"
            ),
            abi=model_factory_v1_abi,
        )
    }

    def __init__(self, **data):
        super().__init__(**data)
        if self.ticker is None:
            self.ticker = self.name  # Default to name if ticker is not provided

    def get_factory_deployment(self, chain_id: int) -> DeployedContract:
        if chain_id in self.factory_deployments:
            return self.factory_deployments[chain_id]
        elif self.local_factory_deployment_path is not None:
            return load_local_deployment(self.local_factory_deployment_path)
        raise Exception(
            f"Chain ID#{chain_id} is unsupported, provide a local_deployment_path"
        )


class ModelFactoryV1:
    config: ModelFactoryV1Config

    def __init__(self, config: ModelFactoryV1Config):
        self.config = config

    def deploy(self, deployer: BaseAccount, member_address: list[str], w3: Web3):
        factory_meta = self.config.get_factory_deployment(w3.eth.chain_id)
        factory = w3.eth.contract(
            abi=factory_meta.abi, address=factory_meta.address_as_bytes()
        )

        tx = factory.functions.createModel(
            self.config.name, self.config.ticker, deployer.address, member_address
        ).build_transaction(
            {
                "from": deployer.address,
                "nonce": w3.eth.get_transaction_count(deployer.address),
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

        return ModelRegistryV1.from_address(proxy_address, deployer, w3)
