from logging import INFO
from typing import Optional, cast

from eth_account.signers.base import BaseAccount
from eth_account.types import TransactionDictType
from flwr.common.logger import log
from pydantic import BaseModel, Field
from rizemind.contracts.abi.model_factory_v1 import model_factory_v1_abi
from rizemind.contracts.deployment import DeployedContract
from rizemind.contracts.local_deployment import load_local_deployment
from rizemind.contracts.models.model_meta_v1 import ModelMetaV1
from rizemind.web3.chains import RIZENET_TESTNET_CHAINID
from web3 import Web3


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
            )
        )
    }

    def __init__(self, **data):
        super().__init__(**data)
        if self.ticker is None:
            self.ticker = self.name  # Default to name if ticker is not provided

    def get_factory_deployment(self, chain_id: int) -> DeployedContract:
        if self.local_factory_deployment_path is not None:
            return load_local_deployment(self.local_factory_deployment_path)
        if chain_id in self.factory_deployments:
            return self.factory_deployments[chain_id]
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
            abi=model_factory_v1_abi, address=factory_meta.address_as_bytes()
        )
        log(INFO, "Web3 model contract address: %s", factory_meta.address)

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

        logs = factory.events.ContractCreated().process_receipt(tx_receipt)
        assert len(logs) == 1, "no events discovered, factory might not be deployed"
        contract_created = logs[0]
        proxy_address = contract_created["args"]["proxyAddress"]

        return ModelMetaV1.from_address(proxy_address, w3, deployer)
