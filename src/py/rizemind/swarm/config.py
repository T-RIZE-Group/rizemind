from pydantic import BaseModel, Field, model_validator
from rizemind.configuration.validators.eth_address import EthereumAddress
from rizemind.contracts.models.model_factory_v1 import (
    ModelFactoryV1,
    ModelFactoryV1Config,
)
from rizemind.contracts.models.model_meta_v1 import ModelMetaV1

from eth_account.signers.base import BaseAccount
from web3 import Web3


class SwarmConfig(BaseModel):
    address: EthereumAddress | None = Field(
        None, description="Ethereum address for the swarm contract"
    )
    factory_v1: ModelFactoryV1Config | None = Field(
        None, description="ModelFactoryV1Config object to deploy on Aggregator side"
    )

    @model_validator(mode="after")
    def _one_of_address_or_factory(self) -> "SwarmConfig":
        if (self.address is None) == (self.factory_v1 is None):  # XOR
            raise ValueError("One of `address` or `factory_v1` must be provided.")
        return self

    def get_or_deploy(self, *, deployer: BaseAccount, w3: Web3) -> ModelMetaV1:
        if self.factory_v1 is not None:
            factory = ModelFactoryV1(self.factory_v1)
            return factory.deploy(deployer, [], w3)

        if self.address is not None:
            return ModelMetaV1.from_address(self.address, w3, deployer)

        raise Exception("No address or factory settings found")
