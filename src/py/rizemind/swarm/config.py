from eth_account.signers.base import BaseAccount
from eth_typing import ChecksumAddress
from pydantic import BaseModel, Field, model_validator
from rizemind.configuration.validators.eth_address import EthereumAddress
from rizemind.contracts.swarm.swarm_v1.swarm_v1_factory import (
    SwarmV1Factory,
    SwarmV1FactoryConfig,
)
from rizemind.swarm.swarm import Swarm
from web3 import Web3


class SwarmConfig(BaseModel):
    address: EthereumAddress | None = Field(
        None, description="Ethereum address for the swarm contract"
    )
    factory_v1: SwarmV1FactoryConfig | None = Field(
        None, description="ModelFactoryV1Config object to deploy on Aggregator side"
    )

    @model_validator(mode="after")
    def _one_of_address_or_factory(self) -> "SwarmConfig":
        if (self.address is None) == (self.factory_v1 is None):  # XOR
            raise ValueError("One of `address` or `factory_v1` must be provided.")
        return self

    def get_or_deploy(
        self, *, deployer: BaseAccount, w3: Web3, trainers: list[ChecksumAddress] = []
    ) -> Swarm:
        if self.factory_v1 is not None:
            factory = SwarmV1Factory(self.factory_v1)
            deployment = factory.deploy(deployer, trainers, w3)
            return Swarm(w3=w3, address=deployment.address, account=deployer)

        if self.address is not None:
            return Swarm(
                address=Web3.to_checksum_address(self.address), w3=w3, account=deployer
            )

        raise Exception("No address or factory settings found")
