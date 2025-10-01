from typing import Annotated, Any

from eth_account.signers.base import BaseAccount
from eth_typing import ChecksumAddress
from flwr.common.context import Context
from pydantic import Field, SkipValidation, model_validator
from rizemind.configuration import BaseConfig, unflatten
from rizemind.configuration.validators import EthereumAddressOrNone
from rizemind.contracts.swarm.swarm_v1.swarm_v1_factory import (
    SwarmV1Factory,
    SwarmV1FactoryConfig,
)
from rizemind.exception import RizemindException
from rizemind.swarm.swarm import Swarm
from web3 import Web3

SWARM_CONFIG_STATE_KEY = "rizemind.swarm"


class SwarmConfigException(RizemindException):
    def __init__(self, field: str):
        super().__init__(code="missing_value", message=f"missing field {field}")


class SwarmConfig(BaseConfig):
    address: EthereumAddressOrNone | None = Field(
        default=None, description="Ethereum address for the swarm contract"
    )
    factory_v1: Annotated[SwarmV1FactoryConfig | None, SkipValidation] = Field(
        default=None,
        description="ModelFactoryV1Config object to deploy on Aggregator side",
    )

    @model_validator(mode="after")
    def _one_of_address_or_factory(self) -> "SwarmConfig":
        # Only validate factory_v1 if address is absent
        if self.address is None:
            # Accept dict or instance; this enforces SwarmV1FactoryConfig now
            self.factory_v1 = SwarmV1FactoryConfig.model_validate(self.factory_v1)
        else:
            self.factory_v1 = None
        return self

    def get(self, *, account: BaseAccount | None = None, w3: Web3) -> Swarm:
        if self.address is None:
            raise SwarmConfigException("address")
        return Swarm(
            address=Web3.to_checksum_address(self.address), w3=w3, account=account
        )

    def deploy(self, *, deployer: BaseAccount, w3: Web3) -> Swarm:
        if self.factory_v1 is None:
            raise SwarmConfigException("factory_v1")
        factory = SwarmV1Factory(self.factory_v1)
        deployment = factory.deploy(deployer, w3)
        return Swarm(w3=w3, address=deployment.address, account=deployer)

    def get_or_deploy(
        self,
        *,
        deployer: BaseAccount,
        w3: Web3,
    ) -> Swarm:
        if self.factory_v1 is not None:
            return self.deploy(deployer=deployer, w3=w3)

        if self.address is not None:
            return Swarm(
                address=Web3.to_checksum_address(self.address), w3=w3, account=deployer
            )

        raise Exception("No address or factory settings found")

    def store_in_context(self, context: Context) -> None:
        self._store_in_context(context, SWARM_CONFIG_STATE_KEY)

    @staticmethod
    def from_context(
        context: Context, *, fallback_address: ChecksumAddress | None = None
    ) -> "SwarmConfig | None":
        if SWARM_CONFIG_STATE_KEY in context.state.config_records:
            records: Any = context.state.config_records[SWARM_CONFIG_STATE_KEY]
            return SwarmConfig(**unflatten(records))
        if fallback_address is not None:
            return SwarmConfig(address=fallback_address)
        return None
