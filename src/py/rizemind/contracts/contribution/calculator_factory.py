import os
from pathlib import Path
from typing import Unpack

from eth_account.signers.base import BaseAccount
from eth_typing import ChecksumAddress
from hexbytes import HexBytes
from pydantic import BaseModel, Field
from web3 import Web3
from web3.contract import Contract

from rizemind.contracts.abi import decode_events_from_tx, encode_with_selector
from rizemind.contracts.abi_helper import load_abi
from rizemind.contracts.base_contract import (
    BaseContract,
    FromAddressKwargs,
    contract_factory,
)
from rizemind.contracts.has_account import HasAccount


def get_id(version: str) -> HexBytes:
    """
    Matches Solidity: keccak256(abi.encodePacked(version))
    Returns 32-byte hash (bytes32) as HexBytes.
    """
    return Web3.keccak(text=version)


class CalculatorParams(BaseModel):
    id: bytes = Field(..., description="The calculator id")
    init_data: bytes = Field(..., description="The calculator init data")


class CalculatorConfig(BaseModel):
    name: str = Field(..., description="The calculator name")
    version: str = Field(..., description="The calculator version")

    def get_calculator_id(self) -> HexBytes:
        """
        Matches Solidity: keccak256(abi.encodePacked(version))
        Returns 32-byte hash (bytes32) as HexBytes.
        """
        return get_id(f"{self.name}-v{self.version}")

    def get_init_data(self, *, swarm_address: ChecksumAddress) -> HexBytes:
        """
        Generate initialization data for ContributionCalculator.

        Returns:
            Encoded initialization data
        """
        return encode_with_selector(
            "initialize()",
            [],
            [],
        )

    def get_calculator_params(
        self,
        *,
        swarm_address: ChecksumAddress,
    ) -> CalculatorParams:
        return CalculatorParams(
            id=self.get_calculator_id(),
            init_data=self.get_init_data(swarm_address=swarm_address),
        )


abi = load_abi(Path(os.path.dirname(__file__)) / "./calculator_factory_abi.json")


class CalculatorFactoryContract(BaseContract, HasAccount):
    def __init__(self, contract: Contract, account: BaseAccount | None):
        BaseContract.__init__(self, contract=contract)
        HasAccount.__init__(self, account=account)

    def get_id(self, version: str) -> HexBytes:
        return self.contract.functions.getID(version).call()

    def create_calculator(
        self, calculator_id: HexBytes, salt: HexBytes, init_data: bytes
    ) -> HexBytes:
        """
        Create a new calculator instance using UUPS proxy.

        Args:
            calculator_id: The identifier of the calculator implementation to use
            salt: The salt for CREATE2 deployment
            init_data: The encoded initialization data for the calculator instance

        Returns:
            The transaction hash of the creation transaction
        """
        account = self.get_account()

        return self.send(
            tx_fn=self.contract.functions.createCalculator(
                calculator_id, salt, init_data
            ),
            from_account=account,
        )

    def get_deployed_calculator(
        self, calculator_deploy_tx: HexBytes
    ) -> ChecksumAddress:
        logs = decode_events_from_tx(
            deploy_tx=calculator_deploy_tx,
            event=self.contract.events.CalculatorInstanceCreated(),
            w3=self.w3,
        )
        assert len(logs) == 1, "no events discovered, factory might not be deployed"
        contract_created = logs[0]
        proxy_address = contract_created["args"]["instance"]
        return Web3.to_checksum_address(proxy_address)

    def is_calculator_registered(self, calculator_id: HexBytes) -> bool:
        """Check if a calculator is registered."""
        return self.contract.functions.isCalculatorRegistered(calculator_id).call()

    def is_calculator_version_registered(self, version: str) -> bool:
        """Check if a calculator version is registered."""
        return self.contract.functions.isCalculatorVersionRegistered(version).call()

    def get_calculator_implementation(self, calculator_id: HexBytes) -> str:
        """Get the implementation address for a calculator ID."""
        return self.contract.functions.getCalculatorImplementation(calculator_id).call()

    def register_calculator_implementation(self, implementation: str) -> HexBytes:
        """
        Register a new calculator implementation.

        Args:
            implementation: The address of the calculator implementation contract

        Returns:
            The transaction hash of the registration transaction
        """
        account = self.get_account()

        return self.send(
            tx_fn=self.contract.functions.registerCalculatorImplementation(
                implementation
            ),
            from_account=account,
        )

    def remove_calculator_implementation(self, calculator_id: HexBytes) -> HexBytes:
        """
        Remove a calculator implementation.

        Args:
            calculator_id: The unique identifier for the calculator implementation

        Returns:
            The transaction hash of the removal transaction
        """
        account = self.get_account()

        return self.send(
            tx_fn=self.contract.functions.removeCalculatorImplementation(calculator_id),
            from_account=account,
        )

    @staticmethod
    def from_address(
        *, account: BaseAccount | None = None, **kwargs: Unpack[FromAddressKwargs]
    ) -> "CalculatorFactoryContract":
        return CalculatorFactoryContract(
            contract=contract_factory(**kwargs, abi=abi), account=account
        )
