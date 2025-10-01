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


class CompensationParams(BaseModel):
    id: bytes = Field(..., description="The compensation id")
    init_data: bytes = Field(..., description="The compensation init data")


class CompensationConfig(BaseModel):
    name: str = Field(..., description="The compensation name")
    version: str = Field(..., description="The compensation version")

    def get_compensation_id(self) -> HexBytes:
        """
        Matches Solidity: keccak256(abi.encodePacked(version))
        Returns 32-byte hash (bytes32) as HexBytes.
        """
        return get_id(f"{self.name}-v{self.version}")

    def get_init_data(self, *, swarm_address: ChecksumAddress) -> HexBytes:
        """
        Generate initialization data for compensation contracts.

        Returns:
            Encoded initialization data
        """
        return encode_with_selector(
            "initialize()",
            [],
            [],
        )

    def get_compensation_params(
        self, *, swarm_address: ChecksumAddress
    ) -> CompensationParams:
        return CompensationParams(
            id=self.get_compensation_id(),
            init_data=self.get_init_data(swarm_address=swarm_address),
        )


abi = load_abi(Path(os.path.dirname(__file__)) / "./compensation_factory_abi.json")


class CompensationFactoryContract(BaseContract, HasAccount):
    def __init__(self, contract: Contract, account: BaseAccount | None):
        BaseContract.__init__(self, contract=contract)
        HasAccount.__init__(self, account=account)

    def get_id(self, version: str) -> HexBytes:
        return self.contract.functions.getID(version).call()

    def create_compensation(
        self, compensation_id: HexBytes, salt: HexBytes, init_data: bytes
    ) -> HexBytes:
        """
        Create a new compensation instance using UUPS proxy.

        Args:
            compensation_id: The identifier of the compensation implementation to use
            salt: The salt for CREATE2 deployment
            init_data: The encoded initialization data for the compensation instance

        Returns:
            The transaction hash of the creation transaction
        """
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.createCompensation(
                compensation_id, salt, init_data
            ),
            from_account=account,
        )

    def get_deployed_compensation(
        self, compensation_deploy_tx: HexBytes
    ) -> ChecksumAddress:
        logs = decode_events_from_tx(
            tx_hash=compensation_deploy_tx,
            event=self.contract.events.CompensationInstanceCreated(),
            w3=self.w3,
        )
        assert len(logs) == 1, "no events discovered, factory might not be deployed"
        contract_created = logs[0]
        proxy_address = contract_created["args"]["instance"]
        return Web3.to_checksum_address(proxy_address)

    def is_compensation_registered(self, compensation_id: HexBytes) -> bool:
        """Check if a compensation is registered."""
        return self.contract.functions.isCompensationRegistered(compensation_id).call()

    def is_compensation_version_registered(self, version: str) -> bool:
        """Check if a compensation version is registered."""
        return self.contract.functions.isCompensationVersionRegistered(version).call()

    def get_compensation_implementation(self, compensation_id: HexBytes) -> str:
        """Get the implementation address for a compensation ID."""
        return self.contract.functions.getCompensationImplementation(
            compensation_id
        ).call()

    def register_compensation_implementation(
        self, implementation: ChecksumAddress
    ) -> HexBytes:
        """
        Register a new compensation implementation.

        Args:
            implementation: The address of the compensation implementation contract

        Returns:
            The transaction hash of the registration transaction
        """
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.registerCompensationImplementation(
                implementation
            ),
            from_account=account,
        )

    def remove_compensation_implementation(self, compensation_id: HexBytes) -> HexBytes:
        """
        Remove a compensation implementation.

        Args:
            compensation_id: The unique identifier for the compensation implementation

        Returns:
            The transaction hash of the removal transaction
        """
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.removeCompensationImplementation(
                compensation_id
            ),
            from_account=account,
        )

    @staticmethod
    def from_address(
        *, account: BaseAccount | None = None, **kwargs: Unpack[FromAddressKwargs]
    ) -> "CompensationFactoryContract":
        return CompensationFactoryContract(
            contract=contract_factory(**kwargs, abi=abi), account=account
        )
