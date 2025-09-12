import os
from pathlib import Path
from typing import Unpack

from eth_account.signers.base import BaseAccount
from eth_typing import ChecksumAddress
from hexbytes import HexBytes
from pydantic import BaseModel, Field
from rizemind.contracts.abi import decode_events_from_tx, encode_with_selector
from rizemind.contracts.abi_helper import load_abi
from rizemind.contracts.base_contract import (
    BaseContract,
    FromAddressKwargs,
    contract_factory,
)
from rizemind.contracts.has_account import HasAccount
from web3 import Web3
from web3.contract import Contract


def get_id(version: str) -> HexBytes:
    """
    Matches Solidity: keccak256(abi.encodePacked(version))
    Returns 32-byte hash (bytes32) as HexBytes.
    """
    return Web3.keccak(text=version)


class AccessControlParams(BaseModel):
    id: bytes = Field(..., description="The access control id")
    init_data: bytes = Field(..., description="The access control init data")


class AccessControlConfig(BaseModel):
    name: str = Field(..., description="The access control name")
    version: str = Field(..., description="The access control version")

    def get_access_control_id(self) -> HexBytes:
        """
        Matches Solidity: keccak256(abi.encodePacked(version))
        Returns 32-byte hash (bytes32) as HexBytes.
        """
        return get_id(f"{self.name}-v{self.version}")

    def get_init_data(self) -> HexBytes:
        """
        Generate initialization data for BaseAccessControl.

        Args:
            aggregator: The aggregator address
            trainers: List of trainer addresses
            evaluators: List of evaluator addresses

        Returns:
            Encoded initialization data
        """
        return encode_with_selector(
            "initialize()",
            [],
            [],
        )

    def get_access_control_params(
        self,
    ) -> AccessControlParams:
        return AccessControlParams(
            id=self.get_access_control_id(),
            init_data=self.get_init_data(),
        )


abi = load_abi(Path(os.path.dirname(__file__)) / "./access_control_factory_abi.json")


class AccessControlFactoryContract(BaseContract, HasAccount):
    def __init__(self, contract: Contract, account: BaseAccount | None):
        BaseContract.__init__(self, contract=contract)
        HasAccount.__init__(self, account=account)

    def get_id(self, version: str) -> HexBytes:
        return self.contract.functions.getID(version).call()

    def create_access_control(
        self, access_control_id: HexBytes, salt: HexBytes, init_data: bytes
    ) -> HexBytes:
        """
        Create a new access control instance using UUPS proxy.

        Args:
            access_control_id: The identifier of the access control implementation to use
            salt: The salt for CREATE2 deployment
            init_data: The encoded initialization data for the access control instance

        Returns:
            The transaction hash of the creation transaction
        """
        account = self.get_account()

        return self.send(
            tx_fn=self.contract.functions.createAccessControl(
                access_control_id, salt, init_data
            ),
            from_account=account,
        )

    def get_deployed_access_control(
        self, access_control_deploy_tx: HexBytes
    ) -> ChecksumAddress:
        logs = decode_events_from_tx(
            deploy_tx=access_control_deploy_tx,
            event=self.contract.events.AccessControlInstanceCreated(),
            w3=self.w3,
        )
        assert len(logs) == 1, "no events discovered, factory might not be deployed"
        contract_created = logs[0]
        proxy_address = contract_created["args"]["instance"]
        return Web3.to_checksum_address(proxy_address)

    def is_access_control_registered(self, access_control_id: HexBytes) -> bool:
        """Check if an access control is registered."""
        return self.contract.functions.isAccessControlRegistered(
            access_control_id
        ).call()

    def is_access_control_version_registered(self, version: str) -> bool:
        """Check if an access control version is registered."""
        return self.contract.functions.isAccessControlVersionRegistered(version).call()

    def get_access_control_implementation(
        self, access_control_id: HexBytes
    ) -> ChecksumAddress:
        """Get the implementation address for an access control ID."""
        return self.contract.functions.getAccessControlImplementation(
            access_control_id
        ).call()

    def register_access_control_implementation(self, implementation: str) -> HexBytes:
        """
        Register a new access control implementation.

        Args:
            implementation: The address of the access control implementation contract

        Returns:
            The transaction hash of the registration transaction
        """
        account = self.get_account()

        return self.send(
            tx_fn=self.contract.functions.registerAccessControlImplementation(
                implementation
            ),
            from_account=account,
        )

    def remove_access_control_implementation(
        self, access_control_id: HexBytes
    ) -> HexBytes:
        """
        Remove an access control implementation.

        Args:
            access_control_id: The unique identifier for the access control implementation

        Returns:
            The transaction hash of the removal transaction
        """
        account = self.get_account()

        return self.send(
            tx_fn=self.contract.functions.removeAccessControlImplementation(
                access_control_id
            ),
            from_account=account,
        )

    @staticmethod
    def from_address(
        *, account: BaseAccount | None = None, **kwargs: Unpack[FromAddressKwargs]
    ) -> "AccessControlFactoryContract":
        return AccessControlFactoryContract(
            contract=contract_factory(**kwargs, abi=abi), account=account
        )
