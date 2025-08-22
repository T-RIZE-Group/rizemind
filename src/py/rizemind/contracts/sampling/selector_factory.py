import os
from pathlib import Path
from typing import Unpack, cast

from eth_account.signers.base import BaseAccount
from eth_account.types import TransactionDictType
from hexbytes import HexBytes
from pydantic import BaseModel, Field
from rizemind.contracts.abi import encode_with_selector
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


class SelectorParams(BaseModel):
    id: bytes = Field(..., description="The selector id")
    init_data: bytes = Field(..., description="The selector init data")


class SelectorConfig(BaseModel):
    name: str = Field(..., description="The selector name")
    version: str = Field(..., description="The selector version")

    def get_selector_id(self) -> HexBytes:
        """
        Matches Solidity: keccak256(abi.encodePacked(version))
        Returns 32-byte hash (bytes32) as HexBytes.
        """
        return get_id(f"{self.name}-v{self.version}")

    def get_init_data(self) -> HexBytes:
        return encode_with_selector("initialize()", [], [])

    def get_selector_params(self) -> SelectorParams:
        return SelectorParams(
            id=self.get_selector_id(),
            init_data=self.get_init_data(),
        )


abi = load_abi(Path(os.path.dirname(__file__)) / "./selector_factory_abi.json")


class SelectorFactoryContract(BaseContract, HasAccount):
    def __init__(self, contract: Contract, account: BaseAccount | None):
        BaseContract.__init__(self, contract=contract)
        HasAccount.__init__(self, account=account)

    def get_id(self, version: str) -> HexBytes:
        return self.contract.functions.getID(version).call()

    def create_selector(
        self, selector_id: HexBytes, salt: HexBytes, init_data: bytes
    ) -> str:
        """
        Create a new selector instance using UUPS proxy.

        Args:
            selector_id: The identifier of the selector implementation to use
            salt: The salt for CREATE2 deployment
            init_data: The encoded initialization data for the selector instance

        Returns:
            The transaction hash of the creation transaction
        """
        account = self.get_account()
        if account is None:
            raise ValueError("Account is required to create selector")

        tx = self.contract.functions.createSelector(
            selector_id, salt, init_data
        ).build_transaction(
            {
                "from": account.address,
                "nonce": self.w3.eth.get_transaction_count(account.address),
            }
        )

        signed = account.sign_transaction(cast(TransactionDictType, tx))
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        return tx_hash.hex()

    def is_selector_registered(self, selector_id: HexBytes) -> bool:
        """Check if a selector is registered."""
        return self.contract.functions.isSelectorRegistered(selector_id).call()

    def is_selector_version_registered(self, version: str) -> bool:
        """Check if a selector version is registered."""
        return self.contract.functions.isSelectorVersionRegistered(version).call()

    def get_selector_implementation(self, selector_id: HexBytes) -> str:
        """Get the implementation address for a selector ID."""
        return self.contract.functions.getSelectorImplementation(selector_id).call()

    @staticmethod
    def from_address(
        *, account: BaseAccount | None = None, **kwargs: Unpack[FromAddressKwargs]
    ) -> "SelectorFactoryContract":
        return SelectorFactoryContract(
            contract=contract_factory(**kwargs, abi=abi), account=account
        )
