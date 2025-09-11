import os
from pathlib import Path
from typing import Unpack, cast

from eth_account.signers.base import BaseAccount
from eth_account.types import TransactionDictType
from eth_typing import ChecksumAddress
from rizemind.contracts.abi import encode_with_selector
from rizemind.contracts.abi_helper import load_abi
from rizemind.contracts.access_control.access_control_factory import AccessControlConfig
from rizemind.contracts.base_contract import (
    FromAddressKwargs,
    contract_factory,
)
from rizemind.contracts.erc.erc5267.erc5267 import ERC5267
from rizemind.contracts.has_account import HasAccount
from hexbytes import HexBytes
from web3.contract import Contract


class BaseAccessControlConfig(AccessControlConfig):
    name: str = "base-access-control"
    version: str = "1.0.0"

    aggregator: ChecksumAddress
    trainers: list[ChecksumAddress]
    evaluators: list[ChecksumAddress]

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
            "initialize(address,address[],address[])",
            ["address", "address[]", "address[]"],
            [self.aggregator, self.trainers, self.evaluators],
        )


abi = load_abi(Path(os.path.dirname(__file__)) / "./abi.json")


class BaseAccessControl(ERC5267, HasAccount):
    def __init__(self, contract: Contract, *, account: BaseAccount | None = None):
        ERC5267.__init__(self, contract=contract)
        HasAccount.__init__(self, account=account)

    @staticmethod
    def from_address(**kwargs: Unpack[FromAddressKwargs]) -> "BaseAccessControl":
        return BaseAccessControl(contract_factory(**kwargs, abi=abi))

    def is_trainer(self, address: str) -> bool:
        return self.contract.functions.isTrainer(address).call()

    def is_aggregator(self, address: str) -> bool:
        return self.contract.functions.isAggregator(address).call()

    def initialize(
        self,
        aggregator: ChecksumAddress,
        trainers: list[ChecksumAddress],
        evaluators: list[ChecksumAddress],
    ) -> str:
        """Initialize the BaseAccessControl contract with aggregator, trainers, and evaluators."""
        account = self.get_account()

        tx = self.contract.functions.initialize(
            aggregator, trainers, evaluators
        ).build_transaction(
            {
                "from": account.address,
                "nonce": self.w3.eth.get_transaction_count(account.address),
            }
        )
        signed = account.sign_transaction(cast(TransactionDictType, tx))
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        return tx_hash.hex()

    def is_evaluator(self, address: ChecksumAddress) -> bool:
        """Check if an address is an evaluator."""
        return self.contract.functions.isEvaluator(address).call()

    def add_trainer(self, trainer: ChecksumAddress) -> str:
        """Add a new trainer (only callable by aggregator)."""
        account = self.get_account()

        tx = self.contract.functions.addTrainer(trainer).build_transaction(
            {
                "from": account.address,
                "nonce": self.w3.eth.get_transaction_count(account.address),
            }
        )
        signed = account.sign_transaction(cast(TransactionDictType, tx))
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        return tx_hash.hex()

    def add_aggregator(self, aggregator: ChecksumAddress) -> str:
        """Add a new aggregator (only callable by existing aggregator)."""
        account = self.get_account()

        tx = self.contract.functions.addAggregator(aggregator).build_transaction(
            {
                "from": account.address,
                "nonce": self.w3.eth.get_transaction_count(account.address),
            }
        )
        signed = account.sign_transaction(cast(TransactionDictType, tx))
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        return tx_hash.hex()

    def add_evaluator(self, evaluator: ChecksumAddress) -> str:
        """Add a new evaluator (only callable by aggregator)."""
        account = self.get_account()

        tx = self.contract.functions.addEvaluator(evaluator).build_transaction(
            {
                "from": account.address,
                "nonce": self.w3.eth.get_transaction_count(account.address),
            }
        )
        signed = account.sign_transaction(cast(TransactionDictType, tx))
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        return tx_hash.hex()

    def supports_interface(self, interface_id: str) -> bool:
        """Check if the contract supports a specific interface."""
        return self.contract.functions.supportsInterface(interface_id).call()

    def eip712_domain(self) -> dict:
        """Get the EIP712 domain information."""
        return self.contract.functions.eip712Domain().call()
