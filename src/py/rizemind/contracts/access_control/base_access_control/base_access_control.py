import os
from pathlib import Path
from typing import Unpack

from eth_account.signers.base import BaseAccount
from eth_typing import ChecksumAddress
from hexbytes import HexBytes
from rizemind.contracts.abi import encode_with_selector
from rizemind.contracts.abi_helper import load_abi
from rizemind.contracts.access_control.access_control_factory import AccessControlConfig
from rizemind.contracts.base_contract import (
    FromAddressKwargs,
    contract_factory,
)
from rizemind.contracts.erc.erc5267.erc5267 import ERC5267
from rizemind.contracts.has_account import HasAccount
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
    ) -> HexBytes:
        """Initialize the BaseAccessControl contract with aggregator, trainers, and evaluators."""
        account = self.get_account()

        return self.send(
            tx_fn=self.contract.functions.initialize(aggregator, trainers, evaluators),
            from_account=account,
        )

    def is_evaluator(self, address: ChecksumAddress) -> bool:
        """Check if an address is an evaluator."""
        return self.contract.functions.isEvaluator(address).call()

    def add_trainer(self, trainer: ChecksumAddress) -> HexBytes:
        """Add a new trainer (only callable by aggregator)."""
        account = self.get_account()

        return self.send(
            tx_fn=self.contract.functions.addTrainer(trainer),
            from_account=account,
        )

    def add_aggregator(self, aggregator: ChecksumAddress) -> HexBytes:
        """Add a new aggregator (only callable by existing aggregator)."""
        account = self.get_account()

        return self.send(
            tx_fn=self.contract.functions.addAggregator(aggregator),
            from_account=account,
        )

    def add_evaluator(self, evaluator: ChecksumAddress) -> HexBytes:
        """Add a new evaluator (only callable by aggregator)."""
        account = self.get_account()

        return self.send(
            tx_fn=self.contract.functions.addEvaluator(evaluator),
            from_account=account,
        )

    def supports_interface(self, interface_id: str) -> bool:
        """Check if the contract supports a specific interface."""
        return self.contract.functions.supportsInterface(interface_id).call()

    def eip712_domain(self) -> dict:
        """Get the EIP712 domain information."""
        return self.contract.functions.eip712Domain().call()
