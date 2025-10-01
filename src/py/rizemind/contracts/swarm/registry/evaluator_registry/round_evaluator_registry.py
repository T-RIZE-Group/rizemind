import os
from pathlib import Path
from typing import Unpack

from eth_account.signers.base import BaseAccount
from eth_typing import ChecksumAddress
from hexbytes import HexBytes
from rizemind.contracts.abi_helper import load_abi
from rizemind.contracts.base_contract import (
    BaseContract,
    FromAddressKwargs,
    contract_factory,
)
from rizemind.contracts.has_account import HasAccount
from web3.contract import Contract

abi = load_abi(Path(os.path.dirname(__file__)) / "./abi.json")


class RoundEvaluatorRegistry(HasAccount, BaseContract):
    """RoundEvaluatorRegistry contract class for managing evaluators per round."""

    def __init__(self, contract: Contract, account: BaseAccount | None = None):
        HasAccount.__init__(self, account=account)
        BaseContract.__init__(self, contract=contract)

    @staticmethod
    def from_address(
        account: BaseAccount | None, **kwargs: Unpack[FromAddressKwargs]
    ) -> "RoundEvaluatorRegistry":
        return RoundEvaluatorRegistry(
            contract_factory(**kwargs, abi=abi), account=account
        )

    def initialize(self) -> HexBytes:
        """Initialize the contract. This function can only be called once during proxy deployment."""
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.initialize(),
            from_account=account,
        )

    def get_evaluator_id(self, round_id: int, evaluator: ChecksumAddress) -> int:
        """Get the ID of a specific evaluator for a round."""
        return self.contract.functions.getEvaluatorId(round_id, evaluator).call()

    def get_evaluator_id_or_throw(
        self, round_id: int, evaluator: ChecksumAddress
    ) -> int:
        """Get the ID of a specific evaluator for a round, throwing if not found."""
        return self.contract.functions.getEvaluatorIdOrThrow(round_id, evaluator).call()

    def get_evaluator_count(self, round_id: int) -> int:
        """Get the total number of evaluators for a round."""
        return self.contract.functions.getEvaluatorCount(round_id).call()

    def is_evaluator_registered(
        self, round_id: int, evaluator: ChecksumAddress
    ) -> bool:
        """Check if an evaluator is registered for a round."""
        return self.contract.functions.isEvaluatorRegistered(round_id, evaluator).call()
