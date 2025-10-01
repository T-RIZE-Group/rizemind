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


class SwarmCore(BaseContract, HasAccount):
    """SwarmCore contract class for managing swarm configuration."""

    def __init__(self, contract: Contract, *, account: BaseAccount | None = None):
        BaseContract.__init__(self, contract=contract)
        HasAccount.__init__(self, account=account)

    @staticmethod
    def from_address(**kwargs: Unpack[FromAddressKwargs]) -> "SwarmCore":
        return SwarmCore(contract_factory(**kwargs, abi=abi))

    def initialize(
        self,
        initial_trainer_selector: ChecksumAddress,
        initial_evaluator_selector: ChecksumAddress,
        initial_contribution_calculator: ChecksumAddress,
        initial_access_control: ChecksumAddress,
        initial_compensation: ChecksumAddress,
    ) -> HexBytes:
        """Initialize the contract with initial component addresses."""
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.initialize(
                initial_trainer_selector,
                initial_evaluator_selector,
                initial_contribution_calculator,
                initial_access_control,
                initial_compensation,
            ),
            from_account=account,
        )

    def get_trainer_selector(self) -> ChecksumAddress:
        """Get the current trainer selector contract address."""
        return self.contract.functions.getTrainerSelector().call()

    def get_evaluator_selector(self) -> ChecksumAddress:
        """Get the current evaluator selector contract address."""
        return self.contract.functions.getEvaluatorSelector().call()

    def get_contribution_calculator(self) -> ChecksumAddress:
        """Get the current contribution calculator contract address."""
        return self.contract.functions.getContributionCalculator().call()

    def get_access_control(self) -> ChecksumAddress:
        """Get the current access control contract address."""
        return self.contract.functions.getAccessControl().call()

    def get_compensation(self) -> ChecksumAddress:
        """Get the current compensation contract address."""
        return self.contract.functions.getCompensation().call()
