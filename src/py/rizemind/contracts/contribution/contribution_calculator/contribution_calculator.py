import os
from pathlib import Path
from typing import Unpack

from eth_account.signers.base import BaseAccount
from eth_typing import ChecksumAddress
from hexbytes import HexBytes
from web3.contract import Contract

from rizemind.configuration.validators.eth_address import EthereumAddress
from rizemind.contracts.abi import encode_with_selector
from rizemind.contracts.abi_helper import load_abi
from rizemind.contracts.base_contract import (
    FromAddressKwargs,
    contract_factory,
)
from rizemind.contracts.contribution.calculator_factory import CalculatorConfig
from rizemind.contracts.erc.erc5267.erc5267 import ERC5267
from rizemind.contracts.has_account import HasAccount


class ContributionCalculatorConfig(CalculatorConfig):
    name: str = "contribution-calculator"
    version: str = "1.0.0"

    initial_admin: EthereumAddress | None = None
    initial_num_samples: int

    def get_init_data(self, *, swarm_address: ChecksumAddress) -> HexBytes:
        """
        Generate initialization data for ContributionCalculator.

        Args:
            initial_admin: The initial admin address. Defaults to swarm address
            initial_num_samples: The initial number of samples

        Returns:
            Encoded initialization data
        """
        return encode_with_selector(
            "initialize(address,uint256)",
            ["address", "uint256"],
            [self.initial_admin or swarm_address, self.initial_num_samples],
        )


abi = load_abi(Path(os.path.dirname(__file__)) / "./abi.json")


class ContributionCalculator(ERC5267, HasAccount):
    def __init__(self, contract: Contract, *, account: BaseAccount | None = None):
        ERC5267.__init__(self, contract=contract)
        HasAccount.__init__(self, account=account)

    @staticmethod
    def from_address(**kwargs: Unpack[FromAddressKwargs]) -> "ContributionCalculator":
        return ContributionCalculator(contract_factory(**kwargs, abi=abi))

    def initialize(
        self,
        initial_admin: ChecksumAddress,
        initial_num_samples: int,
    ) -> HexBytes:
        """Initialize the ContributionCalculator contract with initial admin and number of samples."""
        account = self.get_account()

        return self.send(
            tx_fn=self.contract.functions.initialize(
                initial_admin, initial_num_samples
            ),
            from_account=account,
        )

    def register_result(
        self,
        round_id: int,
        sample_id: int,
        set_id: int,
        model_hash: HexBytes,
        result: int,
        number_of_players: int,
    ) -> HexBytes:
        """Register an evaluation result."""
        account = self.get_account()

        return self.send(
            tx_fn=self.contract.functions.registerResult(
                round_id, sample_id, set_id, model_hash, result, number_of_players
            ),
            from_account=account,
        )

    def set_evaluations_required(
        self,
        round_id: int,
        evaluations_required: int,
    ) -> HexBytes:
        """Set the number of evaluations required for a round."""
        account = self.get_account()

        return self.send(
            tx_fn=self.contract.functions.setEvaluationsRequired(
                round_id, evaluations_required
            ),
            from_account=account,
        )

    def grant_admin_role(self, account_address: ChecksumAddress) -> HexBytes:
        """Grant admin role to an account."""
        account = self.get_account()

        return self.send(
            tx_fn=self.contract.functions.grantAdminRole(account_address),
            from_account=account,
        )

    def revoke_admin_role(self, account_address: ChecksumAddress) -> HexBytes:
        """Revoke admin role from an account."""
        account = self.get_account()

        return self.send(
            tx_fn=self.contract.functions.revokeAdminRole(account_address),
            from_account=account,
        )

    def calculate_contribution(
        self, round_id: int, trainer_index: int, number_of_trainers: int
    ) -> int:
        """Calculate Shapley value for a specific trainer in a round."""
        return self.contract.functions.calculateContribution(
            round_id, trainer_index, number_of_trainers
        ).call()

    def get_evaluations_required(self, round_id: int, number_of_players: int) -> int:
        """Get the number of evaluations required for a round."""
        return self.contract.functions.getEvaluationsRequired(
            round_id, number_of_players
        ).call()

    def get_total_evaluations(self, round_id: int, number_of_players: int) -> int:
        """Get the total number of evaluations for a round."""
        return self.contract.functions.getTotalEvaluations(
            round_id, number_of_players
        ).call()

    def get_result(self, round_id: int, set_id: int) -> int:
        """Get evaluation result by round ID and set ID."""
        return self.contract.functions.getResult(round_id, set_id).call()

    def get_result_or_throw(self, round_id: int, set_id: int) -> int:
        """Get evaluation result by round ID and set ID, throws if not found."""
        return self.contract.functions.getResultOrThrow(round_id, set_id).call()

    def get_mask(self, round_id: int, i: int, number_of_players: int) -> int:
        """Get mask for a specific round, sample index, and number of players."""
        return self.contract.functions.getMask(round_id, i, number_of_players).call()

    def supports_interface(self, interface_id: str) -> bool:
        """Check if the contract supports a specific interface."""
        return self.contract.functions.supportsInterface(interface_id).call()
