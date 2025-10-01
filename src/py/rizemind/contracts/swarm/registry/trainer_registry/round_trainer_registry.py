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


class RoundTrainerRegistry(HasAccount, BaseContract):
    """RoundTrainerRegistry contract class for managing trainers per round."""

    def __init__(self, contract: Contract, account: BaseAccount | None = None):
        HasAccount.__init__(self, account=account)
        BaseContract.__init__(self, contract=contract)

    @staticmethod
    def from_address(
        account: BaseAccount | None, **kwargs: Unpack[FromAddressKwargs]
    ) -> "RoundTrainerRegistry":
        return RoundTrainerRegistry(
            contract_factory(**kwargs, abi=abi), account=account
        )

    def initialize(self) -> HexBytes:
        """Initialize the contract. This function can only be called once during proxy deployment."""
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.initialize(),
            from_account=account,
        )

    def get_trainer_id(self, round_id: int, trainer: ChecksumAddress) -> int:
        """Get the ID of a specific trainer for a round."""
        return self.contract.functions.getTrainerId(round_id, trainer).call()

    def get_trainer_id_or_throw(self, round_id: int, trainer: ChecksumAddress) -> int:
        """Get the ID of a specific trainer for a round, throwing if not found."""
        return self.contract.functions.getTrainerIdOrThrow(round_id, trainer).call()

    def get_model_hash(self, round_id: int, trainer: ChecksumAddress) -> HexBytes:
        """Get the model hash of a specific trainer for a round."""
        return self.contract.functions.getModelHash(round_id, trainer).call()

    def get_model_hash_or_throw(
        self, round_id: int, trainer: ChecksumAddress
    ) -> HexBytes:
        """Get the model hash of a specific trainer for a round, throwing if not found."""
        return self.contract.functions.getModelHashOrThrow(round_id, trainer).call()

    def get_trainer_info(
        self, round_id: int, trainer: ChecksumAddress
    ) -> tuple[int, HexBytes]:
        """Get both ID and model hash of a specific trainer for a round.

        Returns:
            A tuple of (trainer_id, model_hash)
        """
        result = self.contract.functions.getTrainerInfo(round_id, trainer).call()
        return result[0], result[1]

    def get_trainer_count(self, round_id: int) -> int:
        """Get the total number of trainers for a round."""
        return self.contract.functions.getTrainerCount(round_id).call()

    def is_trainer_registered(self, round_id: int, trainer: ChecksumAddress) -> bool:
        """Check if a trainer is registered for a round."""
        return self.contract.functions.isTrainerRegistered(round_id, trainer).call()

    def has_claimed_rewards(self, round_id: int, trainer: ChecksumAddress) -> bool:
        """Check if a trainer has claimed their rewards for a round."""
        return self.contract.functions.hasClaimedRewards(round_id, trainer).call()

    def get_trainer_address_by_id(
        self, round_id: int, trainer_id: int
    ) -> ChecksumAddress | None:
        """Get the trainer address by its ID for a specific round by querying TrainerRegistered events.

        Args:
            round_id: The round ID to search in
            trainer_id: The trainer ID to look for

        Returns:
            The trainer address if found, None otherwise
        """
        # Create filter for TrainerRegistered events
        event_filter = self.contract.events.TrainerRegistered.create_filter(
            from_block=0,
            to_block="latest",
            argument_filters={"roundId": round_id, "trainerId": trainer_id},
        )

        # Get all matching events
        events = event_filter.get_all_entries()

        if events:
            # Return the first matching trainer address
            return events[0]["args"]["trainer"]

        return None
