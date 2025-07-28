import os
from pathlib import Path
from typing import Unpack

from eth_typing import ChecksumAddress
from rizemind.contracts.abi_helper import load_abi
from rizemind.contracts.base_contract import (
    BaseContract,
    FromAddressKwargs,
    contract_factory,
)
from rizemind.contracts.swarm.constants import CONTRIBUTION_DECIMALS
from web3 import Web3
from web3.contract import Contract

abi = load_abi(Path(os.path.dirname(__file__)) / "./trainer_contributed_event.abi.json")


class TrainersContributedEventHelper(BaseContract):
    def __init__(self, contract: Contract):
        super().__init__(contract=contract)

    @staticmethod
    def from_address(
        **kwargs: Unpack[FromAddressKwargs],
    ) -> "TrainersContributedEventHelper":
        return TrainersContributedEventHelper(contract_factory(**kwargs, abi=abi))

    def get_latest_contribution(
        self,
        trainer: ChecksumAddress,
        from_block: int | str = 0,
        to_block: int | str = "latest",
    ) -> float | None:
        latest = self.get_latest_contribution_log(trainer, from_block, to_block)

        if latest is None:
            return None

        contribution: int = latest["args"]["contribution"]

        return contribution / 10**CONTRIBUTION_DECIMALS

    def get_latest_contribution_log(
        self,
        trainer: ChecksumAddress,
        from_block: int | str = 0,
        to_block: int | str = "latest",
    ) -> dict | None:
        """
        Returns the latest contribution even if the round is not finished
        """
        trainer = Web3.to_checksum_address(trainer)

        event = self.contract.events.TrainerContributed()

        if type(from_block) is int:
            from_block = from_block - 1 if from_block > 0 else from_block
        logs: list[dict] = event.get_logs(
            from_block=from_block,
            to_block=to_block,
            argument_filters={"trainer": trainer},
        )
        if not logs:
            return None

        return max(
            logs,
            key=lambda log: (
                log["blockNumber"],
                log["transactionIndex"],
                log["logIndex"],
            ),
        )
