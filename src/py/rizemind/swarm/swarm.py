from typing import Union
from eth_account.signers.base import BaseAccount
from eth_typing import ChecksumAddress
from rizemind.contracts.access_control.fl_access_control.FlAccessControl import (
    FlAccessControl,
)
from rizemind.contracts.erc.erc5267.erc5267 import (
    ERC5267,
    EIP712Domain,
)
from rizemind.contracts.swarm.contributions.trainers_contributed import (
    TrainersContributedEventHelper,
)
from rizemind.contracts.swarm.swarm_v1.swarm_v1 import SwarmV1
from rizemind.contracts.swarm.training.round_training import RoundSummary, RoundTraining
from web3 import Web3


class Swarm:
    access_control: FlAccessControl
    training: RoundTraining
    swarm: SwarmV1
    contribution: TrainersContributedEventHelper
    erc5267: ERC5267

    def __init__(
        self, *, w3: Web3, address: ChecksumAddress, account: BaseAccount | None
    ) -> None:
        self.access_control = FlAccessControl.from_address(w3=w3, address=address)
        self.training = RoundTraining.from_address(
            w3=w3, address=address, account=account
        )
        self.swarm = SwarmV1.from_address(w3=w3, address=address, account=account)
        self.contribution = TrainersContributedEventHelper.from_address(
            w3=w3, address=address
        )
        self.erc5267 = ERC5267.from_address(w3=w3, address=address)

    def can_train(self, trainer: ChecksumAddress, round_id: int) -> bool:
        return self.access_control.is_trainer(trainer)

    def distribute(self, trainer_scores: list[tuple[ChecksumAddress, float]]) -> str:
        return self.swarm.distribute(trainer_scores).to_0x_hex()

    def get_current_round(self) -> int:
        return self.training.current_round()

    def next_round(
        self,
        round_id: int,
        n_trainers: int,
        model_score: float,
        total_contributions: float,
    ) -> str:
        return self.training.next_round(
            round_id=round_id,
            n_trainers=n_trainers,
            model_score=model_score,
            total_contributions=total_contributions,
        ).to_0x_hex()

    def current_round(self) -> int:
        return self.training.current_round()

    def get_latest_contribution(
        self,
        trainer: ChecksumAddress,
        from_block: int | str = 0,
        to_block: int | str = "latest",
    ) -> float | None:
        return self.contribution.get_latest_contribution(
            trainer, from_block=from_block, to_block=to_block
        )

    def get_latest_contribution_log(
        self,
        trainer: ChecksumAddress,
        from_block: int | str = 0,
        to_block: int | str = "latest",
    ) -> dict | None:
        return self.contribution.get_latest_contribution_log(
            trainer, from_block=from_block, to_block=to_block
        )

    def get_eip712_domain(self) -> EIP712Domain:
        return self.erc5267.get_eip712_domain()

    def get_last_contributed_round_summary(
        self,
        trainer: ChecksumAddress,
        from_block: int | str = 0,
        to_block: int | str = "latest",
    ) -> RoundSummary | None:
        """
        Returns the summary of the latest FINISHED round the trainer has contributed to
        """
        latest_contribution = self.get_latest_contribution_log(
            trainer, from_block, to_block
        )

        if latest_contribution is None:
            return None
        return self.training.get_round_at(latest_contribution["blockNumber"])
