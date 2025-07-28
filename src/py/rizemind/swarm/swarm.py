from eth_account.signers.base import BaseAccount
from eth_typing import ChecksumAddress
from rizemind.contracts.access_control.fl_access_control.FlAccessControl import (
    FlAccessControl,
)
from rizemind.contracts.swarm.contributions.trainers_contributed import (
    TrainersContributedEventHelper,
)
from rizemind.contracts.swarm.swarm_v1.swarm_v1 import SwarmV1
from rizemind.contracts.swarm.training.round_training import RoundTraining
from rizemind.swarm.specs.supports_can_train import SupportsCanTrain
from rizemind.swarm.specs.supports_distribute import SupportsDistribute
from rizemind.swarm.specs.supports_round import SupportsRound
from rizemind.swarm.specs.supports_trainer_contributed import SupportsTrainerContributed
from web3 import Web3


class Swarm(
    SupportsCanTrain, SupportsDistribute, SupportsRound, SupportsTrainerContributed
):
    access_control: FlAccessControl
    training: RoundTraining
    swarm: SwarmV1
    contribution: TrainersContributedEventHelper

    def __init__(
        self, *, w3: Web3, address: ChecksumAddress, account: BaseAccount
    ) -> None:
        self.access_control = FlAccessControl.from_address(w3=w3, address=address)
        self.training = RoundTraining.from_address(
            w3=w3, address=address, account=account
        )
        self.swarm = SwarmV1.from_address(w3=w3, address=address, account=account)
        self.contribution = TrainersContributedEventHelper.from_address(
            w3=w3, address=address
        )

    # Implements SupportsCanTrain
    def can_train(self, trainer: ChecksumAddress) -> bool:
        return self.access_control.is_trainer(trainer)

    # Implements SupportsDistribute
    def distribute(self, trainer_scores: list[tuple[ChecksumAddress, float]]) -> str:
        return self.swarm.distribute(trainer_scores).to_0x_hex()

    # Implements SupportsRound
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
