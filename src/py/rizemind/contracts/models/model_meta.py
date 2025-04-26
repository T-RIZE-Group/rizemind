from typing import List, Optional, Union
from eth_typing import HexAddress
from pydantic import BaseModel
from rizemind.contracts.models.constants import (
    CONTRIBUTION_DECIMALS,
    MODEL_SCORE_DECIMALS,
)
from rizemind.contracts.models.erc5267 import ERC5267
from web3.contract import Contract
from web3 import Web3


class RoundMetrics(BaseModel):
    n_trainers: int
    model_score: float
    total_contributions: float


class RoundSummary(BaseModel):
    round_id: int
    finished: bool
    metrics: Optional[RoundMetrics]


class ModelMeta(ERC5267):
    model: Contract

    def __init__(self, model: Contract):
        ERC5267.__init__(self, model)
        self.model = model

    def can_train(self, trainer: HexAddress, round_id: int) -> bool:
        return self.model.functions.canTrain(trainer, round_id).call()

    def current_round(self) -> int:
        return self.model.functions.currentRound().call()

    def get_latest_contribution(
        self,
        trainer: HexAddress,
        from_block: Union[int, str] = 0,
        to_block: Union[int, str] = "latest",
    ) -> Optional[float]:
        latest = self.get_latest_contribution_log(trainer, from_block, to_block)

        if latest is None:
            return None

        contribution: int = latest["args"]["contribution"]

        return contribution / 10**CONTRIBUTION_DECIMALS

    def get_latest_contribution_log(
        self,
        trainer: HexAddress,
        from_block: Union[int, str] = 0,
        to_block: Union[int, str] = "latest",
    ) -> dict | None:
        """
        Returns the latest contribution even if the round is not finished
        """
        trainer = Web3.to_checksum_address(trainer)

        event = self.model.events.TrainerContributed()

        logs: List[dict] = event.get_logs(
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

    def get_last_contributed_round_summary(
        self,
        trainer: HexAddress,
        from_block: Union[int, str] = 0,
        to_block: Union[int, str] = "latest",
    ) -> RoundSummary | None:
        """
        Returns the summary of the latest FINISHED round the trainer has contributed to
        """
        latest_contribution = self.get_latest_contribution_log(
            trainer, from_block, to_block
        )
        print(latest_contribution)
        if latest_contribution is None:
            return None
        return self.get_round_at(latest_contribution["blockNumber"])

    def get_round_at(self, block_height: int) -> RoundSummary:
        event = self.model.events.RoundFinished()
        logs: List[dict] = event.get_logs()
        for log in logs:
            if log["blockNumber"] >= block_height:
                metrics = RoundMetrics(
                    n_trainers=log["args"]["nTrainers"],
                    model_score=log["args"]["modelScore"] / 10**MODEL_SCORE_DECIMALS,
                    total_contributions=log["args"]["totalContribution"]
                    / 10**CONTRIBUTION_DECIMALS,
                )
                return RoundSummary(
                    round_id=log["args"]["roundId"], finished=True, metrics=metrics
                )

        round_id: int = 1 if len(logs) == 0 else logs[-1]["args"]["roundId"] + 1

        return RoundSummary(round_id=round_id, finished=False, metrics=None)
