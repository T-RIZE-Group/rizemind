import math
import os
from pathlib import Path
from typing import Unpack, cast

from eth_account.signers.base import BaseAccount
from eth_account.types import TransactionDictType
from hexbytes import HexBytes
from pydantic import BaseModel
from rizemind.contracts.abi_helper import load_abi
from rizemind.contracts.base_contract import (
    BaseContract,
    FromAddressKwargs,
    contract_factory,
)
from rizemind.contracts.has_account import HasAccount
from rizemind.contracts.swarm.constants import (
    CONTRIBUTION_DECIMALS,
    MODEL_SCORE_DECIMALS,
)
from web3.contract import Contract

abi = load_abi(Path(os.path.dirname(__file__)) / "./abi.json")


class RoundMetrics(BaseModel):
    n_trainers: int
    model_score: float
    total_contributions: float


class RoundSummary(BaseModel):
    round_id: int
    finished: bool
    metrics: RoundMetrics | None


class RoundTraining(HasAccount, BaseContract):
    def __init__(self, contract: Contract, account: BaseAccount | None = None):
        HasAccount.__init__(self, account=account)
        BaseContract.__init__(self, contract=contract)

    @staticmethod
    def from_address(
        account: BaseAccount | None, **kwargs: Unpack[FromAddressKwargs]
    ) -> "RoundTraining":
        return RoundTraining(contract_factory(**kwargs, abi=abi), account=account)

    def current_round(self) -> int:
        return self.contract.functions.currentRound().call()

    def next_round(
        self,
        round_id: int,
        n_trainers: int,
        model_score: float,
        total_contributions: float,
    ) -> HexBytes:
        account = self.get_account()

        address = account.address
        round_summary_data: tuple[int, int, int, int] = (
            round_id,
            n_trainers,
            math.floor(model_score * 10**MODEL_SCORE_DECIMALS),
            math.floor(total_contributions * 10**CONTRIBUTION_DECIMALS),
        )
        tx = self.contract.functions.nextRound(round_summary_data).build_transaction(
            {"from": address, "nonce": self.w3.eth.get_transaction_count(address)}
        )
        signed_tx = account.sign_transaction(cast(TransactionDictType, tx))
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

        assert tx_receipt["status"] != 0, "nextRound returned an error"
        return tx_hash

    def get_round_at(self, block_height: int) -> RoundSummary:
        event = self.contract.events.RoundFinished()

        # Since event.get_logs returns logs as below:
        # range: (from_block, latest]
        # then we need to subtract '1' from from_block
        from_block = block_height - 1 if block_height > 0 else block_height
        logs: list[dict] = event.get_logs(
            from_block=from_block,
            to_block="latest",
        )
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
