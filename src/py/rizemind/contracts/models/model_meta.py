from typing import Optional, Union
from eth_typing import HexAddress
from rizemind.contracts.models.constants import CONTRIBUTION_DECIMALS
from rizemind.contracts.models.erc5267 import ERC5267
from web3.contract import Contract
from web3 import Web3


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
        trainer = Web3.to_checksum_address(trainer)

        event = self.model.events.TrainerContributed()

        logs = event.get_logs(
            from_block=from_block,
            to_block=to_block,
            argument_filters={"trainer": trainer},
        )
        if not logs:
            return None

        latest = max(
            logs,
            key=lambda log: (
                log["blockNumber"],
                log["transactionIndex"],
                log["logIndex"],
            ),
        )

        contribution: int = latest["args"]["contribution"]

        return contribution / 10**CONTRIBUTION_DECIMALS
