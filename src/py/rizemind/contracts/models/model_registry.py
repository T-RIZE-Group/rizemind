from eth_typing import HexAddress
from rizemind.contracts.models.erc5267 import ERC5267
from web3.contract import Contract


class ModelRegistry(ERC5267):
    model: Contract

    def __init__(self, model: Contract):
        ERC5267.__init__(self, model)
        self.model = model

    def can_train(self, trainer: HexAddress, round_id: int) -> bool:
        return self.model.functions.canTrain(trainer, round_id).call()

    def current_round(self) -> int:
        return self.model.functions.currentRound().call()
