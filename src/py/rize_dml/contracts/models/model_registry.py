from eth_typing import Address
from web3.contract import Contract

class ModelRegistry():

  model: Contract

  def __init__(self, model: Contract):
    self.model = model

  def can_train(self, trainer: Address, round_id: int) -> bool:
    return self.model.functions.canTrain(trainer, round_id).call()
  
  def current_round(self) -> int:
    return self.model.functions.currentRound().call()

