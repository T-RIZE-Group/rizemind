from typing import NamedTuple
from eth_typing import HexAddress
from web3.contract import Contract


class EIP712Domain(NamedTuple):
    fields: bytes
    name: str
    version: str
    chainId: int
    verifyingContract: str
    salt: bytes
    extensions: list[int]


class ModelRegistry:
    model: Contract

    def __init__(self, model: Contract):
        self.model = model

    def can_train(self, trainer: HexAddress, round_id: int) -> bool:
        return self.model.functions.canTrain(trainer, round_id).call()

    def current_round(self) -> int:
        return self.model.functions.currentRound().call()

    def get_eip712_domain(self) -> EIP712Domain:
        resp = self.model.functions.eip712Domain().call()
        return EIP712Domain(
            fields=resp[0],
            name=resp[1],
            version=resp[2],
            chainId=resp[3],
            verifyingContract=resp[4],
            salt=resp[5],
            extensions=resp[6],
        )
