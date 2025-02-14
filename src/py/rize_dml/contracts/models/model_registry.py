from typing import List, NamedTuple
from eth_typing import Address
from web3.contract import Contract


class EIP712Domain(NamedTuple):
    fields: bytes
    name: str
    version: str
    chainId: int
    verifyingContract: str
    salt: bytes
    extensions: List[int]


class ModelRegistry:
    model: Contract

    def __init__(self, model: Contract):
        self.model = model

    def can_train(self, trainer: Address, round_id: int) -> bool:
        return self.model.functions.canTrain(trainer, round_id).call()

    def current_round(self) -> int:
        return self.model.functions.currentRound().call()

    def get_eip712_domain(self) -> EIP712Domain:
        resp = self.model.functions.eip712Domain().call()
        return EIP712Domain(
            fields=resp.fields,
            name=resp.name,
            version=resp.version,
            chainId=resp.chainId,
            verifyingContract=resp.verifyingContract,
            salt=resp.salt,
            extensions=resp.extensions,
        )
