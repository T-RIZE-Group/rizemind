from typing import Any, TypedDict, Unpack

from eth_typing import ChecksumAddress
from web3 import Web3
from web3.contract import Contract


class FromAddressKwargs(TypedDict):
    w3: Web3
    address: ChecksumAddress


class FactoryKwargs(TypedDict):
    w3: Web3
    address: ChecksumAddress
    abi: Any


def contract_factory(**kwargs: Unpack[FactoryKwargs]) -> Contract:
    return kwargs["w3"].eth.contract(
        address=kwargs["address"],
        abi=kwargs["abi"],
    )


class BaseContract:
    contract: Contract

    def __init__(self, *, contract: Contract):
        self.contract = contract

    @property
    def w3(self) -> Web3:
        return self.contract.w3

    @property
    def address(self) -> ChecksumAddress:
        return self.contract.address
