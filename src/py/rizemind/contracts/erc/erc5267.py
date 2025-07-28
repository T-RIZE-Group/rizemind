from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import NamedTuple, Unpack

from eth_typing import ChecksumAddress
from rizemind.contracts.abi_helper import load_abi
from rizemind.contracts.base_contract import (
    BaseContract,
    FromAddressKwargs,
    contract_factory,
)
from web3 import Web3
from web3.contract import Contract

abi = load_abi(Path(os.path.dirname(__file__)) / "./erc5267.abi.json")


class EIP712Domain(NamedTuple):
    fields: bytes
    name: str
    version: str
    chainId: int
    verifyingContract: ChecksumAddress
    salt: bytes
    extensions: list[int]


class SupportsERC5267(ABC):
    @abstractmethod
    def get_eip712_domain(self) -> EIP712Domain:
        pass


class ERC5267(SupportsERC5267, BaseContract):
    def __init__(self, contract: Contract):
        super().__init__(contract=contract)

    @staticmethod
    def from_address(**kwargs: Unpack[FromAddressKwargs]) -> "ERC5267":
        return ERC5267(contract_factory(**kwargs, abi=abi))

    def get_eip712_domain(self) -> EIP712Domain:
        resp = self.contract.functions.eip712Domain().call()
        return EIP712Domain(
            fields=resp[0],
            name=resp[1],
            version=resp[2],
            chainId=resp[3],
            verifyingContract=Web3.to_checksum_address(resp[4]),
            salt=resp[5],
            extensions=resp[6],
        )
