from web3.contract import Contract
from typing import NamedTuple
from rizemind.contracts.abi.erc5267 import abi
from web3 import Web3


class EIP712Domain(NamedTuple):
    fields: bytes
    name: str
    version: str
    chainId: int
    verifyingContract: str
    salt: bytes
    extensions: list[int]


class ERC5267:
    erc5267: Contract

    def __init__(self, contract: Contract):
        self.erc5267 = contract

    @staticmethod
    def from_address(address: str, w3: Web3) -> "ERC5267":
        checksum_address = Web3.to_checksum_address(address)
        return ERC5267(w3.eth.contract(address=checksum_address, abi=abi))

    def get_eip712_domain(self) -> EIP712Domain:
        resp = self.erc5267.functions.eip712Domain().call()
        return EIP712Domain(
            fields=resp[0],
            name=resp[1],
            version=resp[2],
            chainId=resp[3],
            verifyingContract=resp[4],
            salt=resp[5],
            extensions=resp[6],
        )
