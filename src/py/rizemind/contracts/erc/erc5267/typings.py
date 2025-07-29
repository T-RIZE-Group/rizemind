from abc import ABC, abstractmethod
from typing import NamedTuple

from eth_typing import ChecksumAddress


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
