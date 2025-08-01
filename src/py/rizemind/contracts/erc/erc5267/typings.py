from abc import ABC, abstractmethod

from eth_typing import ChecksumAddress
from pydantic import BaseModel


class EIP712DomainMinimal(BaseModel):
    name: str
    version: str
    chainId: int
    verifyingContract: ChecksumAddress


class EIP712Domain(EIP712DomainMinimal):
    fields: bytes
    salt: bytes
    extensions: list[int]


class SupportsERC5267(ABC):
    @abstractmethod
    def get_eip712_domain(self) -> EIP712Domain:
        pass
