from abc import ABC, abstractmethod

from eth_typing import Address
from hexbytes import HexBytes


class SupportsDistribute(ABC):
    @abstractmethod
    def distribute(self, trainer_scores: list[tuple[Address, float]]) -> HexBytes:
        pass
