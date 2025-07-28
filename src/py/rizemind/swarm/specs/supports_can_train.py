from abc import ABC, abstractmethod

from eth_typing import HexAddress


class SupportsCanTrain(ABC):
    @abstractmethod
    def can_train(self, trainer: HexAddress, round_id: int) -> bool:
        pass
