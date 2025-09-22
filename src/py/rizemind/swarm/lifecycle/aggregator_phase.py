from abc import ABC, abstractmethod
from typing import Self

from flwr.common import Context
from flwr.server import Grid


class AggregatorPhase(ABC):
    """
    AggregatorPhase is a base class for all aggregator phases.
    """

    @abstractmethod
    def execute(self, grid: Grid, context: Context) -> Self:
        pass

    @abstractmethod
    def can_execute(self) -> bool:
        pass

    @abstractmethod
    def next_phase(self, grid: Grid, context: Context) -> Self | None:
        pass
