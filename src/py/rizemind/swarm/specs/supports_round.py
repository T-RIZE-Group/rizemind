from abc import ABC, abstractmethod

from hexbytes import HexBytes
from pydantic import BaseModel


class RoundMetrics(BaseModel):
    n_trainers: int
    model_score: float
    total_contributions: float


class RoundSummary(BaseModel):
    round_id: int
    finished: bool
    metrics: RoundMetrics | None


class SupportsRound(ABC):
    @abstractmethod
    def current_round(self) -> int:
        pass

    @abstractmethod
    def next_round(
        self,
        round_id: int,
        n_trainers: int,
        model_score: float,
        total_contributions: float,
    ) -> HexBytes:
        pass
