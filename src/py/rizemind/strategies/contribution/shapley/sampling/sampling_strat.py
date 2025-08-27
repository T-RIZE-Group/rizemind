from abc import ABC, abstractmethod

from flwr.common import FitRes
from flwr.server.client_proxy import ClientProxy
from rizemind.strategies.contribution.shapley.trainer_set import (
    TrainerSet,
)


class ShapleySamplingStrategy(ABC):
    @abstractmethod
    def sample_trainer_sets(
        self, server_round: int, results: list[tuple[ClientProxy, FitRes]]
    ) -> list[TrainerSet]:
        pass

    @abstractmethod
    def get_sets(self, round_id: int) -> list[TrainerSet]:
        pass

    @abstractmethod
    def get_set(self, round_id: int, id: str) -> TrainerSet:
        pass
