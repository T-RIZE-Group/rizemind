from abc import ABC, abstractmethod

from flwr.common import Parameters, Scalar


class BaseMetricsStorage(ABC):
    @abstractmethod
    def write_metrics(self, server_round: int, metrics: dict[str, Scalar]):
        pass

    @abstractmethod
    def update_current_round_model(self, parameters: Parameters):
        pass

    @abstractmethod
    def update_best_model(self, server_round: int, loss: float):
        pass
