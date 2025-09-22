"""Base abstract Metric Storage class"""

from abc import ABC, abstractmethod

from flwr.common import Parameters, Scalar


class BaseMetricStorage(ABC):
    """An abstract base class (ABC) for metric storage backends.

    This class defines the standard interface that all metric storage implementations
    must adhere to. It ensures that different storage mechanisms (e.g., in-memory,
    file-based, or a dedicated server) can be used interchangeably by a strategy that
    requires metric (i.e. `MetricStorageStrategy`).

    To create a custom metric storage backend, you must subclass `BaseMetricStorage`
    and provide concrete implementations for all of its abstract methods.
    """

    @abstractmethod
    def write_metrics(self, server_round: int, metrics: dict[str, Scalar]) -> None:
        """Writes a dictionary of metrics for a specific server round.

        This method is called to persist the metrics (e.g., accuracy, precision)
        during various phases.

        Args:
            server_round (int): The current round of federated learning.
            metrics (dict[str, Scalar]): A dictionary mapping metric names
                (e.g., "accuracy") to their scalar values.
        """

    @abstractmethod
    def update_current_round_model(self, parameters: Parameters) -> None:
        """Updates the model parameters from the most recent round.

        This is used for temporarily keeping the parameters during
        the `configure_evaluate` phase so that it can later be written to the disk
        if the evaluation indicates it is the best model.

        Args:
            parameters (Parameters): The model parameters from the current round
                to be saved.
        """

    @abstractmethod
    def update_best_model(self, server_round: int, loss: float) -> None:
        """Updates the stored "best" model if the current model's loss is lower.

        This method is responsible for tracking the best-performing model
        encountered throughout the entire federated learning process. It should
        compare the provided loss with a stored minimum loss and overwrite the
        saved "best model" if the new one is better.

        Args:
            server_round (int): The server round that produced this model.
            loss (float): The loss value of the current model, used as the
                primary criterion for determining the "best" model.
        """
