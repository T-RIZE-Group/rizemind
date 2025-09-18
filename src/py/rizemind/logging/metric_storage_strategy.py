from enum import Enum
from logging import WARNING

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    log,
)
from flwr.server import ClientManager
from flwr.server.strategy import Strategy

from rizemind.authentication.eth_account_strategy import ClientProxy
from rizemind.logging.base_metric_storage import BaseMetricStorage


class MetricPhases(Enum):
    """MetricPhases based on phases that the ServerApp receives metrics from the clients"""

    AGGREGATE_FIT = 1
    AGGREGATE_EVALUATE = 2
    EVALUATE = 3


class MetricStorageStrategy(Strategy):
    """The `MetricStorageStrategy` capable of logging metrics at `MetricPhases` given a metric storage."""

    def __init__(
        self,
        strategy: Strategy,
        metrics_storage: BaseMetricStorage,
        enabled_metric_phases: list[MetricPhases] = [
            MetricPhases.AGGREGATE_FIT,
            MetricPhases.AGGREGATE_EVALUATE,
            MetricPhases.EVALUATE,
        ],
        save_best_model: bool = True,
    ):
        """Initializes a MetricStorageStrategy.

        Args:
            strategy: The base Flower strategy to wrap.
            metrics_storage: The storage object for writing metrics.
            enabled_metric_phases: A list of phases during which metrics will be logged. Defaults to all phases.
            save_best_model: A boolean indicating whether to save the best model parameters. Defaults to True.
        """
        self._strategy = strategy
        self._metrics_storage = metrics_storage
        self._enabled_metric_phases = enabled_metric_phases
        self._save_best_model = save_best_model

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        return self._strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        return self._strategy.configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate fit results and log metrics.

        If the `save_best_model` is enabled, then the aggregated parameters will
            be kept in memory to be used later on if they represent the best model.
        If logging is enabled with `AGGREGATE_FIT`, then it will log the metrics to
            the given metric storage.

        Args:
            server_round: The current round of federated learning.
            results: Successful fit results from clients.
            failures: Failures from clients during fitting.

        Returns:
            A tuple containing the aggregated parameters and a dictionary of metrics.
        """
        parameters, metrics = self._strategy.aggregate_fit(
            server_round, results, failures
        )
        if self._save_best_model:
            if parameters is None:
                log(
                    level=WARNING,
                    msg="No model parameter provided, best model will not be saved.",
                )
            else:
                self._metrics_storage.update_current_round_model(parameters)
        if MetricPhases.AGGREGATE_FIT in self._enabled_metric_phases:
            self._metrics_storage.write_metrics(server_round, metrics)
        return (parameters, metrics)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        return self._strategy.configure_evaluate(
            server_round, parameters, client_manager
        )

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """Aggregate evaluation results and log metrics.

        If the `save_best_model` is enabled, then the last best evaluation is compared
            with the current evaluation to log the parameters of the best model.
        If logging is enabled with `AGGREGATE_EVALUATE`, then it will log the metrics to
            the given metric storage.

        Args:
            server_round: The current round of federated learning.
            results: Successful evaluation results from clients.
            failures: Failures from clients during evaluation.

        Returns:
            A tuple containing the aggregated loss and a dictionary of metrics.
        """
        evaluation, metrics = self._strategy.aggregate_evaluate(
            server_round, results, failures
        )
        if self._save_best_model:
            if evaluation is None:
                log(
                    level=WARNING,
                    msg="No metric provided for evaluation, best model will not be saved.",
                )
            else:
                self._metrics_storage.update_best_model(
                    server_round=server_round, loss=evaluation
                )
        if MetricPhases.AGGREGATE_EVALUATE in self._enabled_metric_phases:
            if evaluation is not None:
                self._metrics_storage.write_metrics(
                    server_round, {"loss_aggregated": evaluation}
                )
            self._metrics_storage.write_metrics(server_round, metrics)
        return (evaluation, metrics)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, Scalar]] | None:
        """Evaluate model parameters on the server and log metrics.

        If logging is enabled with `EVALUATE`, then it will log the metrics to
            the given metric storage.

        Args:
            server_round: The current round of federated learning.
            parameters: The current global model parameters to be evaluated.

        Returns:
            An optional tuple containing the loss and a dictionary of metrics from the evaluation.
        """
        evaluation_result = self._strategy.evaluate(server_round, parameters)
        if MetricPhases.EVALUATE in self._enabled_metric_phases:
            if evaluation_result is None:
                return None
            self._metrics_storage.write_metrics(
                server_round, {"loss": evaluation_result[0]}
            )
            self._metrics_storage.write_metrics(server_round, evaluation_result[1])
        return evaluation_result
