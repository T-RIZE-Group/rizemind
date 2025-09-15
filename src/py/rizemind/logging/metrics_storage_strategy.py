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
from rizemind.logging.base_metrics_storage import BaseMetricsStorage


class MetricPhases(Enum):
    AGGREGATE_FIT = 1
    AGGREGATE_EVALUATE = 2
    EVALUATE = 3


class MetricsStorageStrategy(Strategy):
    def __init__(
        self,
        strategy: Strategy,
        metrics_storage: BaseMetricsStorage,
        enabled_metric_phases: list[MetricPhases] = [
            MetricPhases.AGGREGATE_FIT,
            MetricPhases.AGGREGATE_EVALUATE,
            MetricPhases.EVALUATE,
        ],
        save_best_model: bool = True,
    ):
        self.strategy = strategy
        self.metrics_storage = metrics_storage
        self.enabled_metric_phases = enabled_metric_phases
        self.save_best_model = save_best_model

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        return self.strategy.configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        parameters, metrics = self.strategy.aggregate_fit(
            server_round, results, failures
        )
        if self.save_best_model:
            if parameters is None:
                log(
                    level=WARNING,
                    msg="No model parameter provided, best model will not be saved.",
                )
            else:
                self.metrics_storage.update_current_round_model(parameters)
        if MetricPhases.AGGREGATE_FIT in self.enabled_metric_phases:
            self.metrics_storage.write_metrics(server_round, metrics)
        return (parameters, metrics)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        return self.strategy.configure_evaluate(
            server_round, parameters, client_manager
        )

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        evaluation, metrics = self.strategy.aggregate_evaluate(
            server_round, results, failures
        )
        if MetricPhases.AGGREGATE_EVALUATE in self.enabled_metric_phases:
            if evaluation is not None:
                self.metrics_storage.write_metrics(
                    server_round, {"loss_aggregated": evaluation}
                )
            self.metrics_storage.write_metrics(server_round, metrics)
        return (evaluation, metrics)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, Scalar]] | None:
        evaluation_result = self.strategy.evaluate(server_round, parameters)
        if MetricPhases.EVALUATE in self.enabled_metric_phases:
            if evaluation_result is None:
                return None
            self.metrics_storage.write_metrics(
                server_round, {"loss": evaluation_result[0]}
            )
            self.metrics_storage.write_metrics(server_round, evaluation_result[1])
        return evaluation_result
