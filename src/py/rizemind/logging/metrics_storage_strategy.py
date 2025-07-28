from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.server import ClientManager
from flwr.server.strategy import Strategy
from rizemind.authentication.eth_account_strategy import ClientProxy
from rizemind.logging.metrics_storage import MetricsStorage


class MetricsStorageStrategy(Strategy):
    def __init__(self, strategy: Strategy, metrics_storage: MetricsStorage):
        self.strategy = strategy
        self.metrics_storage = metrics_storage

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        res = self.strategy.configure_fit(server_round, parameters, client_manager)
        metrics_list = [metrics.config for _, metrics in res]
        for metrics in metrics_list:
            self.metrics_storage.write_metrics(server_round, metrics)
        return res

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        parameters, metrics = self.strategy.aggregate_fit(
            server_round, results, failures
        )
        self.metrics_storage.write_metrics(server_round, metrics)
        return (parameters, metrics)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        res = self.strategy.configure_evaluate(server_round, parameters, client_manager)
        metrics_list = [metrics.config for _, metrics in res]
        for metrics in metrics_list:
            self.metrics_storage.write_metrics(server_round, metrics)
        return res

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        evaluation, metrics = self.strategy.aggregate_evaluate(
            server_round, results, failures
        )
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
        if evaluation_result is None:
            return None
        self.metrics_storage.write_metrics(server_round, {"loss": evaluation_result[0]})
        self.metrics_storage.write_metrics(server_round, evaluation_result[1])
        return evaluation_result
