from typing import cast
from eth_typing import Address
from flwr.common import EvaluateIns, EvaluateRes, FitIns, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy

from rizemind.contracts.compensation.compensation_strategy import (
    CompensationStrategy,
)
from flwr.server.client_proxy import ClientProxy
from rizemind.contracts.models.model_registry_v1 import ModelRegistryV1


class SimpleCompensationStrategy(CompensationStrategy):
    def __init__(self, strategy: Strategy, model: ModelRegistryV1):
        CompensationStrategy.__init__(self, strategy, model)

    def calculate(self, client_ids: list[Address]):
        return client_ids, [1 for _ in client_ids]

    def aggregate_fit(self, server_round, results, failures):
        trainers, contributions = self.calculate(
            [cast(Address, res.metrics["trainer_address"]) for _, res in results]
        )
        self.model.distribute(trainers, contributions)
        return self.strategy.aggregate_fit(server_round, results, failures)

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        return self.strategy.configure_fit(server_round, parameters, client_manager)

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
    ) -> tuple[float | None, dict[str, bool | bytes | float | int | str]]:
        return self.strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        return self.strategy.evaluate(server_round, parameters)
