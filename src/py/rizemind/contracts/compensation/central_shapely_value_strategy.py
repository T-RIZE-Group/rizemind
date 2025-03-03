from typing import cast
from eth_typing import Address
from flwr.server.strategy import Strategy
from rizemind.contracts.compensation.shapely_value_strategy import ShapelyValueStrategy
from rizemind.contracts.models.model_registry_v1 import ModelRegistryV1
from flwr.common.typing import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


class CentralShapelyValueStrategy(ShapelyValueStrategy):
    def __init__(self, strategy: Strategy, model: ModelRegistryV1) -> None:
        super().__init__(strategy, model)

    def evaluate_coalition(
        self, server_round: int, results: list[tuple[ClientProxy, FitRes]]
    ):
        aggregated_parameters = self.strategy.aggregate_fit(server_round, results, [])[
            0
        ]
        if aggregated_parameters is None:
            raise ValueError("Aggregated Parameteres should not be None.")
        evaluted_result = super().evaluate(server_round, aggregated_parameters)
        if evaluted_result is None:
            raise ValueError("Aggregated evaluated result should not be None.")
        return evaluted_result[0]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        coalitions = self.create_coalitions(results)

        # Evaluate Coalitions
        evaluated_coalitions = [
            self.evaluate_coalition(server_round, coalition) for coalition in coalitions
        ]

        # Do reward calculation -> compute_contributions
        addresses = [
            [
                cast(Address, result[1].metrics["trainer_address"])
                for result in coalition
            ]
            for coalition in coalitions
        ]
        coalition_and_scores = [
            (address, score) for address, score in zip(addresses, evaluated_coalitions)
        ]

        trainers_and_contributions = self.compute_contributions(coalition_and_scores)
        trainers = [trainer for trainer, _ in trainers_and_contributions]
        contributions = [contribution for _, contribution in trainers_and_contributions]
        self.model.distribute(trainers, contributions)

        # What is the best way to aggregate
        return self.strategy.aggregate_fit(server_round, results, failures)

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        return super().initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        return super().configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, bool | bytes | float | int | str]]:
        return super().aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        return super().evaluate(server_round, parameters)
