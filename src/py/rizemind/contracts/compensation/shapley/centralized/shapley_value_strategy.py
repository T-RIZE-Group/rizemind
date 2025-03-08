from typing import cast
from eth_typing import Address
from flwr.server.strategy import Strategy
from rizemind.contracts.compensation.shapley.shapley_value_strategy import (
    ShapleyValueStrategy,
)
from rizemind.contracts.models.model_registry_v1 import ModelRegistryV1
from flwr.common.typing import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


class CentralShapleyValueStrategy(ShapleyValueStrategy):
    last_round_parameters: Parameters

    def __init__(
        self, strategy: Strategy, model: ModelRegistryV1, initial_parameters: Parameters
    ) -> None:
        ShapleyValueStrategy.__init__(self, strategy, model, initial_parameters)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        # Create Coalitions
        coalitions = self.create_coalitions(server_round, results)

        # Evaluate Coalitions
        evaluated_coalitions: list[tuple[list[Address], float]] = []
        for coalition in coalitions:
            evaluation = self.strategy.evaluate(server_round, coalition.parameters)
            if evaluation is None:
                ValueError(
                    "Evaluation cannot be None. Coalition members:", coalition.members
                )
            evaluation = cast(tuple[float, dict], evaluation)
            evaluated_coalitions.append((coalition.members, evaluation[0]))

        # Do reward calculation
        player_scores = self.compute_contributions(evaluated_coalitions)
        trainers, contributions = self.normalize_contribution_scores(player_scores)
        self.model.distribute(trainers, contributions)

        res = self.strategy.aggregate_fit(server_round, results, failures)
        self.last_round_parameters = cast(Parameters, res[0])
        return res

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
