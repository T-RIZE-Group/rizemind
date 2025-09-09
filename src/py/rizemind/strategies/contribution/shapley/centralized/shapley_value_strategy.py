from flwr.common import Code, Status
from flwr.common.typing import EvaluateIns, EvaluateRes, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from rizemind.strategies.contribution.shapley.shapley_value_strategy import (
    ShapleyValueStrategy,
    SupportsShapleyValueStrategy,
)


class CentralShapleyValueStrategy(ShapleyValueStrategy):
    def __init__(
        self,
        strategy: Strategy,
        model: SupportsShapleyValueStrategy,
        **kwargs,
    ) -> None:
        ShapleyValueStrategy.__init__(self, strategy, model, **kwargs)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        return []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, bool | bytes | float | int | str]]:
        return None, {}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        coalitions = self.get_coalitions()
        for coalition in coalitions:
            evaluation = self.strategy.evaluate(server_round, coalition.parameters)
            if evaluation is None:
                raise ValueError(
                    "Evaluation cannot be None. Coalition members:", coalition.members
                )
            loss, metrics = evaluation
            coalition.insert_res(
                EvaluateRes(
                    loss=loss,
                    metrics=metrics,
                    status=Status(code=Code.OK, message=""),
                    num_examples=0,
                )
            )

        return self.close_round(server_round)
