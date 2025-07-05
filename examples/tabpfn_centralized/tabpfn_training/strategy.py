from pathlib import Path
from typing import cast

import torch
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from tabpfn.model.loading import load_model
from tabpfn_centralized.tabpfn_training.client_app import (
    load_weights_into_model,
)


class SimpleTabPFNRegressorStrategy(Strategy):
    strategy: Strategy
    base_model_path: str

    def __init__(self, *, strategy: Strategy, base_model_path: str) -> None:
        self.strategy = strategy
        self.base_model_path

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        return self.strategy.configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self, server_round, results, failures
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        parameters, res = self.strategy.aggregate_fit(server_round, results, failures)
        model, criterion, checkpoint_config = load_model(
            path=Path(self.base_model_path), model_seed=42
        )
        model.criterion = criterion
        np_parameters = parameters_to_ndarrays(cast(Parameters, parameters))
        model = load_weights_into_model(
            model=model, parameters=np_parameters, config={}
        )
        torch.save(
            dict(
                state_dict=model.state_dict(),
                config=checkpoint_config.__dict__,
            ),
            str(self.base_model_path),
        )
        return parameters, res

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
        return self.strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        return self.strategy.evaluate(server_round, parameters)
