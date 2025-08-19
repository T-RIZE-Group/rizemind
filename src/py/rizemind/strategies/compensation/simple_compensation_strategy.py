from logging import INFO
from typing import cast

from eth_typing import Address
from flwr.common import EvaluateIns, EvaluateRes, FitIns, Parameters
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from rizemind.strategies.compensation.typings import SupportsDistribute
from web3 import Web3


class SimpleCompensationStrategy(Strategy):
    strategy: Strategy
    model: SupportsDistribute

    def __init__(self, strategy: Strategy, model: SupportsDistribute) -> None:
        self.strategy = strategy
        self.model = model

    def calculate(self, client_ids: list[Address]):
        log(INFO, "calculate: calculating compensations.")
        return [(Web3.to_checksum_address(id), 1.0) for id in client_ids]

    def aggregate_fit(self, server_round, results, failures):
        log(
            INFO,
            "aggregate_fit: training results received from the clients",
        )
        log(INFO, "aggregate_fit: initializing aggregation")
        trainer_scores = self.calculate(
            [cast(Address, res.metrics["trainer_address"]) for _, res in results]
        )
        self.model.distribute(trainer_scores)
        return self.strategy.aggregate_fit(server_round, results, failures)

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        log(
            INFO,
            "initialize_parameters: first training phase started",
        )
        log(
            INFO,
            "initialize_parameters: initializing model parameters for the first time",
        )
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
