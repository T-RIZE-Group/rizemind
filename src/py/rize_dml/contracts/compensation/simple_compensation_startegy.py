from eth_typing import Address
from flwr.common import EvaluateIns, EvaluateRes, FitIns, Parameters
from flwr.server.client_manager import ClientManager
from rize_dml.authentication.eth_account_strategy import EthAccountStrategy
from rize_dml.contracts.compensation.compensation_strategy import (
    CompensationStrategy,
)
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy


class SimpleCompensationStrategy(CompensationStrategy):
    def __init__(self, strategy: EthAccountStrategy):
        CompensationStrategy.__init__(self, strategy)

    def calculate(self, client_ids: list[Address]):
        return client_ids, [1 for _ in client_ids]

    def aggregate_fit(self, server_round, results, failures):
        whitelisted: list[tuple[ClientProxy, FitRes]] = []
        whitelisted_address: list[Address] = []
        for client, res in results:
            signer = self.strategy._recover_signer(res, server_round)
            if self.strategy.model.can_train(signer, server_round):
                whitelisted.append((client, res))
                whitelisted_address.append(signer)
        trainers, contributions = self.calculate(whitelisted_address)
        self.strategy.model.distribute(trainers, contributions)
        return self.strategy.strat.aggregate_fit(server_round, whitelisted, failures)

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
