from abc import abstractmethod
from eth_typing import Address
from flwr.server.strategy import Strategy
from rizemind.contracts.models.model_registry_v1 import ModelRegistryV1


class CompensationStrategy(Strategy):
    strategy: Strategy
    model: ModelRegistryV1

    def __init__(self, strategy: Strategy, model: ModelRegistryV1) -> None:
        self.strategy = strategy
        self.model = model

    @abstractmethod
    def calculate(self, client_ids: list[Address]) -> tuple[list[Address], list[int]]:
        "Calculates and returns the compensation for each client in the list"
