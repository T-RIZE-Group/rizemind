from flwr.server.strategy import Strategy
from rizemind.contracts.models.model_registry_v1 import ModelRegistryV1


class CompensationStrategy(Strategy):
    strategy: Strategy
    model: ModelRegistryV1

    def __init__(self, strategy: Strategy, model: ModelRegistryV1) -> None:
        self.strategy = strategy
        self.model = model
