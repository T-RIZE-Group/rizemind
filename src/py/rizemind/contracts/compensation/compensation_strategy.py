from flwr.server.strategy import Strategy
from rizemind.contracts.models.model_meta_v1 import ModelMetaV1


class CompensationStrategy(Strategy):
    strategy: Strategy
    model: ModelMetaV1

    def __init__(self, strategy: Strategy, model: ModelMetaV1) -> None:
        self.strategy = strategy
        self.model = model
