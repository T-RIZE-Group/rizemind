from eth_typing import ChecksumAddress
from flwr.common.typing import Parameters, Scalar


class TrainerSet:
    id: str
    members: list[ChecksumAddress]

    def __init__(
        self,
        id: str,
        members: list[ChecksumAddress],
    ) -> None:
        self.id = id
        self.members = members

    def size(self):
        return len(self.members)


class TrainerSetAggregate(TrainerSet):
    parameters: Parameters
    config: dict[str, Scalar]
    loss: float | None
    metrics: dict[str, Scalar] | None

    def __init__(
        self,
        id: str,
        members: list[ChecksumAddress],
        parameters: Parameters,
        config: dict[str, Scalar],
    ) -> None:
        super().__init__(id, members=members)
        self.parameters = parameters
        self.config = config

    def get_loss(self):
        return self.loss or float("Inf")

    def get_metric(self, name: str, default):
        if self.metrics:
            return self.metrics[name] or default
        return default
