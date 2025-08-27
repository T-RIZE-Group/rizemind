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

    def size(self) -> int:
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


class TrainerSetAggregateStore:
    set_aggregates: dict[str, TrainerSetAggregate]

    def __init__(self) -> None:
        self.set_aggregates = {}

    def insert(self, aggregate: TrainerSetAggregate) -> None:
        self.set_aggregates[aggregate.id] = aggregate

    def get_sets(self) -> list[TrainerSetAggregate]:
        return list(self.set_aggregates.values())

    def clear(self) -> None:
        self.set_aggregates = {}

    def get_set(self, id: str) -> TrainerSetAggregate:
        if id in self.set_aggregates:
            return self.set_aggregates[id]
        raise Exception(f"Coalition {id} not found")

    def is_available(self, id: str) -> bool:
        return id in self.set_aggregates
