import statistics
from collections.abc import Callable

from eth_typing import ChecksumAddress
from flwr.common import EvaluateRes
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
    _evaluation_res: list[EvaluateRes]

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
        self._evaluation_res = []

    def insert_res(self, eval_res: EvaluateRes):
        self._evaluation_res.append(eval_res)

    def get_loss(
        self,
        aggregator: Callable[[list[float]], float] = statistics.mean,
    ):
        if len(self._evaluation_res) == 0:
            return float("Inf")
        losses = [res.loss for res in self._evaluation_res]
        return aggregator(losses)

    def get_metric(
        self,
        name: str,
        default: Scalar,
        aggregator: Callable,
    ):
        if not self._evaluation_res:
            return default

        metric_values = [res.metrics.get(name) for res in self._evaluation_res]
        valid_metrics = [v for v in metric_values if v is not None]

        if len(valid_metrics) != len(self._evaluation_res):
            return default

        return aggregator(valid_metrics)


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
