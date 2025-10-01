import statistics
from collections.abc import Callable

from eth_typing import ChecksumAddress
from flwr.common import EvaluateRes
from flwr.common.typing import Parameters, Scalar


class TrainerSet:
    """A set of trainers forming a coalition.

    Attributes:
        id: Unique identifier for this trainer set.
        members: List of trainer addresses that are members of this set.
    """

    order: int
    id: str
    members: list[ChecksumAddress]

    def __init__(
        self,
        id: str,
        members: list[ChecksumAddress],
        order: int = 0,
    ) -> None:
        """Initialize a trainer set.

        Args:
            id: Unique identifier for this set.
            members: List of trainer addresses in this set.
        """
        self.id = id
        self.members = members
        self.order = order

    def size(self) -> int:
        """Get the number of trainers in this set.

        Returns:
            The count of member trainers.
        """
        return len(self.members)


class TrainerSetAggregate(TrainerSet):
    """A trainer set with aggregated model parameters and configuration dictionary.

    Attributes:
        parameters: Aggregated model parameters for this coalition.
        config: Configuration dictionary.
    """

    parameters: Parameters
    config: dict[str, Scalar]
    _evaluation_res: list[EvaluateRes]

    def __init__(
        self,
        id: str,
        members: list[ChecksumAddress],
        parameters: Parameters,
        config: dict[str, Scalar],
        order: int = 0,
    ) -> None:
        """Initialize a trainer set aggregate.

        Args:
            id: Unique identifier for this coalition.
            members: List of trainer addresses in this coalition.
            parameters: Aggregated model parameters.
            config: Configuration dictionary.
        """
        super().__init__(id, members=members, order=order)
        self.parameters = parameters
        self.config = config
        self._evaluation_res = []

    def insert_res(self, eval_res: EvaluateRes):
        """Add an evaluation result to this coalition.

        Args:
            eval_res: The evaluation result to store.
        """
        self._evaluation_res.append(eval_res)

    def get_loss(
        self,
        aggregator: Callable[[list[float]], float] = statistics.mean,
    ):
        """Get the aggregated loss for this coalition.

        Args:
            aggregator: Function to aggregate multiple loss values.
            Defaults to mean.

        Returns:
            The aggregated loss value, or infinity if no evaluations exist.
        """
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
        """Get an aggregated metric value for this coalition.

        Args:
            name: The metric name to retrieve.
            default: Default value to return if metric is unavailable.
            aggregator: Function to aggregate multiple metric values.

        Returns:
            The aggregated metric value, or default if not all evaluations
            contain this metric.
        """
        if not self._evaluation_res or len(self._evaluation_res) == 0:
            return default

        metric_values = [res.metrics.get(name) for res in self._evaluation_res]
        valid_metrics = [v if v is not None else default for v in metric_values]

        return aggregator(valid_metrics)


class TrainerSetAggregateStore:
    """Storage system for trainer set aggregates.

    Maintains a collection of coalition aggregates indexed by their ids.

    Attributes:
        set_aggregates: Dictionary mapping coalition IDs to their aggregates.
    """

    set_aggregates: dict[str, TrainerSetAggregate]

    def __init__(self) -> None:
        """Initialize an empty aggregate store."""
        self.set_aggregates = {}

    def insert(self, aggregate: TrainerSetAggregate) -> None:
        """Insert a coalition aggregate in the store.

        Args:
            aggregate: The coalition aggregate to store.
        """
        self.set_aggregates[aggregate.id] = aggregate

    def get_sets(self) -> list[TrainerSetAggregate]:
        """Get all coalition aggregates.

        Returns:
            List of all stored coalition aggregates.
        """
        return list(self.set_aggregates.values())

    def clear(self) -> None:
        """Remove all coalition aggregates from the store."""
        self.set_aggregates = {}

    def get_set(self, id: str) -> TrainerSetAggregate:
        """Retrieve a specific coalition aggregate by ID.

        Args:
            id: The unique identifier of the coalition.

        Returns:
            The requested coalition aggregate.

        Raises:
            Exception: If no coalition with the given ID exists.
        """
        if id in self.set_aggregates:
            return self.set_aggregates[id]
        raise Exception(
            f"Coalition {id} not found, available sets: {self.set_aggregates.keys()}"
        )

    def get_set_by_order(self, order: int) -> list[TrainerSetAggregate]:
        return [set for set in self.set_aggregates.values() if set.order == order]

    def is_available(self, id: str) -> bool:
        """Check if a coalition aggregate exists in the store.

        Args:
            id: The unique identifier to check.

        Returns:
            True if the coalition exists, False otherwise.
        """
        return id in self.set_aggregates
