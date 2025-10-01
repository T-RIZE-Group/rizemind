"""Shapley value contribution calculation for federated learning."""

# TODO:
# Fix the following circular dependency:
# calculators/__init__.py → imports ShapleyValueCalculator from calculators.shapley_value
# calculators/shapley_value.py → imports from calculators.calculator
# calculators/calculator.py → imports from shapley.trainer_mapping
# When Python imports shapley.trainer_mapping, it first executes shapley/__init__.py
# shapley/__init__.py → imports from shapley.shapley_value_strategy
# shapley/shapley_value_strategy.py → imports ShapleyValueCalculator from calculators
# This results in a cycle.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rizemind.strategies.contribution.shapley.shapley_value_strategy import (
        ShapleyValueStrategy,
    )
    from rizemind.strategies.contribution.shapley.trainer_mapping import (
        ParticipantMapping,
    )
    from rizemind.strategies.contribution.shapley.trainer_set import (
        TrainerSet,
        TrainerSetAggregate,
        TrainerSetAggregateStore,
    )


def __getattr__(name: str):
    """Lazy import to avoid circular dependencies."""
    if name == "ShapleyValueStrategy":
        from rizemind.strategies.contribution.shapley.shapley_value_strategy import (
            ShapleyValueStrategy,
        )

        return ShapleyValueStrategy
    elif name == "ParticipantMapping":
        from rizemind.strategies.contribution.shapley.trainer_mapping import (
            ParticipantMapping,
        )

        return ParticipantMapping
    elif name == "TrainerSetAggregate":
        from rizemind.strategies.contribution.shapley.trainer_set import (
            TrainerSetAggregate,
        )

        return TrainerSetAggregate
    elif name == "TrainerSetAggregateStore":
        from rizemind.strategies.contribution.shapley.trainer_set import (
            TrainerSetAggregateStore,
        )

        return TrainerSetAggregateStore
    elif name == "TrainerSet":
        from rizemind.strategies.contribution.shapley.trainer_set import TrainerSet

        return TrainerSet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ShapleyValueStrategy",
    "ParticipantMapping",
    "TrainerSetAggregate",
    "TrainerSetAggregateStore",
    "TrainerSet",
]
