from abc import ABC, abstractmethod
from collections.abc import Callable

from eth_typing import ChecksumAddress
from pydantic import BaseModel
from rizemind.strategies.contribution.shapley.trainer_mapping import ParticipantMapping
from rizemind.strategies.contribution.shapley.trainer_set import (
    TrainerSetAggregate,
    TrainerSetAggregateStore,
)


def default_coalition_to_score(set: TrainerSetAggregate) -> float:
    score = set.get_loss()
    if score is None:
        raise Exception(f"Trainer set ID {set.id} not evaluated")
    return score


class PlayerScore(BaseModel):
    trainer_address: ChecksumAddress
    score: float


class ContributionCalculator(ABC):
    @abstractmethod
    def get_scores(
        self,
        *,
        participant_mapping: ParticipantMapping,
        store: TrainerSetAggregateStore,
        coalition_to_score_fn: Callable[[TrainerSetAggregate], float]
        | None = default_coalition_to_score,
    ) -> dict[ChecksumAddress, PlayerScore]:
        pass
