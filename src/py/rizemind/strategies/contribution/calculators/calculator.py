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
    """Returns the loss value of a set as its score.

    Raises:
        Exception: if the coalition was not evaluated.
    """
    score = set.get_loss()
    if score is None:
        raise Exception(f"Trainer set ID {set.id} not evaluated")
    return score


class PlayerScore(BaseModel):
    """The contribution score for a single participant.

    Represents the calculated contribution value for a participant and
    their blockchain address.

    Attributes:
        trainer_address: The blockchain address of the participant.
        score: The calculated contribution score value.
    """

    trainer_address: ChecksumAddress
    score: float


class ContributionCalculator(ABC):
    """Participant contribution calculator.

    Defines the interface that all contribution calculation algorithms must
    implement. Contribution calculators assess how much each participant
    contributed to the overall model performance in federated learning.
    """

    @abstractmethod
    def get_scores(
        self,
        *,
        participant_mapping: ParticipantMapping,
        store: TrainerSetAggregateStore,
        coalition_to_score_fn: Callable[[TrainerSetAggregate], float]
        | None = default_coalition_to_score,
    ) -> dict[ChecksumAddress, PlayerScore]:
        """Calculates contribution scores for all participants.

        Computes the contribution score for each participant.
        The specific calculation method depends on the implementing class.

        Args:
            participant_mapping: Maps participants to their coalition memberships.
            store: Contains metrics and losses for all evaluated coalitions.
            coalition_to_score_fn: Function to convert a coalition's aggregate
            to a numerical score. If None, uses default_coalition_to_score.

        Returns:
            A dictionary mapping each participant's address to their PlayerScore.
        """
        pass
