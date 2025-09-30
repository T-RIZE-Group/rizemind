from abc import ABC, abstractmethod

from flwr.common import FitRes
from flwr.server.client_proxy import ClientProxy

from rizemind.strategies.contribution.shapley.trainer_mapping import ParticipantMapping
from rizemind.strategies.contribution.shapley.trainer_set import (
    TrainerSet,
)


class SetsSamplingStrategy(ABC):
    """Abstract strategy for sampling trainer sets during federated learning rounds.

    This abstract base class defines the interface for strategies that create sets
    of trainers to evaluate during contribution assessment.
    """

    @abstractmethod
    def sample_trainer_sets(
        self, server_round: int, results: list[tuple[ClientProxy, FitRes]]
    ) -> list[TrainerSet]:
        """Samples and generates trainer sets for the given round.

        Args:
            server_round: The current server round number.
            results: A list of tuples containing client proxies and their
            corresponding fit results from the training round.

        Returns:
            A list of TrainerSet objects representing the sampled combinations
            of trainers for this round.
        """
        pass

    @abstractmethod
    def get_sets(self, round_id: int) -> list[TrainerSet]:
        """Returns all trainer sets for the specified round.

        Args:
            round_id: The round identifier to retrieve sets for.
        """
        pass

    @abstractmethod
    def get_set(self, round_id: int, id: str) -> TrainerSet:
        """Returns a specific trainer set by ID for the specified round.

        Args:
            round_id: The round identifier to retrieve the set from.
            id: The unique identifier of the trainer set to retrieve.
        """
        pass

    @abstractmethod
    def get_trainer_mapping(self, round_id: int) -> ParticipantMapping:
        """Returns the participant mapping for the specified round.

        Args:
            round_id: The round identifier to retrieve the mapping for.
        """
        pass
