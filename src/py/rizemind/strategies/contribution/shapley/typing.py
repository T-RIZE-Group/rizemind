from typing import Protocol

from eth_typing import ChecksumAddress


class SupportsShapleyValueStrategy(Protocol):
    """Protocol defining the interface for swarm management in Shapley value strategies.

    This protocol specifies the required methods for managing trainer compensation
    and round progression in a federated learning swarm using Shapley value-based
    contribution calculation.
    """

    def distribute(
        self, round_id: int, trainer_scores: list[tuple[ChecksumAddress, float]]
    ) -> str:
        """Distribute rewards to trainers based on their contribution scores.

        Args:
            round_id: The identifier of the current round.
            trainer_scores: List of tuples containing trainer addresses and their
            corresponding contribution scores.

        Returns:
            Transaction hash or confirmation string of the distribution operation.
        """
        ...

    def next_round(
        self,
        round_id: int,
        n_trainers: int,
        model_score: float,
        total_contributions: float,
    ) -> str:
        """Advance to the next training round and record round statistics.

        Args:
            round_id: The identifier of the current round.
            n_trainers: Number of trainers participating in this round.
            model_score: Performance score of the selected model for next round.
            total_contributions: Sum of all trainer contribution scores.

        Returns:
            Transaction hash or confirmation string of the next round operation.
        """
        ...
