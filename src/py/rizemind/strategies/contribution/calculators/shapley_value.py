from collections.abc import Callable
from math import factorial

from eth_typing import ChecksumAddress
from rizemind.strategies.contribution.calculators.calculator import (
    ContributionCalculator,
    PlayerScore,
    default_coalition_to_score,
)
from rizemind.strategies.contribution.shapley.trainer_mapping import ParticipantMapping
from rizemind.strategies.contribution.shapley.trainer_set import (
    TrainerSetAggregate,
    TrainerSetAggregateStore,
)


class ShapleyValueCalculator(ContributionCalculator):
    def get_scores(
        self,
        *,
        participant_mapping: ParticipantMapping,
        store: TrainerSetAggregateStore,
        coalition_to_score_fn: Callable[[TrainerSetAggregate], float]
        | None = default_coalition_to_score,
    ) -> dict[ChecksumAddress, PlayerScore]:
        num_players = participant_mapping.get_total_participants()
        player_scores: dict[ChecksumAddress, PlayerScore] = dict()
        for player in participant_mapping.get_participants():
            weighted_sum = 0.0
            weight_total = 0.0
            for set in store.get_sets():
                if not participant_mapping.in_set(
                    player, set.id
                ):  # Player is not in the coalition
                    in_set_id = participant_mapping.include_participants(
                        player, set.id
                    )  # Add player to the coalition

                    if not store.is_available(set.id) or not store.is_available(
                        in_set_id
                    ):
                        continue  # skips if either of the coalitions are unavailable

                    not_in_set_aggregate = store.get_set(set.id)
                    to_score = (
                        default_coalition_to_score
                        if coalition_to_score_fn is None
                        else coalition_to_score_fn
                    )
                    not_in_set_score = to_score(not_in_set_aggregate)
                    in_set_aggregate = store.get_set(in_set_id)
                    in_set_score = to_score(in_set_aggregate)
                    marginal_contribution = in_set_score - not_in_set_score
                    s = set.size()
                    w = factorial(s) * factorial(
                        num_players - s - 1
                    )  # Shapley weight numerator
                    weighted_sum += w * marginal_contribution
                    weight_total += w

            score = (weighted_sum / weight_total) if weight_total > 0 else 0
            # Renormalize by the total weight we actually used
            player_scores[player] = PlayerScore(trainer_address=player, score=score)
        return player_scores
