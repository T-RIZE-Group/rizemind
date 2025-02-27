from abc import abstractmethod
import itertools
from math import factorial
from rizemind.contracts.compensation.compensation_strategy import CompensationStrategy
from flwr.server.strategy import Strategy
from rizemind.contracts.models.model_registry_v1 import ModelRegistryV1
from flwr.common import FitRes

# TODO: fix naming
type CoalitionScore = list[tuple[int, float]]


class ShapelyValueStrategy(CompensationStrategy):
    strategy: Strategy
    model: ModelRegistryV1
    coalitions_scores: CoalitionScore

    def create_coalitions(self, results: list[FitRes]):
        trainer_addresses: list[str] = [
            str(res.metrics["trainer_address"]) for res in results
        ]
        coalitions: list[list] = [
            list(combination)
            for r in range(len(trainer_addresses) + 1)
            for combination in itertools.combinations(trainer_addresses, r)
        ]
        return coalitions

    @abstractmethod
    def evaluate_coalitions(self):
        "Evaluates coallitions"

    # TODO: Improve this function to not take any players.
    # TODO: Improve the variable namings
    def compute_contributions(self, player, cs: CoalitionScore) -> float:
        """
        Calculate the Shapley value for a single player using the correct Shapley formula.

        :param player_bit: The bit representation of the player.
        :param outcomes: A list of tuples where each tuple consists of a coalition (bitmask) and its value.
        :return: The Shapley value of the given player.
        """
        value_dict = dict(cs)
        num_players = bin(max(value_dict.keys())).count(
            "1"
        )  # Count bits in the largest coalition

        shapley = 0

        # Iterate over all possible coalitions excluding the player
        for coalition, value in cs:
            if coalition & player == 0:  # Player is not in the coalition
                new_coalition = coalition | player  # Add player to the coalition
                marginal_contribution = value_dict[new_coalition] - value
                s = bin(coalition).count("1")  # Size of coalition
                shapley += (
                    factorial(s)
                    * factorial(num_players - s - 1)
                    * marginal_contribution
                )

        return shapley / factorial(num_players)
