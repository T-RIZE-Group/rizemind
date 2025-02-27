from abc import abstractmethod
import itertools
from math import factorial
from rizemind.contracts.compensation.compensation_strategy import CompensationStrategy
from flwr.server.strategy import Strategy
from rizemind.contracts.models.model_registry_v1 import ModelRegistryV1
from flwr.common import FitRes
from bidict import bidict

type CoalitionScore = tuple[list[str], float]
type PlayerScore = tuple[str, float]


class ShapelyValueStrategy(CompensationStrategy):
    strategy: Strategy
    model: ModelRegistryV1

    @abstractmethod
    def evaluate_coalitions(self):
        "Evaluates coallitions"

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

    def compute_contributions(
        self, coalition_and_scores: list[CoalitionScore]
    ) -> list[PlayerScore]:
        coalition_and_scores.sort(key=lambda v: len(v[0]))
        list_of_addresses = coalition_and_scores[-1][0]
        address_map: bidict = bidict()
        bit = 0b1
        for address in list_of_addresses:
            address_map[address] = bit
            bit = bit << 1

        # Create coalition_set
        coalition_set = dict()
        for addresses, score in coalition_and_scores:
            bit_value = sum([address_map[address] for address in addresses])
            coalition_set[bit_value] = score

        num_players = len(list_of_addresses)

        player_scores = dict()
        for player in address_map.values():
            shapley = 0
            for coalition, value in coalition_set.items():
                if coalition & player == 0:  # Player is not in the coalition
                    new_coalition = coalition | player  # Add player to the coalition
                    marginal_contribution = coalition_set[new_coalition] - value
                    s = bin(coalition).count("1")  # Size of coalition
                    shapley += (
                        factorial(s)
                        * factorial(num_players - s - 1)
                        * marginal_contribution
                    )
            shapley = shapley / factorial(num_players)
            player_scores[address_map.inverse[player]] = shapley
        return list(player_scores.items())
