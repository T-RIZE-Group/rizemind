import itertools
from math import factorial
from typing import cast
import uuid
from eth_typing import Address
from rizemind.contracts.compensation.compensation_strategy import CompensationStrategy
from flwr.server.strategy import Strategy
from rizemind.contracts.models.model_registry_v1 import ModelRegistryV1
from flwr.common import FitRes
from flwr.common.typing import Parameters, Scalar
from bidict import bidict
from flwr.server.client_proxy import ClientProxy

type CoalitionScore = tuple[list[Address], float]
type PlayerScore = tuple[Address, float]


class Coalition:
    id: str
    members: list[Address]
    parameters: Parameters
    config: dict[str, Scalar]

    def __init__(
        self,
        id: str,
        members: list[Address],
        parameters: Parameters,
        config: dict[str, Scalar],
    ) -> None:
        self.id = id
        self.members = members
        self.parameters = parameters
        self.config = config


class ShapleyValueStrategy(CompensationStrategy):
    strategy: Strategy
    model: ModelRegistryV1
    # Last round's aggregated model parameters selected based on evaluation performance.
    last_round_parameters: Parameters

    def __init__(
        self, strategy: Strategy, model: ModelRegistryV1, initial_parameters: Parameters
    ) -> None:
        super().__init__(strategy, model)
        self.last_round_parameters = initial_parameters
        self.strategy = strategy
        self.model = model

    def create_coalitions(
        self, server_round: int, results: list[tuple[ClientProxy, FitRes]]
    ) -> list[Coalition]:
        results_coalitions = [
            list(combination)
            for r in range(len(results) + 1)
            for combination in itertools.combinations(results, r)
        ]
        coalitions = []
        for results_coalition in results_coalitions:
            id = uuid.uuid4()
            id = str(id)
            members: list[Address] = []
            for _, fit_res in results_coalition:
                members.append(cast(Address, fit_res.metrics["trainer_address"]))
            if len(results_coalition) == 0:
                parameters = self.last_round_parameters
            else:
                parameters, _ = self.strategy.aggregate_fit(
                    server_round, results_coalition, []
                )

            if parameters is None:
                raise Exception("Aggregated parameters is None")
            coalitions.append(Coalition(id, members, parameters, {}))

        return coalitions

    def compute_contributions(
        self, coalition_and_scores: list[CoalitionScore]
    ) -> list[PlayerScore]:
        # Create a bijective mapping between addresses and a bit_based representation
        # First the coalition_and_scores is sorted based on the length of list of addresses
        # Then given that the largest list has all addresses, it will assign it to
        # list_of_addresses
        coalition_and_scores.sort(key=lambda v: len(v[0]))
        list_of_addresses, _ = coalition_and_scores[-1]
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

    def normalize_contribution_scores(
        self, trainers_and_contributions: list[PlayerScore]
    ):
        trainers = [trainer for trainer, _ in trainers_and_contributions]
        contributions = [
            max(int(contribution * 100), 0)
            for _, contribution in trainers_and_contributions
        ]
        return trainers, contributions
