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

            coalitions.append(Coalition(id, members, cast(Parameters, parameters), {}))

        return coalitions

    """
    Usage in decentralized:

        self.id_to_addresses = dict()
        self.id_to_coalitions = []
        coalitions = self.create_coalitions(results)
        random.shuffle(
            coalitions
        )  # Making sure the order of designated coalitions is different each round
        for coalition in coalitions:
            id = uuid.uuid4()
            id = str(id)
            addresses: list[Address] = []
            for _, fit_res in coalition:
                addresses.append(cast(Address, fit_res.metrics["trainer_address"]))
            self.id_to_addresses[id] = addresses
            self.id_to_coalitions.append((id, coalition))

        return self.strategy.aggregate_fit(server_round, results, failures)

    Usage in centralized:
        coalitions = self.create_coalitions(results)

        # Evaluate Coalitions
        evaluated_coalitions = [
            self.evaluate_coalition(server_round, coalition) for coalition in coalitions
        ]

        # Do reward calculation
        addresses = [
            [
                cast(Address, result[1].metrics["trainer_address"])
                for result in coalition
            ]
            for coalition in coalitions
        ]
        coalition_and_scores = [
            (address, score) for address, score in zip(addresses, evaluated_coalitions)
        ]
        player_scores = self.compute_contributions(coalition_and_scores)
        trainers, contributions = self.normalize_contribution_scores(player_scores)
        self.model.distribute(trainers, contributions)

        res = self.strategy.aggregate_fit(server_round, results, failures)
        self.last_round_parameters = cast(Parameters, res[0])
        return res
    """

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
            int(contribution * 100) for _, contribution in trainers_and_contributions
        ]
        min_contrib = min(contributions)
        if min_contrib < 0:
            min_contrib *= -1
            contributions = [
                contribution + min_contrib for contribution in contributions
            ]
        return trainers, contributions
