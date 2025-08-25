from collections.abc import Callable
from logging import DEBUG, INFO, WARNING
from math import factorial
from typing import Protocol

from bidict import bidict
from eth_typing import ChecksumAddress
from flwr.common import FitRes
from flwr.common.logger import log
from flwr.common.typing import FitIns, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from rizemind.authentication.authenticated_client_properties import (
    AuthenticatedClientProperties,
)
from rizemind.strategies.contribution.shapley.sampling.full import FullShapley
from rizemind.strategies.contribution.shapley.sampling.sampling_strat import (
    ShapleySamplingStrategy,
)
from rizemind.strategies.contribution.shapley.trainer_set import TrainerSetAggregate

type CoalitionScore = tuple[list[ChecksumAddress], float]
type PlayerScore = tuple[ChecksumAddress, float]


class SupportsShapleyValueStrategy(Protocol):
    def distribute(
        self, trainer_scores: list[tuple[ChecksumAddress, float]]
    ) -> str: ...
    def next_round(
        self,
        round_id: int,
        n_trainers: int,
        model_score: float,
        total_contributions: float,
    ) -> str: ...


class ShapleyValueStrategy(Strategy):
    strategy: Strategy
    swarm: SupportsShapleyValueStrategy
    coalition_to_score_fn: Callable[[TrainerSetAggregate], float] | None = None
    last_round_parameters: Parameters | None
    aggregate_coalition_metrics: (
        Callable[[list[TrainerSetAggregate]], dict[str, Scalar]] | None
    ) = None
    shapley_sampling_strat: ShapleySamplingStrategy
    set_aggregates: dict[str, TrainerSetAggregate]

    def __init__(
        self,
        strategy: Strategy,
        swarm: SupportsShapleyValueStrategy,
        coalition_to_score_fn: Callable[[TrainerSetAggregate], float] | None = None,
        aggregate_coalition_metrics_fn: Callable[
            [list[TrainerSetAggregate]], dict[str, Scalar]
        ]
        | None = None,
        shapley_sampling_strat: ShapleySamplingStrategy = FullShapley(),
    ) -> None:
        log(DEBUG, "ShapleyValueStrategy: initializing")
        self.strategy = strategy
        self.swarm = swarm
        self.coalition_to_score_fn = coalition_to_score_fn
        self.set_aggregates = {}
        self.aggregate_coalition_metrics = aggregate_coalition_metrics_fn
        self.shapley_sampling_strat = shapley_sampling_strat

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        """
        Delegate the initialization of model parameters to the underlying strategy.

        :param client_manager: Manager handling available clients.
        :type client_manager: ClientManager
        :return: The initialized model parameters, or None if not applicable.
        :rtype: Parameters | None
        """
        self.last_round_parameters = self.strategy.initialize_parameters(client_manager)
        return self.last_round_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        log(DEBUG, "configure_fit: creating fit instructions for clients")
        log(
            DEBUG,
            "configure_fit: selecting the base coalition for next round",
        )
        coalition = self.select_aggregate()
        parameters = parameters if coalition is None else coalition.parameters
        log(
            DEBUG,
            "configure_fit: setting the previous rounds best parameter from the selected coalition",
        )
        self.last_round_parameters = parameters
        return self.strategy.configure_fit(server_round, parameters, client_manager)

    def select_aggregate(self) -> TrainerSetAggregate | None:
        coalitions = self.get_coalitions()
        if len(coalitions) == 0:
            log(DEBUG, "select_coalition: no coalition was found")
            return None
        # Find the coalition with the highest number of members
        log(DEBUG, "select_coalition: get coalition with the highest number of members")
        return max(coalitions, key=lambda coalition: len(coalition.members))

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        """
        Aggregate client training (fit) results and form coalitions.

        This method performs the following steps:
          1. Creates coalitions from client fit results.
          5. Delegates further parameter aggregation to the underlying strategy.

        :param server_round: The current server round number.
        :type server_round: int
        :param results: List of tuples containing client proxies and their fit results.
        :type results: list[tuple[ClientProxy, FitRes]]
        :param failures: List of any failed client results.
        :type failures: list[tuple[ClientProxy, FitRes] | BaseException]
        :return: A tuple containing the aggregated parameters (or None) and a dictionary of metrics.
        :rtype: tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]
        """
        if len(failures) > 0:
            log(
                level=WARNING,
                msg=f"aggregate_fit: there have been {len(failures)} failures in round {server_round}",
            )
        self.create_coalitions(server_round, results)

        return self.strategy.aggregate_fit(server_round, results, failures)

    def create_coalitions(
        self, server_round: int, results: list[tuple[ClientProxy, FitRes]]
    ) -> list[TrainerSetAggregate]:
        log(DEBUG, "create_coalitions: initializing")
        trainer_sets = self.shapley_sampling_strat.sample_trainer_sets(
            server_round=server_round, results=results
        )

        for trainer_set in trainer_sets:
            set_results: list[tuple[ClientProxy, FitRes]] = []
            for client, result in results:
                auth = AuthenticatedClientProperties.from_client(client)
                if auth.trainer_address in trainer_set.members:
                    set_results.append((client, result))

            if trainer_set.size() == 0:
                parameters, config = self.last_round_parameters, {}
            else:
                parameters, config = self.strategy.aggregate_fit(
                    server_round, set_results, []
                )

            if parameters is None:
                raise ValueError(f"Aggregate is none for ID {trainer_set.id}")

            self.set_aggregates[trainer_set.id] = TrainerSetAggregate(
                trainer_set.id, trainer_set.members, parameters, config
            )

        return self.get_coalitions()

    def get_coalitions(self) -> list[TrainerSetAggregate]:
        return list(self.set_aggregates.values())

    def get_coalition(self, id: str) -> TrainerSetAggregate:
        if id in self.set_aggregates:
            return self.set_aggregates[id]
        raise Exception(f"Coalition {id} not found")

    def compute_contributions(
        self, coalitions: list[TrainerSetAggregate] | None
    ) -> list[PlayerScore]:
        # Create a bijective mapping between addresses and a bit_based representation
        # First the coalition_and_scores is sorted based on the length of list of addresses
        # Then given that the largest list has all addresses, it will assign it to
        # list_of_addresses
        log(DEBUG, "compute_contributions: initializing")
        if coalitions is None:
            coalitions = self.get_coalitions()

        if len(coalitions) == 0:
            log(DEBUG, "compute_contributions: no coalition was found, returning empty")
            return []

        coalitions.sort(key=lambda coalition: len(coalition.members))
        list_of_addresses = coalitions[-1].members
        address_map: bidict[ChecksumAddress, int] = bidict()
        bit = 0b1
        for address in list_of_addresses:
            address_map[address] = bit
            bit = bit << 1

        # Create coalition_set
        coalition_set: dict[int, float] = dict()
        for coalition in coalitions:
            addresses, score = coalition.members, self.get_coalition_score(coalition)
            bit_value = sum([address_map[address] for address in addresses])
            coalition_set[bit_value] = score

        num_players = len(list_of_addresses)

        player_scores: dict[ChecksumAddress, float] = dict()
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

        log(
            INFO,
            "compute_contributions: calculated player contributions.",
            extra={"player_scores": player_scores},
        )
        return list(player_scores.items())

    def get_coalition_score(self, coalition: TrainerSetAggregate) -> float:
        score = None
        if self.coalition_to_score_fn is None:
            score = coalition.loss
        else:
            score = self.coalition_to_score_fn(coalition)
        if score is None:
            raise Exception(f"Coalition {coalition.id} not evaluated")
        return score

    def normalize_contribution_scores(
        self, trainers_and_contributions: list[PlayerScore]
    ) -> list[PlayerScore]:
        return [
            (trainer, max(contribution, 0))
            for trainer, contribution in trainers_and_contributions
        ]

    def close_round(self, round_id: int) -> tuple[float, dict[str, Scalar]]:
        coalitions = self.get_coalitions()
        player_scores = self.compute_contributions(coalitions)
        player_scores = self.normalize_contribution_scores(player_scores)
        for address, score in player_scores:
            if score == 0:
                log(
                    WARNING,
                    f"aggregate_evaluate: free rider detected! Trainer address: {address}, Score: {score}",
                )
        self.swarm.distribute(player_scores)

        loss, metrics = self.evaluate_coalitions()
        next_model = self.select_aggregate()
        score = 0 if next_model is None else self.get_coalition_score(next_model)
        self.swarm.next_round(
            round_id,
            len(player_scores),
            score,
            sum(score[1] for score in player_scores),
        )

        return loss, metrics

    def evaluate_coalitions(self) -> tuple[float, dict[str, Scalar]]:
        log(
            DEBUG,
            "evaluate_coalitions: evaluating coalitions by calculating their loss and optional metrics",
        )
        coalitions = self.get_coalitions()
        if len(coalitions) == 0:
            log(
                DEBUG,
                "evaluate_coalitions: no coalition found, returning inf as the loss value",
            )
            return float("inf"), {}

        coalition_losses = [coalition.loss or float("inf") for coalition in coalitions]
        metrics = (
            {}
            if self.aggregate_coalition_metrics is None
            else self.aggregate_coalition_metrics(coalitions)
        )

        return min(coalition_losses), metrics
