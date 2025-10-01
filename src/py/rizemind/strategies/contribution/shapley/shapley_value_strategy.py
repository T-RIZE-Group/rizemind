from collections.abc import Callable
from logging import DEBUG, INFO, WARNING

from eth_typing import ChecksumAddress
from flwr.common import FitRes
from flwr.common.logger import log
from flwr.common.typing import FitIns, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from rizemind.authentication import (
    AuthenticatedClientProperties,
)
from rizemind.strategies.contribution.calculators import (
    ShapleyValueCalculator,
)
from rizemind.strategies.contribution.calculators.calculator import (
    ContributionCalculator,
    PlayerScore,
)
from rizemind.strategies.contribution.sampling import AllSets
from rizemind.strategies.contribution.sampling.sets_sampling_strat import (
    SetsSamplingStrategy,
)
from rizemind.strategies.contribution.shapley.trainer_set import (
    TrainerSetAggregate,
    TrainerSetAggregateStore,
)
from rizemind.strategies.contribution.shapley.typing import SupportsShapleyValueStrategy
from rizemind.strategies.fit_res_store import InMemoryFitResStore, SupportsFitResStore


class ShapleyValueStrategy(Strategy):
    """Federated learning strategy using Shapley values for contribution calculation.

    This strategy extends the Flower Strategy to incorporate Shapley value-based
    contribution calculation. It creates coalitions of trainers, aggregates their models,
    and evaluates each coalition's performance to compute fair contribution scores using
    the Shapley value method.

    Attributes:
        strategy: The underlying federated learning strategy for aggregation.
        swarm: The swarm manager handling reward distribution and round progression.
        coalition_to_score_fn: Optional function to compute a score from a coalition aggregate.
        last_round_parameters: Parameters from the previous round.
        aggregate_coalition_metrics: Optional function to aggregate metrics across coalitions.
        sets_sampling_strat: Strategy for sampling trainer subsets/coalitions.
        set_aggregates: Store for coalitions.
        contribution_calculator: Calculator for computing Shapley value contributions.
    """

    # TODO:
    #     There is a mismatch between the loss returned by `evaluate_coalitions` and the loss of the
    #     selected model parameters for next round. This is due to the fact that for `evaluate_coalitions`
    #     returns the minimum loss among all coalitions, while the selected model is from the coalition
    #     that all trainers participated in. Therefore if this model does not have the lowest loss
    #     (which can occur often) there will be a mismatch between the selected parameter's loss vs
    #     what is displayed. This needs to be addresses in later versions by selecting the model
    #     that its loss is returned by `evaluate_coalitions`.

    strategy: Strategy
    swarm: SupportsShapleyValueStrategy
    coalition_to_score_fn: Callable[[TrainerSetAggregate], float] | None = None
    last_round_parameters: Parameters | None
    aggregate_coalition_metrics: (
        Callable[[list[TrainerSetAggregate]], dict[str, Scalar]] | None
    ) = None
    sets_sampling_strat: SetsSamplingStrategy
    set_aggregates: TrainerSetAggregateStore
    contribution_calculator: ContributionCalculator
    fit_res_store: SupportsFitResStore

    def __init__(
        self,
        strategy: Strategy,
        swarm: SupportsShapleyValueStrategy,
        coalition_to_score_fn: Callable[[TrainerSetAggregate], float] | None = None,
        aggregate_coalition_metrics_fn: Callable[
            [list[TrainerSetAggregate]], dict[str, Scalar]
        ]
        | None = None,
        shapley_sampling_strat: SetsSamplingStrategy = AllSets(),
        contribution_calculator: ContributionCalculator = ShapleyValueCalculator(),
        fit_res_store: SupportsFitResStore = InMemoryFitResStore(),
    ) -> None:
        """Initialize the Shapley value strategy.

        Args:
            strategy: Base federated learning strategy for model aggregation.
            swarm: Swarm manager for reward distribution and round management.
            coalition_to_score_fn: Optional function to extract score from coalition.
            If None, uses the coalition's loss value.
            aggregate_coalition_metrics_fn: Optional function to compute aggregate
            metrics across all coalitions.
            shapley_sampling_strat: Strategy for sampling trainer coalitions.
            Defaults to AllSets() which generates all possible subsets.
            contribution_calculator: Calculator for computing contribution scores.
            Defaults to ShapleyValueCalculator().
        """
        log(DEBUG, "ShapleyValueStrategy: initializing")
        self.strategy = strategy
        self.swarm = swarm
        self.coalition_to_score_fn = coalition_to_score_fn
        self.set_aggregates = TrainerSetAggregateStore()
        self.aggregate_coalition_metrics = aggregate_coalition_metrics_fn
        self.sets_sampling_strat = shapley_sampling_strat
        self.contribution_calculator = contribution_calculator
        self.fit_res_store = fit_res_store

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        """Delegate the initialization of model parameters to the underlying strategy.

        Args:
            client_manager: Manager handling available clients.

        Returns:
            The initialized model parameters, or None if not applicable.
        """
        self.last_round_parameters = self.strategy.initialize_parameters(client_manager)
        return self.last_round_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training for clients.

        Selects the coalition where every trainer participated from the previous round
        and uses its parameters as the starting point for the next training round.

        Args:
            server_round: The current server round number.
            parameters: Current model parameters.
            client_manager: Manager handling available clients.

        Returns:
            List of tuples containing client proxies and their fit instructions.
        """
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
        """Select the coalition aggregate to use for the next training round.

        Selects the coalition with the highest number of members as the base for
        the next round.

        Returns:
            The selected coalition aggregate, or None if no coalitions exist.
        """
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
        """Aggregate client training results and form coalitions.

        Creates coalitions from client fit results and delegates parameter aggregation
        to the underlying strategy.

        Args:
            server_round: The current server round number.
            results: List of tuples containing client proxies and their fit results.
            failures: List of any failed client results.

        Returns:
            A tuple containing the aggregated parameters (or None) and a dictionary
            of metrics.
        """
        if len(failures) > 0:
            log(
                level=WARNING,
                msg=f"aggregate_fit: there have been {len(failures)} failures in round {server_round}",
            )

        self.fit_res_store.clear()
        for [client, fit_res] in results:
            auth = AuthenticatedClientProperties.from_client(client)
            self.fit_res_store.insert(auth.trainer_address, fit_res)

        return self.strategy.aggregate_fit(server_round, results, failures)

    def create_coalitions(
        self, server_round: int, client_manager: ClientManager
    ) -> list[TrainerSetAggregate]:
        """Create coalitions from client training results.

        Samples trainer subsets using the sampling strategy, aggregates parameters for
        each subset, and stores the resulting coalition aggregates.

        Args:
            server_round: The current server round number.
            results: List of tuples containing client proxies and their fit results.

        Returns:
            List of created coalitions.

        Raises:
            ValueError: If no aggregate parameters are returned for a trainer set.
        """
        log(DEBUG, "create_coalitions: initializing")
        results = self.fit_res_store.items()
        trainer_sets = self.sets_sampling_strat.sample_trainer_sets(
            server_round=server_round, results=results
        )

        address_to_proxy: dict[ChecksumAddress, ClientProxy] = {}
        for client in client_manager.all().values():
            auth = AuthenticatedClientProperties.from_client(client)
            address_to_proxy[auth.trainer_address] = client

        for trainer_set in trainer_sets:
            set_results: list[tuple[ClientProxy, FitRes]] = []
            for client_address, result in results:
                if client_address in trainer_set.members:
                    set_results.append((address_to_proxy[client_address], result))

            if trainer_set.size() == 0:
                parameters, config = self.last_round_parameters, {}
            else:
                parameters, config = self.strategy.aggregate_fit(
                    server_round, set_results, []
                )

            if parameters is None:
                raise ValueError(
                    f"No aggregate returned for trainer set ID {trainer_set.id}"
                )

            self.set_aggregates.insert(
                TrainerSetAggregate(
                    trainer_set.id, trainer_set.members, parameters, config
                )
            )

        return self.get_coalitions()

    def get_coalitions(self) -> list[TrainerSetAggregate]:
        """Returns all coalitions."""
        return self.set_aggregates.get_sets()

    def get_coalition(self, id: str) -> TrainerSetAggregate:
        """Get a specific coalition by ID.

        Args:
            id: The identifier of the coalition.

        Returns:
            The requested coalition aggregate.

        Raises:
            Exception: If the coalition with the given ID is not found.
        """
        return self.set_aggregates.get_set(id)

    def compute_contributions(
        self, round_id: int, coalitions: list[TrainerSetAggregate] | None
    ) -> list[PlayerScore]:
        """Compute Shapley value contribution score for each trainer.

        Uses the contribution calculator to determine each trainer's contribution
        to the overall model performance based on coalition evaluations.

        Args:
            round_id: The current round identifier.
            coalitions: Optional list of coalitions to compute contributions from.
            If None, uses all available coalitions.

        Returns:
            List of player scores containing trainer addresses and their contribution values.
        """
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

        trainer_mapping = self.sets_sampling_strat.get_trainer_mapping(round_id)
        player_scores = self.contribution_calculator.get_scores(
            participant_mapping=trainer_mapping,
            store=self.set_aggregates,
            coalition_to_score_fn=self.coalition_to_score_fn,
        )

        log(
            INFO,
            "compute_contributions: calculated player contributions.",
            extra={"player_scores": player_scores},
        )
        return list(player_scores.values())

    def get_coalition_score(self, coalition: TrainerSetAggregate) -> float:
        """Get the performance score for a coalition.

        If no `coalition_to_score_fn` is provided it defaults to the loss value.
        The loss value is inversed since higher loss means lower performance.

        Args:
            coalition: The coalition aggregate to score.

        Returns:
            The performance score of the coalition.

        Raises:
            Exception: If the coalition has not been evaluated.
        """
        score = None
        if self.coalition_to_score_fn is None:
            score = 1 / coalition.get_loss()
        else:
            score = self.coalition_to_score_fn(coalition)
        if score is None:
            raise Exception(f"Coalition {coalition.id} not evaluated")
        return score

    def normalize_contribution_scores(
        self, trainers_and_contributions: list[PlayerScore]
    ) -> list[PlayerScore]:
        """Normalize contribution scores to ensure non-negative values.

        Args:
            trainers_and_contributions: List of player scores to normalize.

        Returns:
            List of player scores with negative values clamped to zero.
        """
        return [
            PlayerScore(
                trainer_address=score.trainer_address, score=max(score.score, 0)
            )
            for score in trainers_and_contributions
        ]

    def close_round(self, round_id: int) -> tuple[float, dict[str, Scalar]]:
        """Finalize the current round by computing contributions and distributing rewards.

        Computes trainer contributions, normalizes scores, distributes rewards,
        and prepares for the next round.

        Args:
            round_id: The current round identifier.

        Returns:
            A tuple containing the best coalition loss and aggregated metrics.
        """
        coalitions = self.get_coalitions()
        player_scores = self.compute_contributions(round_id, coalitions)
        player_scores = self.normalize_contribution_scores(player_scores)
        for player_score in player_scores:
            if player_score.score == 0:
                log(
                    WARNING,
                    f"aggregate_evaluate: free rider detected! Trainer address: {player_score.trainer_address}, Score: {player_score.score}",
                )
        self.swarm.distribute(
            round_id,
            [
                (player_score.trainer_address, player_score.score)
                for player_score in player_scores
            ],
        )

        return self.evaluate_coalitions()

    def evaluate_coalitions(self) -> tuple[float, dict[str, Scalar]]:
        """Evaluate all coalitions and determine the best performance.

        Calculates loss values for all coalitions and optionally aggregates metrics.

        Returns:
            A tuple containing the minimum coalition loss and aggregated metrics.
        """
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

        coalition_losses = [
            coalition.get_loss() or float("inf") for coalition in coalitions
        ]
        metrics = (
            {}
            if self.aggregate_coalition_metrics is None
            else self.aggregate_coalition_metrics(coalitions)
        )

        return min(coalition_losses), metrics
