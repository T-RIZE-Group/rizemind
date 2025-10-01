"""Centralized Shapley value strategy implementation.

This module implements a centralized version of the Shapley value strategy where
coalition evaluation is performed on the server side. The server evaluates each
coalition's model parameters directly instead of distributing evaluation tasks to clients.

This approach is possible when the server has access to a validation dataset
"""

from flwr.common import Code, Status
from flwr.common.typing import EvaluateIns, EvaluateRes, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from rizemind.strategies.contribution.shapley.shapley_value_strategy import (
    ShapleyValueStrategy,
)
from rizemind.strategies.contribution.shapley.typing import SupportsShapleyValueStrategy


class CentralShapleyValueStrategy(ShapleyValueStrategy):
    """Centralized Shapley value strategy with server-side evaluation."""

    def __init__(
        self,
        strategy: Strategy,
        model: SupportsShapleyValueStrategy,
        **kwargs,
    ) -> None:
        """Initialize the centralized Shapley value strategy.

        Args:
            strategy: The base federated learning strategy.
            model: The swarm manager for reward distribution.
            **kwargs: Additional arguments passed to ShapleyValueStrategy.
        """
        ShapleyValueStrategy.__init__(self, strategy, model, **kwargs)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure client evaluation (not used in centralized mode).

        Returns an empty list since evaluation is performed centrally on the server.

        Args:
            server_round: The current server round number.
            parameters: Current model parameters.
            client_manager: Manager handling available clients.

        Returns:
            Empty list as no client evaluation is needed.
        """
        return []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, bool | bytes | float | int | str]]:
        """Aggregate client evaluation results (not used in centralized mode).

        Returns None and empty metrics since evaluation is handled by the evaluate method.

        Args:
            server_round: The current server round number.
            results: List of client evaluation results (unused).
            failures: List of failed evaluations (unused).

        Returns:
            Tuple of (None, empty dict) as no client evaluation is performed.
        """
        return None, {}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        """Evaluate all coalitions on the server and close the round.

        Evaluates each coalition's parameters using the underlying strategy's
        evaluation method and finalizes the round by computing contributions
        and distributing rewards.

        Args:
            server_round: The current server round number.
            parameters: Current model parameters (unused, each coalitions parameter
            is available in its own object).

        Returns:
            Tuple containing the best coalition loss and aggregated metrics.

        Raises:
            ValueError: If any coalition evaluation returns None.
        """
        coalitions = self.get_coalitions()
        for coalition in coalitions:
            evaluation = self.strategy.evaluate(server_round, coalition.parameters)
            if evaluation is None:
                raise ValueError(
                    "Evaluation cannot be None. Coalition members:", coalition.members
                )
            loss, metrics = evaluation
            coalition.insert_res(
                EvaluateRes(
                    loss=loss,
                    metrics=metrics,
                    status=Status(code=Code.OK, message=""),
                    num_examples=0,
                )
            )

        return self.close_round(server_round)
