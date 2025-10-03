import random
from logging import DEBUG, INFO, WARNING
from typing import cast

from flwr.common.logger import log
from flwr.common.typing import (
    EvaluateIns,
    EvaluateRes,
    Parameters,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from rizemind.strategies.contribution.shapley.shapley_value_strategy import (
    ShapleyValueStrategy,
    SupportsShapleyValueStrategy,
)
from rizemind.swarm.modules.evaluation.assigment.task_assigner import (
    SupportsTaskAssignement,
    TaskAssigner,
)


class DecentralShapleyValueStrategy(ShapleyValueStrategy):
    """Decentralized Shapley value strategy with client-side evaluation.

    This strategy extends ShapleyValueStrategy to distribute coalition evaluation
    tasks to clients rather than performing evaluation on the server. Coalitions
    are created from client training results, and their parameters are sent to
    available clients for evaluation in a round-robin fashion.
    """

    _task_assigner: TaskAssigner

    def __init__(
        self,
        strategy: Strategy,
        model: SupportsShapleyValueStrategy,
        task_assigner: SupportsTaskAssignement,
        **kwargs,
    ) -> None:
        """Initialize the decentralized Shapley value strategy.

        Args:
            strategy: The base federated learning strategy.
            model: The swarm manager for reward distribution.
            **kwargs: Additional arguments passed to ShapleyValueStrategy.
        """
        log(DEBUG, "DecentralShapleyValueStrategy: initializing")
        ShapleyValueStrategy.__init__(self, strategy, model, **kwargs)
        self._task_assigner = TaskAssigner(task_assigner, self.set_aggregates)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Create evaluation instructions for distributing coalition evaluation to clients.

        Assigns each coalition's parameters to available clients in a round-robin fashion
        for decentralized evaluation. Coalition order is randomized to prevent bias.

        Args:
            server_round: The current server round number.
            parameters: Current model parameters (unused, coalitions have their own).
            client_manager: Manager handling available clients.

        Returns:
            List of tuples containing client proxies and their evaluation instructions.
        """
        log(
            DEBUG,
            "configure_evaluate: clients' parameters received, initiating evaluation phase",
        )
        num_clients = client_manager.num_available()
        log(INFO, f"configure_evaluate: available number clients: {num_clients}")

        configurations: list[tuple[ClientProxy, EvaluateIns]] = []
        self.create_coalitions(server_round, client_manager)
        configurations = self._task_assigner.configure_evaluate(
            server_round, client_manager
        )
        log(
            DEBUG,
            "configure_evaluate: client evaluation configurations generated",
        )
        return configurations

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, bool | bytes | float | int | str]]:
        """Aggregate client evaluation results and finalize the round.

        Collects evaluation results from clients, associates them with their
        respective coalitions, and closes the round by computing contributions
        and distributing rewards.

        Args:
            server_round: The current server round number.
            results: List of tuples containing client proxies and their evaluation results.
            failures: List of any failed evaluation results.

        Returns:
            Tuple containing the best coalition loss and aggregated metrics.
        """
        log(DEBUG, "aggregate_evaluate: client evaluations received")
        if len(failures) > 0:
            log(
                level=WARNING,
                msg=f"aggregate_evaluate: there have been {len(failures)} on aggregate_evaluate in round {server_round}",
            )

        # Evaluate each coalition result to determine the best performing one.
        for result in results:
            _, evaluate_res = result
            id = str(evaluate_res.metrics["id"])
            coalition = self.get_coalition(id)
            coalition.insert_res(evaluate_res)

        return self.close_round(server_round)

    def evaluate(self, server_round: int, parameters: Parameters):
        """Server-side evaluation (disabled for decentralized mode).

        Returns None since evaluation is performed by clients in decentralized mode.

        Args:
            server_round: The current server round number.
            parameters: Model parameters (unused).

        Returns:
            None to indicate no centralized evaluation.
        """
        return None
