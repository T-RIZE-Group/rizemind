import random
from logging import DEBUG, INFO, WARNING
from typing import cast

from flwr.common.logger import log
from flwr.common.typing import (
    EvaluateIns,
    EvaluateRes,
    Parameters,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from rizemind.strategies.contribution.shapley.shapley_value_strategy import (
    ShapleyValueStrategy,
    SupportsShapleyValueStrategy,
)


class DecentralShapleyValueStrategy(ShapleyValueStrategy):
    """
    A federated learning strategy that extends the ShapleyValueStrategy to incorporate
    decentralized coalition-based evaluation and reward distribution.

    This strategy creates coalitions from client fit results, aggregates model parameters
    per coalition, and then evaluates these coalitions. Based on evaluation metrics, it selects
    the best performing coalition's parameters to be used in the next round and distributes rewards
    to clients according to their coalition contributions.
    """

    def __init__(
        self,
        strategy: Strategy,
        model: SupportsShapleyValueStrategy,
        **kwargs,
    ) -> None:
        """
        Initialize the DecentralShapleyValueStrategy.

        :param strategy: The base federated learning strategy to extend.
        :type strategy: Strategy
        :param model: The model registry containing model definitions and methods.
        :type model: ModelRegistryV1
        :param initial_parameters: The initial model parameters for the federation.
        :type initial_parameters: Parameters
        """
        log(DEBUG, "DecentralShapleyValueStrategy: initializing")
        ShapleyValueStrategy.__init__(self, strategy, model, **kwargs)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """
        Create evaluation instructions for participating clients.

        For each previously formed coalition, this method aggregates the model parameters
        using a custom aggregation method and packages them into evaluation instructions
        along with a unique coalition ID in the configuration. The evaluation instructions
        are then assigned to available clients in a round-robin fashion.

        :param server_round: The current server round number.
        :type server_round: int
        :param parameters: Model parameters used during aggregation.
        :type parameters: Parameters
        :param client_manager: Manager handling available clients.
        :type client_manager: ClientManager
        :return: A list of (client, EvaluateIns) pairs.
        :rtype: list[tuple[ClientProxy, EvaluateIns]]
        """
        log(
            DEBUG,
            "configure_evaluate: clients' parameters received, initiating evaluation phase",
        )
        num_clients = client_manager.num_available()
        log(INFO, f"configure_evaluate: available number clients: {num_clients}")
        clients = client_manager.sample(
            num_clients=num_clients, min_num_clients=num_clients
        )

        configurations: list[tuple[ClientProxy, EvaluateIns]] = []
        coalitions = self.get_coalitions()
        # Making sure the order of the coalitions is different each time
        # to prevent giving the same client the same coalition each single time
        random.shuffle(coalitions)
        for i, coalition in enumerate(coalitions):
            config: dict[str, Scalar] = {"id": coalition.id}
            evaluate_ins = EvaluateIns(coalition.parameters, config)
            # Distribute evaluation instructions among clients using round-robin assignment.
            configurations.append((clients[i % num_clients], evaluate_ins))
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
        """
        Aggregate client evaluation results and distribute rewards.

        This method:
          1. Extracts the accuracy from each evaluation result and associates it with the coalition's client addresses.
          4. Distributes rewards to the clients by invoking the model's distribution mechanism.

        Expected evaluation metrics structure:

        .. code-block:: python

           metrics = {
               "id": <coalition_id>,
               "accuracy": <float>,
               ...
           }

        :param server_round: The current server round number.
        :type server_round: int
        :param results: List of tuples containing client proxies and their evaluation results.
        :type results: list[tuple[ClientProxy, EvaluateRes]]
        :param failures: List of any failed evaluation results.
        :type failures: list[tuple[ClientProxy, EvaluateRes] | BaseException]
        :return: A tuple containing the loss value from the best performing coalition and an empty metrics dictionary.
        :rtype: tuple[float | None, dict[str, bool | bytes | float | int | str]]
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
            id = cast(str, evaluate_res.metrics["id"])
            coalition = self.get_coalition(id)
            coalition.insert_res(evaluate_res)

        return self.close_round(server_round)

    def evaluate(self, server_round: int, parameters: Parameters):
        """
        Always return None because we don't want to do centralized evaluation

        :param server_round: The current server round number.
        :type server_round: int
        :param parameters: Model parameters to evaluate.
        :type parameters: Parameters
        :return: The result of the evaluation as determined by the underlying strategy.
        """
        return None
