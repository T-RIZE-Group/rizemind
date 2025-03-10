from logging import INFO, WARNING
from typing import cast
from flwr.server.strategy import Strategy
from rizemind.contracts.compensation.shapley.shapley_value_strategy import (
    Coalition,
    CoalitionScore,
    ShapleyValueStrategy,
)
from rizemind.contracts.models.model_registry_v1 import ModelRegistryV1
from flwr.common.typing import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
import random


class DecentralShapleyValueStrategy(ShapleyValueStrategy):
    """
    A federated learning strategy that extends the ShapleyValueStrategy to incorporate
    decentralized coalition-based evaluation and reward distribution.

    This strategy creates coalitions from client fit results, aggregates model parameters
    per coalition, and then evaluates these coalitions. Based on evaluation metrics, it selects
    the best performing coalition's parameters to be used in the next round and distributes rewards
    to clients according to their coalition contributions.
    """

    # List to store evaluation results from clients.
    evaluation_results: list[EvaluateRes]
    # Mapping of coalition IDs to coalitions, where each coalition is a list of (ClientProxy, FitRes) tuples.
    id_to_coalition: dict[str, Coalition]

    def __init__(
        self,
        strategy: Strategy,
        model: ModelRegistryV1,
        initial_parameters: Parameters,
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
        ShapleyValueStrategy.__init__(self, strategy, model, initial_parameters)

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        """
        Delegate the initialization of model parameters to the underlying strategy.

        :param client_manager: Manager handling available clients.
        :type client_manager: ClientManager
        :return: The initialized model parameters, or None if not applicable.
        :rtype: Parameters | None
        """
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """
        Configure the training instructions for clients by delegating to the underlying strategy.
        It overrides the provided parameters with the last aggregated parameters.

        :param server_round: The current server round number.
        :type server_round: int
        :param parameters: Model parameters (ignored in favor of last_round_parameters).
        :type parameters: Parameters
        :param client_manager: Manager handling available clients.
        :type client_manager: ClientManager
        :return: A list of client-fit instruction pairs.
        :rtype: list[tuple[ClientProxy, FitIns]]
        """
        parameters = self.last_round_parameters
        return self.strategy.configure_fit(server_round, parameters, client_manager)

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
          2. Randomly shuffles the coalitions to vary their order each round.
          3. Assigns a unique identifier (UUID) to each coalition.
          4. Extracts and maps client addresses from each coalition.
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

        coalitions = self.create_coalitions(server_round, results)
        # Making sure the order of the coalitions is different each time
        # to prevent giving the same client the same coalition each single time
        random.shuffle(coalitions)
        # The id_to_coalition must be reinstantiated each time
        # to free up memory from previous round's coalitions
        self.id_to_coalition = dict()
        for coalition in coalitions:
            self.id_to_coalition[coalition.id] = coalition
        return self.strategy.aggregate_fit(server_round, results, failures)

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
        num_clients = client_manager.num_available()
        clients = client_manager.sample(
            num_clients=num_clients, min_num_clients=num_clients
        )
        configurations: list[tuple[ClientProxy, EvaluateIns]] = []
        i = 0

        for id, coalition in self.id_to_coalition.items():
            config: dict[str, Scalar] = {"id": id}
            evaluate_ins = EvaluateIns(coalition.parameters, config)
            # Distribute evaluation instructions among clients using round-robin assignment.
            configurations.append((clients[i % num_clients], evaluate_ins))
            i += 1
        return configurations

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, bool | bytes | float | int | str]]:
        """
        Aggregate client evaluation results, select the best performing coalition, and distribute rewards.

        This method:
          1. Extracts the accuracy from each evaluation result and associates it with the coalition's client addresses.
          2. Determines the best performing coalition based on accuracy and updates the last_round_parameters.
          3. Logs the best accuracy from the previous round.
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
        if len(failures) > 0:
            log(
                level=WARNING,
                msg=f"There have been {len(failures)} on aggregate_evalute in round {server_round}.",
            )
        coalition_and_scores: list[CoalitionScore] = []
        top_accuracy = -1.0
        top_loss = 0.0
        # Evaluate each coalition result to determine the best performing one.
        for result in results:
            evaluated_result = result[1]
            id = cast(str, evaluated_result.metrics["id"])
            accuracy = cast(float, evaluated_result.metrics["accuracy"])
            # address_list = self.id_to_addresses[id]
            coalition: Coalition = self.id_to_coalition[id]
            coalition_and_scores.append((coalition.members, accuracy))
            if top_accuracy < accuracy:
                top_accuracy = accuracy
                self.last_round_parameters = coalition.parameters
                top_loss = evaluated_result.loss

        log(
            level=INFO,
            msg=f"The best accuracy for round {server_round} is {top_accuracy}",
        )

        # Sort coalitions by the number of trainers (clients) and distribute rewards accordingly.
        coalition_and_scores.sort(key=lambda v: len(v[0]))
        player_scores = self.compute_contributions(coalition_and_scores)
        players, contributions = self.normalize_contribution_scores(player_scores)
        self.model.distribute(players, contributions)

        return top_loss, {}

    def evaluate(self, server_round: int, parameters: Parameters):
        """
        Delegate the evaluation of the model parameters to the underlying strategy.

        :param server_round: The current server round number.
        :type server_round: int
        :param parameters: Model parameters to evaluate.
        :type parameters: Parameters
        :return: The result of the evaluation as determined by the underlying strategy.
        """
        return self.strategy.evaluate(server_round, parameters)
