from logging import INFO
import random
from typing import cast
import uuid
from eth_typing import Address
from flwr.server.strategy import Strategy
from rizemind.contracts.compensation.shapely_value_strategy import (
    CoalitionScore,
    ShapelyValueStrategy,
)
from rizemind.contracts.models.model_registry_v1 import ModelRegistryV1
from flwr.common.typing import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from flwr.server.strategy.aggregate import aggregate_inplace as flwr_aggregate_inplace
from flwr.common.parameter import ndarrays_to_parameters as flwr_ndarrays_to_parameters

type ID = str

class DecentralShapelyValueStrategy(ShapelyValueStrategy):
    last_round_parameters: Parameters
    evaluation_results: list[EvaluateRes]
    id_to_coalitions: list[tuple[ID, list[tuple[ClientProxy, FitRes]]]]
    id_to_addresses: dict[ID, list[Address]]
    id_to_parameters: dict[ID, Parameters]

    def __init__(
        self,
        strategy: Strategy,
        model: ModelRegistryV1,
        initial_parameters: Parameters,
    ) -> None:
        ShapelyValueStrategy.__init__(self, strategy, model)
        self.last_round_parameters = initial_parameters
        self.id_to_parameters: dict[ID, Parameters] = dict()

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        parameters = self.last_round_parameters
        return self.strategy.configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        """
        aggregate_fit is responsible to create the following mapping:
        self.coalitions: A mapping of a unique ID to a coalition
        self.addresses: A mapping of the same unique ID to the list of addresses in that coalition
        The parameters that are aggregated in this stage are not important
        because the best aggregation is determined based on the result of federated evaluation
        and the top model is selected at that stage
        """
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

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """
        configure_evaluate creates EvaluateIns for participating clients
        to evaluate a designated list of models from coalitions.
        Returns a configuration with the following shape:
        list of (client, evaluate_instructions)
        such that:
                    evaluate_instructions is (parameters, config)
                    such that:
                                config = {
                                    id: str
                                }
                    and parameters is the aggregated model parameters.
        """
        num_clients = client_manager.num_available()
        clients = client_manager.sample(
            num_clients=num_clients, min_num_clients=num_clients
        )
        configurations: list[tuple[ClientProxy, EvaluateIns]] = []
        i = 0
        for id, coalition in self.id_to_coalitions:
            aggregated_parameters = self.aggregate_parameteres(coalition, parameters)
            config = {"id": id}
            self.id_to_parameters[id] = aggregated_parameters
            evaluate_ins = EvaluateIns(aggregated_parameters, config)  # type: ignore
            configurations.append((clients[i % num_clients], evaluate_ins))
        # Return client/config pairs
        return configurations

    def aggregate_parameteres(
        self, coalition: list[tuple[ClientProxy, FitRes]], parameters: Parameters
    ) -> Parameters:
        # print(len(coalition))
        if len(coalition) == 0:
            return parameters
        return flwr_ndarrays_to_parameters(flwr_aggregate_inplace(coalition))

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, bool | bytes | float | int | str]]:
        """
        aggregate_evaluate is responsible for:
        1. distribute shapely value results
        2. assign the best model's parameters to last_round_parameters
        3. log the best accuracy
        The evaluate result has a metrics such that:
        metrics = {
            evaluated_json = {
                id_1 = {
                    accuracy: float,
                    parameters: Parameters
                },
                id_2 = {
                    accuracy: float,
                    parameters: Parameters
                }
            }
        }
        """
        coalition_and_scores: list[CoalitionScore] = []
        # Find best model accuracy
        top_accuracy = -1.0
        top_loss = 0.0
        print("number of recieved results", len(results))
        print("number of recieved failures", len(failures))
        with open("/home/mikaeil/log.txt", "w") as f:
            print(failures, file=f)
        for result in results:
            evaluated_result = result[1]
            id = cast(str, evaluated_result.metrics["id"])
            accuracy = cast(float, evaluated_result.metrics["accuracy"])
            address_list = self.id_to_addresses[id]
            coalition_and_scores.append((address_list, accuracy))
            if top_accuracy < accuracy:
                top_accuracy = accuracy
                self.last_round_parameters = self.id_to_parameters[id]
                top_loss = evaluated_result.loss

        log(
            level=INFO,
            msg=f"The best accuracy for round {server_round - 1} is {top_accuracy}",
        )

        # Distribute reward
        coalition_and_scores.sort(key=lambda v: len(v[0]))
        print("This is the whole coalition and score", coalition_and_scores)
        for cs in coalition_and_scores:
            print("Printing Coalition and Score", cs)
        trainers, contributions = self.distribute_reward(coalition_and_scores)
        self.model.distribute(trainers, contributions)

        return top_loss, {}

    def evaluate(self, server_round: int, parameters: Parameters):
        return self.strategy.evaluate(server_round, parameters)

    # TODO: It might be better to just remove evaluate coalition from the base class
    def evaluate_coalition(
        self, server_round: int, results: list[tuple[ClientProxy, FitRes]]
    ) -> float:
        return 0.0
