import json
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
from flwr.common.typing import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from flwr.server.strategy.aggregate import aggregate_inplace as flwr_aggregate_inplace
from flwr.common.parameter import ndarrays_to_parameters as flwr_ndarrays_to_parameters

type ID = uuid.UUID


# TODO: If we need fractional selection of clients for evaluation
# we need to add that

class DecentralShapelyValueStrategy(ShapelyValueStrategy):
    last_round_parameters: Parameters
    evaluation_results: list[EvaluateRes]
    coalitions: list[tuple[ID, list[tuple[ClientProxy, FitRes]]]]
    addresses: dict[ID, list[Address]]

    def __init__(
        self,
        strategy: Strategy,
        model: ModelRegistryV1,
        initial_parameters: Parameters,
    ) -> None:
        ShapelyValueStrategy.__init__(self, strategy, model)
        self.last_round_parameters = initial_parameters

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
        self.addresses = dict()
        self.coalitions = []
        coalitions = self.create_coalitions(results)
        random.shuffle(
            coalitions
        )  # Making sure the order of designated coalitions is different each round
        for coalition in coalitions:
            id = uuid.uuid4()
            addresses: list[Address] = []
            for _, fit_res in coalition:
                addresses.append(cast(Address, fit_res.metrics["trainer_address"]))
            self.addresses[id] = addresses
            self.coalitions.append((id, coalition))

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
                                    evaluation_json = {
                                        id: uuid.UUID = parameters: Parameters
                                    }
                                }
        Please note that the dictionary must be dumped using json.dumps because
        flwr only accepts a Scalar (bool | bytes | float | int | str)
        """
        num_clients = client_manager.num_available()
        clients = client_manager.sample(
            num_clients=num_clients, min_num_clients=num_clients
        )

        num_coalitions = len(self.coalitions)
        i = 0
        step = num_coalitions // num_clients
        configurations: list[tuple[ClientProxy, EvaluateIns]] = []
        for client in clients:
            evaluation_json = {}
            # Make sure that the i+step in self.coalitions[i: i+step]
            # is not gonna be bigger than self.coalition size
            if i + step >= num_coalitions:
                step = num_coalitions - i

            for id, coalition in self.coalitions[i : i + step]:
                evaluation_json[id] = self.aggregate_parameteres(coalition)

            config = {}
            config["evaluation_json"] = json.dumps(evaluation_json)
            evaluate_ins = EvaluateIns(parameters, config)
            configurations.append((client, evaluate_ins))

            i += step

        # Return client/config pairs
        return configurations

    def aggregate_parameteres(
        self, coalition: list[tuple[ClientProxy, FitRes]]
    ) -> Parameters:
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
        top_accuracy = -1
        for result in results:
            evaluated_json_loaded: dict = json.loads(
                cast(str, result[1].metrics["evaluated_json"])
            )
            for id, evaluted_result in evaluated_json_loaded.items():
                address_list = self.addresses[id]
                coalition_and_scores.append((address_list, evaluted_result["accuracy"]))
                if top_accuracy < evaluted_result["accuracy"]:
                    top_accuracy = evaluted_result["accuracy"]
                    self.last_round_parameters = cast(
                        Parameters, evaluted_result["parameters"]
                    )

        log(
            level=INFO,
            msg=f"The best accuracy for round {server_round - 1} is {top_accuracy}",
        )

        # Distribute reward
        trainers, contributions = self.distribute_reward(coalition_and_scores)
        self.model.distribute(trainers, contributions)

        return super().aggregate_evaluate(server_round, results, failures)
