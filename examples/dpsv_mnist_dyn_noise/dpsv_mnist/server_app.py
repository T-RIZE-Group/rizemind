import logging
import math
import warnings
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np
from flwr.common import Context, EvaluateRes, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg

from .task import Net, get_weights

warnings.filterwarnings("ignore")

# Configure logging to log output to a file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("fl_log.log", mode="a"), logging.StreamHandler()],
)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted average accuracy."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def calculate_shapley_values(
    metrics: List[Tuple[int, Metrics]], num_clients: int
) -> List[float]:
    """Correctly compute Shapley values for client contributions."""
    shapley_values = np.zeros(num_clients)
    for client in range(num_clients):
        contribution_sum = 0.0
        for subset_size in range(num_clients):
            for subset in combinations(range(num_clients), subset_size):
                if client in subset:
                    continue
                subset_metrics = [metrics[i] for i in subset]
                subset_accuracy = (
                    weighted_average(subset_metrics)["accuracy"] if subset else 0
                )
                new_subset_metrics = subset_metrics + [metrics[client]]
                new_subset_accuracy = weighted_average(new_subset_metrics)["accuracy"]
                marginal_contribution = new_subset_accuracy - subset_accuracy
                weight = (
                    math.factorial(subset_size)
                    * math.factorial(num_clients - subset_size - 1)
                ) / math.factorial(num_clients)
                contribution_sum += weight * marginal_contribution
        shapley_values[client] = contribution_sum
    return shapley_values.tolist()


class CustomFedAvg(FedAvg):
    """Custom FedAvg strategy with Shapley value calculation."""

    def __init__(self, num_clients, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_clients = num_clients
        self.last_shapley_values: Optional[List[float]] = None

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List[BaseException],
    ) -> Optional[float]:
        """
        results is a list of (client_proxy, eval_res).
        Each eval_res might be:
          1. EvaluateRes
          2. A 3-tuple (loss, num_examples, metrics)
        """
        unpacked_results = []

        for client_proxy, eval_res in results:
            # 1) If the client returned an EvaluateRes
            if isinstance(eval_res, EvaluateRes):
                # Extract num_examples and metrics
                unpacked_results.append((eval_res.num_examples, eval_res.metrics))

            # 2) If the client returned a 3-tuple (loss, num_examples, metrics)
            elif isinstance(eval_res, tuple) and len(eval_res) == 3:
                loss_val, num_examples, metrics = eval_res
                # We'll rely on "accuracy" existing in metrics
                unpacked_results.append((num_examples, metrics))

            else:
                logging.info(
                    f"Unexpected evaluate result type: {type(eval_res)} / {eval_res}"
                )

        # Call parent aggregator
        aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Compute Shapley if we have valid results
        if unpacked_results:
            shapley_values = calculate_shapley_values(
                unpacked_results, self.num_clients
            )
            logging.info(f"Round {server_round} Shapley Values: {shapley_values}")
            self.last_shapley_values = shapley_values

        return aggregated_metrics

    def configure_evaluate(
        self,
        server_round: int,
        parameters,
        client_manager: ClientManager,
    ):
        """
        Must return a list of (client_proxy, EvaluateIns).
        We'll add shapley_values and server_round to EvaluateIns.config.
        """
        client_instructions = super().configure_evaluate(
            server_round=server_round,
            parameters=parameters,
            client_manager=client_manager,
        )
        # Convert shapley_values to a string
        if self.last_shapley_values is not None:
            shapley_str = ",".join(str(val) for val in self.last_shapley_values)
        else:
            shapley_str = ""

        for client_proxy, evaluate_ins in client_instructions:
            evaluate_ins.config["shapley_values"] = shapley_str
            # Pass the server_round as well
            evaluate_ins.config["server_round"] = str(server_round)

        return client_instructions


def server_fn(context: Context):
    """Construct the ServerApp behavior."""
    num_rounds = context.run_config["num-server-rounds"]
    num_clients = context.run_config["num-partitions"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define the custom strategy
    strategy = CustomFedAvg(
        num_clients=num_clients,
        fraction_fit=1.0,
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
