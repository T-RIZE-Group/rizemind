from flwr.common import Context, Metrics, ndarrays_to_parameters, EvaluateRes
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from itertools import combinations
import numpy as np
from typing import List, Tuple, Optional
import warnings
from .task import Net, get_weights
import math

warnings.filterwarnings("ignore")


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted average accuracy."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def calculate_shapley_values(
    metrics: List[Tuple[int, Metrics]], num_clients: int
) -> List[float]:
    """Compute Shapley values for client contributions."""
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
        self, server_round: int, results: List[Tuple], failures: List[BaseException]
    ) -> Optional[float]:
        """Aggregate evaluation results and calculate Shapley values."""
        unpacked_results = []

        for client_proxy, eval_res in results:
            if isinstance(eval_res, EvaluateRes):
                unpacked_results.append((eval_res.num_examples, eval_res.metrics))
            elif isinstance(eval_res, tuple) and len(eval_res) == 3:
                loss_val, num_examples, metrics = eval_res
                unpacked_results.append((num_examples, metrics))

        aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        if unpacked_results:
            shapley_values = calculate_shapley_values(
                unpacked_results, self.num_clients
            )
            print(f"Round {server_round} Shapley Values: {shapley_values}")
            self.last_shapley_values = shapley_values

        return aggregated_metrics

    def configure_evaluate(
        self, server_round: int, parameters, client_manager: ClientManager
    ):
        """Send Shapley values and round number to clients."""
        client_instructions = super().configure_evaluate(
            server_round, parameters, client_manager
        )

        shapley_str = (
            ",".join(str(val) for val in self.last_shapley_values)
            if self.last_shapley_values
            else ""
        )

        for client_proxy, evaluate_ins in client_instructions:
            evaluate_ins.config["shapley_values"] = shapley_str
            evaluate_ins.config["server_round"] = str(server_round)

        return client_instructions


def server_fn(context: Context):
    """Construct the ServerApp behavior."""
    num_rounds = context.run_config["num-server-rounds"]
    num_clients = context.run_config["num-partitions"]

    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

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
