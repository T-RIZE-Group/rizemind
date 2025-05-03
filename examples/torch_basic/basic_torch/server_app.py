from pathlib import Path
from typing import List, Tuple, cast

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from rizemind.contracts.logging.metrics_storage import MetricsStorage
from rizemind.contracts.logging.metrics_storage_strategy import MetricsStorageStrategy

from .task import Net, get_weights


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [
        num_examples * cast(float, m["accuracy"]) for num_examples, m in metrics
    ]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    # Read from config
    num_rounds = int(context.run_config["num-server-rounds"])

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy = FedAvg(
        fraction_fit=float(context.run_config["fraction-fit"]),
        fraction_evaluate=float(context.run_config["fraction-evaluate"]),
        min_available_clients=int(context.run_config["min-available-clients"]),
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    metrics_storage = MetricsStorage(
        Path(str(context.run_config["metrics-storage-path"])), "torch-basic"
    )
    metrics_storage.write_config(context.run_config)
    metrics_strategy = MetricsStorageStrategy(strategy, metrics_storage)
    return ServerAppComponents(strategy=metrics_strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
