import logging
from pathlib import Path
from typing import cast

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from rizemind.logging.metric_storage import MetricStorage
from rizemind.logging.metric_storage_strategy import MetricStorageStrategy

from .task import Net, get_weights

# Disable logging propagation to prevent opacus logging
flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.INFO)
flwr_logger.propagate = False


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(cast(list[float], accuracies)) / sum(examples)}


def average_epsilons(metrics: list[tuple[int, Metrics]]) -> Metrics:
    accuracies = [m["epsilon"] for _, m in metrics]
    return {"average_epsilon": sum(cast(list[float], accuracies)) / len(accuracies)}


def server_fn(context: Context) -> ServerAppComponents:
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = FedAvg(
        fraction_fit=float(context.run_config["fraction-fit"]),
        fraction_evaluate=float(context.run_config["fraction-evaluate"]),
        min_available_clients=int(context.run_config["min-available-clients"]),
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=average_epsilons,
    )
    config = ServerConfig(num_rounds=cast(int, context.run_config["num-server-rounds"]))
    metrics_storage = MetricStorage(
        Path(str(context.run_config["metrics-storage-path"])), "torch-diff-privacy"
    )
    metrics_storage.write_config(context.run_config)
    metrics_strategy = MetricStorageStrategy(strategy, metrics_storage)
    return ServerAppComponents(config=config, strategy=metrics_strategy)


app = ServerApp(server_fn=server_fn)
