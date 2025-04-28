import logging
from typing import List, Tuple, cast

from .task import Net, get_weights

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

# Disable logging propagation to prevent opacus logging
flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.INFO)
flwr_logger.propagate = False


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(cast(list[float], accuracies)) / sum(examples)}


def average_epsilons(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [m["epsilon"] for _, m in metrics]
    return {"average_epsilon": sum(cast(list[float], accuracies)) / len(accuracies)}


def server_fn(context: Context) -> ServerAppComponents:
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=average_epsilons,
    )
    config = ServerConfig(num_rounds=cast(int, context.run_config["num-server-rounds"]))

    return ServerAppComponents(config=config, strategy=strategy)


app = ServerApp(server_fn=server_fn)
