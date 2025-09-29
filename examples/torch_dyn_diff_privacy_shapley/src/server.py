import logging
import statistics
from pathlib import Path
from typing import cast

from flwr.common import Context, Metrics, Scalar, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from rizemind.authentication.config import AccountConfig
from rizemind.authentication.eth_account_strategy import EthAccountStrategy
from rizemind.configuration.toml_config import TomlConfig
from rizemind.logging.local_disk_metric_storage import LocalDiskMetricStorage
from rizemind.logging.metric_storage_strategy import MetricStorageStrategy
from rizemind.strategies.contribution.shapley.decentralized.shapley_value_strategy import (
    DecentralShapleyValueStrategy,
)
from rizemind.strategies.contribution.shapley.shapley_value_strategy import Coalition
from rizemind.swarm.config import SwarmConfig
from rizemind.web3 import Web3Config

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


def aggregate_coalitions(coalitions: list[Coalition]) -> dict[str, Scalar]:
    accuracies = [
        float(coalition.get_metric("accuracy", 0)) for coalition in coalitions
    ]
    return {"median_coalition_accuracy": statistics.median(accuracies)}


def server_fn(context: Context) -> ServerAppComponents:
    toml_config = TomlConfig("./pyproject.toml")
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

    toml_config = TomlConfig("./pyproject.toml")
    auth_config = AccountConfig(**toml_config.get("tool.eth.account"))
    web3_config = Web3Config(**toml_config.get("tool.web3"))

    num_supernodes = int(context.run_config["num-supernodes"])
    w3 = web3_config.get_web3()
    account = auth_config.get_account(0)
    members = []
    for i in range(1, num_supernodes + 1):
        trainer = auth_config.get_account(i)
        members.append(trainer.address)

    swarm_config = SwarmConfig(**toml_config.get("tool.web3.swarm"))
    swarm = swarm_config.get_or_deploy(deployer=account, trainers=members, w3=w3)
    authStrategy = EthAccountStrategy(
        DecentralShapleyValueStrategy(
            strategy,
            swarm,
            coalition_to_score_fn=lambda coalition: coalition.metrics["accuracy"],
            aggregate_coalition_metrics_fn=aggregate_coalitions,
        ),
        swarm,
        account,
    )
    server_config = ServerConfig(
        num_rounds=int(context.run_config["num-server-rounds"])
    )
    metrics_storage = LocalDiskMetricStorage(
        Path(str(context.run_config["metrics-storage-path"])),
        "torch-dyn-diff-privacy-shapley",
    )
    metrics_storage.write_config(context.run_config)
    metrics_storage.write_config(toml_config.data)
    metrics_strategy = MetricStorageStrategy(authStrategy, metrics_storage)
    return ServerAppComponents(config=server_config, strategy=metrics_strategy)


app = ServerApp(server_fn=server_fn)
