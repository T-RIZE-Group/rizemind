import statistics
from pathlib import Path

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
from rizemind.strategies.contribution.shapley.shapley_value_strategy import (
    TrainerSetAggregate,
)
from rizemind.swarm.config import SwarmConfig
from rizemind.web3.config import Web3Config

from .task import Net, get_weights


# Define metric aggregation function
def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def aggregate_coalitions(coalitions: list[TrainerSetAggregate]) -> dict[str, Scalar]:
    accuracies = [
        float(coalition.get_metric("accuracy", default=0, aggregator=max))
        for coalition in coalitions
    ]
    return {"median_coalition_accuracy": statistics.median(accuracies)}


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
    server_config = ServerConfig(num_rounds=num_rounds)

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
    metrics_storage = LocalDiskMetricStorage(
        Path(str(context.run_config["metrics-storage-path"])),
        "torch-shapley",
    )
    metrics_storage.write_config(context.run_config)
    metrics_storage.write_config(toml_config.data)
    metrics_strategy = MetricStorageStrategy(authStrategy, metrics_storage)

    return ServerAppComponents(strategy=metrics_strategy, config=server_config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
