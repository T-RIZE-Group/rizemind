"""signedupdates: A Flower / TensorFlow app."""

import statistics

from dotenv import load_dotenv
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.common.typing import Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from rizemind.authentication.config import AccountConfig
from rizemind.authentication.eth_account_strategy import EthAccountStrategy
from rizemind.compensation.shapley.decentralized.shapley_value_strategy import (
    DecentralShapleyValueStrategy,
)
from rizemind.compensation.shapley.shapley_value_strategy import Coalition
from rizemind.configuration.toml_config import TomlConfig
from rizemind.swarm.config import SwarmConfig
from rizemind.web3.config import Web3Config

from .task import load_model


# Define metric aggregation function
def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_metrics_aggregation_fn(metrics):
    return {}


def aggregate_coalitions(coalitions: list[Coalition]) -> dict[str, Scalar]:
    accuracies = [
        float(coalition.get_metric("accuracy", 0)) for coalition in coalitions
    ]
    return {"median_coalition_accuracy": statistics.median(accuracies)}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    parameters = ndarrays_to_parameters(load_model().get_weights())
    strategy = FedAvg(
        fraction_fit=float(context.run_config["fraction-fit"]),
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    )

    # Loading configurations
    load_dotenv()

    num_rounds = int(context.run_config["num-server-rounds"])
    server_config = ServerConfig(num_rounds=int(num_rounds))

    config = TomlConfig("./pyproject.toml")

    web3_config = Web3Config(**config.get("tool.web3"))
    w3 = web3_config.get_web3()

    auth_config = AccountConfig(**config.get("tool.eth.account"))
    account = auth_config.get_account(0)
    members = []
    for i in range(1, 11):
        trainer = auth_config.get_account(i)
        members.append(trainer.address)

    swarm_config = SwarmConfig(**config.get("tool.web3.swarm"))
    swarm = swarm_config.get_or_deploy(deployer=account, trainers=members, w3=w3)
    authStrategy = EthAccountStrategy(
        DecentralShapleyValueStrategy(
            strategy,
            swarm,
            coalition_to_score_fn=lambda coalition: coalition.metrics["accuracy"],
            aggregate_coalition_metrics_fn=aggregate_coalitions,
        ),
        swarm,
    )

    return ServerAppComponents(strategy=authStrategy, config=server_config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
