import statistics
from typing import cast

from flwr.common import Context, Metrics, Scalar, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from rizemind.authentication.config import AccountConfig
from rizemind.authentication.eth_account_strategy import EthAccountStrategy
from rizemind.configuration.toml_config import TomlConfig
from rizemind.contracts.compensation.shapley.decentralized.shapley_value_strategy import (
    DecentralShapleyValueStrategy,
)
from rizemind.contracts.compensation.shapley.shapley_value_strategy import Coalition
from rizemind.contracts.models.model_factory_v1 import (
    ModelFactoryV1,
    ModelFactoryV1Config,
)
from rizemind.web3.config import Web3Config

from .task import Net, get_weights


# Define metric aggregation function
# Define metric aggregation function
def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def aggregate_coalitions(coalitions: list[Coalition]) -> dict[str, Scalar]:
    accuracies = [
        float(coalition.get_metric("accuracy", 0)) for coalition in coalitions
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
        fraction_fit=cast(float, context.run_config["fraction-fit"]),
        fraction_evaluate=cast(float, context.run_config["fraction-evaluate"]),
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )
    server_config = ServerConfig(num_rounds=num_rounds)

    config = TomlConfig("./pyproject.toml")
    auth_config = AccountConfig(**config.get("tool.eth.account"))
    web3_config = Web3Config(**config.get("tool.web3"))

    num_supernodes = int(context.run_config["num-supernodes"])
    w3 = web3_config.get_web3()
    account = auth_config.get_account(0)
    members = []
    for i in range(1, num_supernodes + 1):
        trainer = auth_config.get_account(i)
        members.append(trainer.address)

    model_v1_config = ModelFactoryV1Config(**config.get("tool.web3.model_v1"))
    contract = ModelFactoryV1(model_v1_config).deploy(account, members, w3)
    authStrategy = EthAccountStrategy(
        DecentralShapleyValueStrategy(
            strategy,
            contract,
            coalition_to_score_fn=lambda coalition: coalition.metrics["accuracy"],
            aggregate_coalition_metrics_fn=aggregate_coalitions,
        ),
        contract,
    )

    return ServerAppComponents(strategy=authStrategy, config=server_config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
