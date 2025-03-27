"""signedupdates: A Flower / TensorFlow app."""

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.common.typing import Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from rizemind.authentication.config import AccountConfig

from rizemind.contracts.compensation.shapley.shapley_value_strategy import Coalition
from rizemind.web3.config import Web3Config
from rizemind.configuration.toml_config import TomlConfig
from rizemind.contracts.models.model_factory_v1 import (
    ModelFactoryV1Config,
    ModelFactoryV1,
)
from rizemind.contracts.compensation.shapley.decentralized.shapley_value_strategy import (
    DecentralShapleyValueStrategy,
)
from .task import load_model
from rizemind.authentication.eth_account_strategy import EthAccountStrategy
import statistics
from flwr.common.logger import log
from logging import INFO


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

    # Let's define the global model and pass it to the strategy
    # Note this is optional.
    log(INFO, "Creating base model.")
    log(INFO, "Initializing random weights.")
    parameters = ndarrays_to_parameters(load_model().get_weights())
    # Define the strategy
    log(INFO, "Creating base strategy: FedAvg")
    strategy = FedAvg(
        fraction_fit=float(context.run_config["fraction-fit"]),
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    # Read from config
    num_rounds = int(context.run_config["num-server-rounds"])
    config = TomlConfig("./pyproject.toml")
    auth_config = AccountConfig(**config.get("tool.eth.account"))
    web3_config = Web3Config(**config.get("tool.web3"))
    w3 = web3_config.get_web3()
    account = auth_config.get_account(0)
    members = []
    for i in range(1, 11):
        trainer = auth_config.get_account(i)
        members.append(trainer.address)

    model_v1_config = ModelFactoryV1Config(**config.get("tool.web3.model_v1"))
    log(INFO, "Web3 model contract loaded.")
    log(
        INFO,
        "Web3 model contract address: https://testnet-explorer.rizenet.io/address/0xB88D434B10f0bB783A826bC346396AbB19B6C6F7",
    )
    contract = ModelFactoryV1(model_v1_config).deploy(account, members, w3)
    config = ServerConfig(num_rounds=int(num_rounds))
    log(INFO, "Server configured.")
    authStrategy = EthAccountStrategy(
        DecentralShapleyValueStrategy(
            strategy,
            contract,
            coalition_to_score_fn=lambda coalition: coalition.metrics["accuracy"],
            aggregate_coalition_metrics_fn=aggregate_coalitions,
        ),
        contract,
    )
    log(INFO, "Compensation strategy configured.")
    log(INFO, "Using Shapley values for reward calculation.")
    return ServerAppComponents(strategy=authStrategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
