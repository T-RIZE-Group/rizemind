import logging
import statistics
from typing import List, Tuple, cast

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


def aggregate_coalitions(coalitions: list[Coalition]) -> dict[str, Scalar]:
    accuracies = [
        float(coalition.get_metric("accuracy", 0)) for coalition in coalitions
    ]
    return {"median_coalition_accuracy": statistics.median(accuracies)}


def server_fn(context: Context) -> ServerAppComponents:
    config = TomlConfig("./pyproject.toml")
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

    server_config = ServerConfig(
        num_rounds=int(context.run_config["num-server-rounds"])
    )
    return ServerAppComponents(config=server_config, strategy=authStrategy)


app = ServerApp(server_fn=server_fn)
