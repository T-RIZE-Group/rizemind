"""signedupdates: A Flower / TensorFlow app."""

from logging import INFO

from dotenv import load_dotenv
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from rizemind.authentication.config import AccountConfig
from rizemind.authentication.eth_account_strategy import EthAccountStrategy
from rizemind.compensation.simple_compensation_strategy import (
    SimpleCompensationStrategy,
)
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


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    log(INFO, "Creating base model.")
    log(INFO, "Initializing random weights.")
    parameters = ndarrays_to_parameters(load_model().get_weights())

    log(INFO, "Creating base strategy: FedAvg")
    # Define the strategy
    strategy = FedAvg(
        fraction_fit=float(context.run_config["fraction-fit"]),
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    # Read from config
    num_rounds = int(context.run_config["num-server-rounds"])
    #######
    # Modifications to filter whitelisted trainers
    #######
    # load .env variables into os.environ
    load_dotenv()
    # load config and parses env variables
    config = TomlConfig("./pyproject.toml")
    # loads the account config
    auth_config = AccountConfig(**config.get("tool.eth.account"))
    # loads the gateway config
    web3_config = Web3Config(**config.get("tool.web3"))
    # gets web3 instance
    w3 = web3_config.get_web3()
    # derives the account 0 which will be the aggregator
    account = auth_config.get_account(0)
    members = []
    # derives the trainers account addresses.
    for i in range(1, 11):
        trainer = auth_config.get_account(i)
        members.append(trainer.address)

    swarm_config = SwarmConfig(**config.get("tool.web3.swarm"))
    swarm = swarm_config.get_or_deploy(deployer=account, trainers=members, w3=w3)
    config = ServerConfig(num_rounds=int(num_rounds))
    log(INFO, "Server configured.")
    authStrategy = EthAccountStrategy(
        SimpleCompensationStrategy(strategy, swarm), swarm
    )
    log(INFO, "Compensation strategy configured.")
    return ServerAppComponents(strategy=authStrategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
