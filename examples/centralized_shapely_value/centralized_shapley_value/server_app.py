"""signedupdates: A Flower / TensorFlow app."""

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from rizemind.authentication.config import AccountConfig
from rizemind.contracts.compensation.central_shapely_value_strategy import (
    CentralShapelyValueStrategy,
)
from rizemind.web3.config import Web3Config
from rizemind.configuration.toml_config import TomlConfig
from rizemind.contracts.models.model_factory_v1 import (
    ModelFactoryV1Config,
    ModelFactoryV1,
)
from .task import evaluate_fn, load_model
from rizemind.authentication.eth_account_strategy import EthAccountStrategy


# Define metric aggregation function
def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    # Let's define the global model and pass it to the strategy
    # Note this is optional.
    parameters = ndarrays_to_parameters(load_model().get_weights())
    print(context.run_config)
    print(context.node_config)
    # Define the strategy
    strategy = FedAvg(
        fraction_fit=float(context.run_config["fraction-fit"]),
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=evaluate_fn,  # type:ignore
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
    contract = ModelFactoryV1(model_v1_config).deploy(account, members, w3)
    config = ServerConfig(num_rounds=int(num_rounds))
    authStrategy = EthAccountStrategy(
        CentralShapelyValueStrategy(strategy, contract, parameters), contract
    )
    return ServerAppComponents(strategy=authStrategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
