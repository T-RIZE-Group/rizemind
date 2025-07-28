"""signedupdates: A Flower / TensorFlow app."""

from eth_account import Account
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from rizemind.authentication.config import AccountConfig
from rizemind.authentication.eth_account_client import SigningClient
from rizemind.compensation.shapley.decentralized.shapley_value_client import (
    DecentralShapleyValueClient,
)
from rizemind.configuration.toml_config import TomlConfig
from rizemind.web3.config import Web3Config

from .task import load_data, load_model


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(
        self,
        learning_rate,
        data,
        epochs,
        batch_size,
        verbose,
    ):
        self.model = load_model(learning_rate)
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def get_parameters(self, config):
        """Return the parameters of the model of this client."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    data = load_data(partition_id, num_partitions)

    # Read run_config to fetch hyperparameters relevant to this run
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")
    learning_rate = context.run_config["learning-rate"]

    config = TomlConfig("./pyproject.toml")
    account_config = AccountConfig(**config.get("tool.eth.account"))
    account = account_config.get_account(partition_id + 1)
    web3_config = Web3Config(**config.get("tool.web3"))

    # Return Client instance
    return SigningClient(
        DecentralShapleyValueClient(
            FlowerClient(learning_rate, data, epochs, batch_size, verbose)
        ).to_client(),
        account,
        web3_config.get_web3(),
    )


Account.enable_unaudited_hdwallet_features()
# Flower ClientApp
app = ClientApp(client_fn=client_fn)
