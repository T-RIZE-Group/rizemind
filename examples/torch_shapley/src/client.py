import torch
from eth_account import Account
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from rizemind.authentication import authentication_mod
from rizemind.authentication.config import ACCOUNT_CONFIG_STATE_KEY, AccountConfig
from rizemind.authentication.notary.model.mod import model_notary_mod
from rizemind.configuration.toml_config import TomlConfig
from rizemind.strategies.contribution.shapley.decentralized.shapley_value_client import (
    DecentralShapleyValueClient,
)
from rizemind.web3.config import WEB3_CONFIG_STATE_KEY, Web3Config

from .task import Net, get_weights, load_data, set_weights, test, train


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate):
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
        )
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = int(context.run_config["batch-size"])
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    config = TomlConfig("./pyproject.toml")
    account_config = AccountConfig(
        **config.get("tool.eth.account") | {"default_account_index": partition_id + 1}
    )
    context.state.config_records[ACCOUNT_CONFIG_STATE_KEY] = (
        account_config.to_config_record()
    )
    web3_config = Web3Config(**config.get("tool.web3"))
    context.state.config_records[WEB3_CONFIG_STATE_KEY] = web3_config.to_config_record()
    # Return Client instance
    flwr_client = FlowerClient(trainloader, valloader, local_epochs, learning_rate)
    shapley_client = DecentralShapleyValueClient(client=flwr_client)

    return shapley_client.to_client()


Account.enable_unaudited_hdwallet_features()
# Flower ClientApp
app = ClientApp(client_fn, mods=[authentication_mod, model_notary_mod])
