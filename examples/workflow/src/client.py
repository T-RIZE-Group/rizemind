import timeit
from typing import cast

import torch
from eth_account import Account
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, Scalar
from rizemind.authentication.authentication_mod import authentication_mod
from rizemind.authentication.config import AccountConfig
from rizemind.authentication.notary.model.mod import model_notary_mod
from rizemind.configuration.toml_config import TomlConfig
from rizemind.swarm.config import SwarmConfig
from rizemind.swarm.modules.contribution.register_mod import register_contribution_mod
from rizemind.swarm.modules.evaluation.ins import parse_evaluation_task_ins
from rizemind.swarm.swarm import Swarm
from rizemind.web3.config import Web3Config
from web3 import Web3

from .task import Net, get_weights, load_data, set_weights, test, train


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(
        self, swarm: Swarm, trainloader, valloader, local_epochs, learning_rate
    ):
        self.swarm = swarm
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        start = timeit.default_timer()
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
        )
        end = timeit.default_timer()
        print(f"Train time: {end - start}")
        self.swarm.register_for_round_evaluation(round_id=1)
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            cast(dict[str, Scalar], results),
        )

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        task_ins = parse_evaluation_task_ins(config)
        self.swarm.register_evaluation(
            round_id=task_ins.round_id,
            eval_id=task_ins.eval_id,
            set_id=task_ins.set_id,
            model_hash=task_ins.model_hash,
            result=Web3.to_wei(accuracy, "gwei"),
        )
        return (
            loss,
            len(self.valloader.dataset),
            cast(dict[str, Scalar], {"accuracy": accuracy, "id": task_ins.set_id}),
        )


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
    swarm_config = SwarmConfig.from_context(context)
    w3_config = Web3Config.from_context(context)
    account_config = AccountConfig.from_context(context)
    if swarm_config is None or w3_config is None or account_config is None:
        raise ValueError("SwarmConfig, Web3Config, or AccountConfig is not found")
    swarm = swarm_config.get(
        account=account_config.get_account(), w3=w3_config.get_web3()
    )
    # Return Client instance
    flwr_client = FlowerClient(
        swarm, trainloader, valloader, local_epochs, learning_rate
    )

    return flwr_client.to_client()


Account.enable_unaudited_hdwallet_features()
# Flower ClientApp
app = ClientApp(
    client_fn, mods=[authentication_mod, register_contribution_mod, model_notary_mod]
)


@app.lifespan()
def lifespan(context: Context):
    config = TomlConfig("./pyproject.toml")
    partition_id = int(context.node_config["partition-id"])
    account_config = AccountConfig(
        **config.get("tool.eth.account") | {"default_account_index": partition_id + 1}
    )
    account_config.store_in_context(context)
    web3_config = Web3Config(**config.get("tool.web3"))
    web3_config.store_in_context(context)
    swarm_config = SwarmConfig(**config.get("tool.web3.swarm"))
    swarm_config.store_in_context(context)

    yield
