"""signedupdates: A Flower / TensorFlow app."""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from .task import load_data, load_model
from eth_account import Account
from mnemonic import Mnemonic
from rize_dml.authentication.signature import sign_tf_model
import web3
import tomli

# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(
        self,
        learning_rate,
        data,
        epochs,
        batch_size,
        verbose,
        account: Account
    ):
        self.model = load_model(learning_rate)
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.account = account

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
        signature = sign_tf_model(
            self.account,
            self.model,
            1,
            "0x000000",
            "test_model",
            0    
        )
        print(signature)
        return self.model.get_weights(), len(self.x_train), {
            "r": signature.r,
            "s": signature.s,
            "v": signature.v
        }

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)

    # Read run_config to fetch hyperparameters relevant to this run
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")
    learning_rate = context.run_config["learning-rate"]

    mnemo = Mnemonic("english")
    with open("./pyproject.toml", "rb") as f:
        toml_dict = tomli.load(f)
        web3_config = toml_dict.get("tool", {}).get("web3", {})

    mnemonic_phrase = web3_config.get("mnemonic")
    if not mnemo.check(mnemonic_phrase):
        raise ValueError("Invalid mnemonic phrase")

    hd_path = f"m/44'/60'/{partition_id}'/0/0"
    account = Account.from_mnemonic(mnemonic_phrase, account_path=hd_path)
    print(f"{partition_id}: {account.address}")

    # Return Client instance
    return FlowerClient(learning_rate, data, epochs, batch_size, verbose, account).to_client()

Account.enable_unaudited_hdwallet_features()
# Flower ClientApp
app = ClientApp(client_fn=client_fn)