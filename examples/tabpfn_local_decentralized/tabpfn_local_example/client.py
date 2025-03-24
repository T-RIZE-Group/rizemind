from typing import cast

from eth_account import Account
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.typing import Scalar
from rizemind.authentication.config import AccountConfig
from rizemind.authentication.eth_account_client import SigningClient
from rizemind.configuration.toml_config import TomlConfig
from rizemind.contracts.compensation.shapley.decentralized.shapley_value_client import (
    DecentralShapleyValueClient,
)
from rizemind.web3.config import Web3Config
from sklearn.model_selection import train_test_split

from .task import get_weights, load_data, load_model, set_weights, test


class FlowerClient(NumPyClient):
    def __init__(self, data, sample_X, sample_y) -> None:
        self.model = load_model(sample_X, sample_y)
        self.X, self.y = data

    def get_parameters(self, config: dict[str, bool | bytes | float | int | str]):
        return get_weights(self.model.model_)

    def fit(self, parameters, config: dict[str, Scalar]):
        set_weights(self.model.model_, parameters)
        self.model.fit(self.X, self.y)
        return get_weights(self.model.model_), len(self.X), {}

    def evaluate(
        self, parameters, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        set_weights(self.model.model_, parameters)
        _, X_test, _, y_test = train_test_split(self.X, self.y, test_size=0.2)
        r2, rmse, mae = test(self.model, X_test, y_test)
        return (
            rmse,
            len(X_test),
            {
                "root_mean_squared_error": rmse,
                "mean_absolute_error": mae,
                "r2_score": r2,
            },
        )


def client_fn(context: Context):
    sample_data_path = cast(str, context.node_config["sample_data_path"])
    label_name = cast(str, context.node_config["label_name"])
    sample_X, sample_y = load_data(sample_data_path, label_name)

    dataset_path = cast(str, context.node_config["dataset_path"])
    data = load_data(dataset_path, label_name)

    config = TomlConfig("./pyproject.toml")
    account_config = AccountConfig(**config.get("tool.eth.account"))
    partition_id = int(context.node_config["partition-id"])
    account = account_config.get_account(partition_id + 1)
    web3_config = Web3Config(**config.get("tool.web3"))

    base_client = FlowerClient(data, sample_X, sample_y)
    sc = DecentralShapleyValueClient(base_client).to_client()
    c = SigningClient(sc, account, web3_config.get_web3())
    return c


Account.enable_unaudited_hdwallet_features()
app = ClientApp(client_fn=client_fn)
