import logging
import warnings
from typing import cast

import datasets
import torch
from eth_account import Account
from eth_typing import HexAddress
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, Scalar
from opacus import PrivacyEngine
from rizemind.authentication.config import AccountConfig
from rizemind.authentication.eth_account_client import SigningClient
from rizemind.configuration.toml_config import TomlConfig
from rizemind.contracts.compensation.shapley.decentralized.shapley_value_client import (
    DecentralShapleyValueClient,
)
from rizemind.contracts.models.erc5267 import Web3
from rizemind.contracts.models.model_meta import RoundMetrics
from rizemind.contracts.models.model_meta_v1 import ModelMetaV1
from rizemind.web3.config import Web3Config

from .task import Net, get_weights, load_data, set_weights, test, train

flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.INFO)
flwr_logger.propagate = False

warnings.filterwarnings("ignore", category=UserWarning)


class FlowerClient(NumPyClient):
    def __init__(
        self,
        train_loader,
        test_loader,
        max_grad_norm,
        learning_rate,
        target_delta,
        target_epsilon,
        target_epsilon_upper,
        target_epsilon_lower,
        epsilon_multiplier,
        epochs,
        contribution_upper,
        contribution_lower,
    ) -> None:
        super().__init__()
        self.model = Net()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.target_delta = target_delta
        self.target_epsilon = target_epsilon
        self.target_epsilon_upper = target_epsilon_upper
        self.target_epsilon_lower = target_epsilon_lower
        self.epsilon_multiplier = epsilon_multiplier
        self.epochs = epochs
        self.contribution_upper = contribution_upper
        self.contribution_lower = contribution_lower
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        model = self.model
        set_weights(model, parameters)

        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.learning_rate, momentum=0.9
        )

        trainer_contribution = float(config["trainer_contribution"])
        total_contributions = float(config["total_contributions"])

        if (
            trainer_contribution >= 0
        ):  # If the trainer had a contribution, it is always equal or above zero
            one_tenth_of_contribution = total_contributions / 10
            upper_contribution = self.contribution_upper * one_tenth_of_contribution
            lower_contribution = self.contribution_lower * one_tenth_of_contribution
            if trainer_contribution > upper_contribution:
                self.target_epsilon = self.target_epsilon / self.epsilon_multiplier
                self.target_epsilon = (
                    self.target_epsilon_upper
                    if self.target_epsilon > self.target_epsilon_upper
                    else self.target_epsilon
                )
            elif trainer_contribution < lower_contribution:
                self.target_epsilon = self.target_epsilon * self.epsilon_multiplier
                self.target_epsilon = (
                    self.target_epsilon_lower
                    if self.target_epsilon < self.target_epsilon_lower
                    else self.target_epsilon
                )

        privacy_engine = PrivacyEngine(secure_mode=False)
        (
            model,
            optimizer,
            self.train_loader,
        ) = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            max_grad_norm=self.max_grad_norm,
            target_delta=self.target_delta,
            target_epsilon=self.target_epsilon,
            epochs=self.epochs,
        )

        epsilon = train(
            net=model,
            train_loader=self.train_loader,
            privacy_engine=privacy_engine,
            optimizer=optimizer,
            target_delta=self.target_delta,
            device=self.device,
            epochs=self.epochs,
        )

        return (
            get_weights(model),
            len(cast(datasets.arrow_dataset.Dataset, self.train_loader.dataset)),
            {"epsilon": cast(Scalar, epsilon)},
        )

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        loss, accuracy = test(self.model, self.test_loader, self.device)
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}


class DynamicPrivacyClient(NumPyClient):
    def __init__(self, flwr_client: FlowerClient, w3: Web3, client_address: HexAddress):
        self.flwr_client = flwr_client
        self.w3 = w3
        self.client_address = client_address

    def fit(self, parameters, config):
        contract_address = str(config["contract_address"])
        round = int(config["current_round"])
        trainer_contribution, total_contributions, n_trainers = self._get_contribution(
            self.client_address, round, contract_address
        )
        config["trainer_contribution"] = cast(Scalar, trainer_contribution)
        config["total_contributions"] = cast(Scalar, total_contributions)
        config["n_trainers"] = cast(Scalar, n_trainers)
        return self.flwr_client.fit(parameters, config)

    def evaluate(self, parameters, config):
        return self.flwr_client.evaluate(parameters, config)

    def _get_contribution(
        self, client_address: HexAddress, round: int, contract_address: str
    ):
        model = ModelMetaV1.from_address(contract_address, account=None, w3=self.w3)
        round_summary = model.get_last_contributed_round_summary(trainer=client_address)
        if round_summary is None:
            return -1.0, -1.0, -1
        metrics = cast(RoundMetrics, round_summary.metrics)
        n_trainers, total_contributions = (
            metrics.n_trainers,
            metrics.total_contributions,
        )
        trainer_contribution = cast(
            float, model.get_latest_contribution(trainer=client_address)
        )
        return trainer_contribution, total_contributions, n_trainers


def client_fn(context: Context):
    train_loader, test_loader = load_data(
        partition_id=cast(int, context.node_config["partition-id"]),
        num_partitions=cast(int, context.node_config["num-partitions"]),
        batch_size=cast(int, context.run_config["batch-size"]),
        alpha=cast(float, context.run_config["alpha"]),
    )
    flwr_client = FlowerClient(
        train_loader=train_loader,
        test_loader=test_loader,
        max_grad_norm=context.run_config["max-grad-norm"],
        learning_rate=context.run_config["learning-rate"],
        target_delta=context.run_config["target-delta"],
        target_epsilon=context.run_config["target-epsilon"],
        target_epsilon_lower=context.run_config["target-epsilon-lower"],
        target_epsilon_upper=context.run_config["target-epsilon-upper"],
        epsilon_multiplier=context.run_config["epsilon-multiplier"],
        epochs=context.run_config["epochs"],
        contribution_lower=context.run_config["contribution-lower"],
        contribution_upper=context.run_config["contribution-upper"],
    )

    config = TomlConfig("./pyproject.toml")
    account_config = AccountConfig(**config.get("tool.eth.account"))
    account = account_config.get_account(
        cast(int, context.node_config["partition-id"]) + 1
    )
    web3_config = Web3Config(**config.get("tool.web3"))
    w3 = web3_config.get_web3()

    dp_client = DynamicPrivacyClient(flwr_client, w3, account.address)
    shapley_client = DecentralShapleyValueClient(client=dp_client)
    signing_client = SigningClient(
        client=shapley_client.to_client(), account=account, w3=w3
    )

    return signing_client


Account.enable_unaudited_hdwallet_features()
app = ClientApp(client_fn=client_fn)
