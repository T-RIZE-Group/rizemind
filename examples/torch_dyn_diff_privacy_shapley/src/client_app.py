import logging
import warnings
from typing import cast

import datasets
import torch
from eth_account import Account
from eth_typing import ChecksumAddress
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, Scalar
from opacus import GradSampleModule, PrivacyEngine
from opacus.optimizers import DPOptimizer
from rizemind.authentication.authentication_mod import authentication_mod
from rizemind.authentication.config import ACCOUNT_CONFIG_STATE_KEY, AccountConfig
from rizemind.authentication.notary.model.config import parse_model_notary_config
from rizemind.authentication.notary.model.mod import model_notary_mod
from rizemind.compensation.shapley.decentralized.shapley_value_client import (
    DecentralShapleyValueClient,
)
from rizemind.configuration.toml_config import TomlConfig
from rizemind.contracts.erc.erc5267.erc5267 import Web3
from rizemind.swarm.swarm import Swarm
from rizemind.web3.config import WEB3_CONFIG_STATE_KEY, Web3Config
from torch.utils.data import DataLoader

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
        ) = cast(
            # if `grad_sample_mode` is equal to "ghost",
            # this method returns an additional value typed DPLossFastGradientClipping
            # therefore for proper typing, we need to cast its result
            tuple[GradSampleModule, DPOptimizer, DataLoader],
            privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=self.train_loader,
                max_grad_norm=self.max_grad_norm,
                target_delta=self.target_delta,
                target_epsilon=self.target_epsilon,
                epochs=self.epochs,
                grad_sample_mode="hooks",  # default value
            ),
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
    def __init__(
        self, flwr_client: FlowerClient, w3: Web3, client_address: ChecksumAddress
    ):
        self.flwr_client = flwr_client
        self.w3 = w3
        self.client_address = client_address

    def fit(self, parameters, config):
        notary_config = parse_model_notary_config(config)
        contract_address = notary_config.domain.verifyingContract
        round = notary_config.round_id
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
        self,
        client_address: ChecksumAddress,
        round: int,
        contract_address: ChecksumAddress,
    ):
        swarm = Swarm(address=contract_address, account=None, w3=self.w3)
        round_summary = swarm.get_last_contributed_round_summary(trainer=client_address)
        if round_summary is None or round_summary.metrics is None:
            return -1.0, -1.0, -1
        metrics = round_summary.metrics
        n_trainers, total_contributions = (
            metrics.n_trainers,
            metrics.total_contributions,
        )
        trainer_contribution = cast(
            float, swarm.get_latest_contribution(trainer=client_address)
        )
        return trainer_contribution, total_contributions, n_trainers


def client_fn(context: Context):
    partition_id = int(context.node_config["partition-id"])
    train_loader, test_loader = load_data(
        partition_id=partition_id,
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
    account_config = AccountConfig(
        **config.get("tool.eth.account") | {"default_account_index": partition_id + 1}
    )
    context.state.config_records[ACCOUNT_CONFIG_STATE_KEY] = (
        account_config.to_config_record()
    )
    web3_config = Web3Config(**config.get("tool.web3"))
    context.state.config_records[WEB3_CONFIG_STATE_KEY] = web3_config.to_config_record()
    w3 = web3_config.get_web3()
    account = account_config.get_account()
    dp_client = DynamicPrivacyClient(flwr_client, w3, account.address)
    shapley_client = DecentralShapleyValueClient(client=dp_client)

    return shapley_client.to_client()


Account.enable_unaudited_hdwallet_features()
app = ClientApp(client_fn=client_fn, mods=[authentication_mod, model_notary_mod])
