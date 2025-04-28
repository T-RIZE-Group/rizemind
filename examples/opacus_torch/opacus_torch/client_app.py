from typing import cast
import warnings

import datasets
import torch
from opacus import PrivacyEngine
from .task import Net, get_weights, load_data, set_weights, test, train
import logging
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context, Scalar

flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.INFO)
flwr_logger.propagate = False

warnings.filterwarnings("ignore", category=UserWarning)


class FlowerClient(NumPyClient):
    def __init__(
        self,
        train_loader,
        test_loader,
        target_delta,
        noise_multiplier,
        max_grad_norm,
        learning_rate,
    ) -> None:
        super().__init__()
        self.model = Net()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.target_delta = target_delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        model = self.model
        set_weights(model, parameters)

        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.learning_rate, momentum=0.9
        )

        privacy_engine = PrivacyEngine(secure_mode=False)
        (
            model,
            optimizer,
            self.train_loader,
        ) = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )

        epsilon = train(
            model,
            self.train_loader,
            privacy_engine,
            optimizer,
            self.target_delta,
            device=self.device,
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


def client_fn(context: Context):
    train_loader, test_loader = load_data(
        partition_id=cast(int, context.node_config["partition-id"]),
        num_partitions=cast(int, context.node_config["num-partitions"]),
        batch_size=cast(int, context.run_config["batch-size"]),
        alpha=cast(float, context.run_config["alpha"]),
    )

    return FlowerClient(
        train_loader,
        test_loader,
        context.run_config["target-delta"],
        context.run_config["noise-multiplier"],
        context.run_config["max-grad-norm"],
        context.run_config["learning-rate"],
    ).to_client()


app = ClientApp(client_fn=client_fn)
