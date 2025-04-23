import logging

import numpy as np
import numpy.random as rnd
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from .task import (
    Net,
    get_weights,
    load_data,
    set_weights,
    test,
    train,
)

# Configure logging to save logs to "client_app.log"
logging.basicConfig(
    filename="client_app.log",
    filemode="a",  # Append mode; change to "w" to overwrite on each run
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


# Define Flower Client with LDP
class FlowerClient(NumPyClient):
    def __init__(
        self,
        trainloader,
        valloader,
        local_epochs,
        learning_rate,
        epsilon=1.0,
        seed=42,  # Add a seed for reproducibility
    ):
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epsilon = epsilon
        self.noise_scale = self.calculate_noise_scale()
        self.prng = rnd.default_rng(seed)
        logging.info(
            "Initialized FlowerClient with epsilon=%s and seed=%d", epsilon, seed
        )

    def calculate_noise_scale(self):
        """
        Calculate the standard deviation of the Gaussian noise based on epsilon.
        Here, we use a reduced scaling where noise_scale = 0.1 / epsilon.
        """
        noise_scale = 0.1 / self.epsilon
        logging.info("Calculated noise scale: %f", noise_scale)
        return noise_scale

    def fit(self, parameters, config):
        """Train the model and return the updated parameters with noise added for LDP."""
        logging.info("Starting fit with parameters and config: %s", config)
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
        )
        updated_weights = get_weights(self.net)
        noisy_weights = self.add_noise(updated_weights)
        logging.info("Completed fit; returning noisy weights.")
        return noisy_weights, len(self.trainloader.dataset), results

    def add_noise(self, weights):
        """Add Gaussian noise to the model weights for Local Differential Privacy and save the noise values."""
        noisy_weights = []
        with open("prng_noise_output.bin", "ab") as f:
            for w in weights:
                noise = self.prng.normal(loc=0.0, scale=self.noise_scale, size=w.shape)
                f.write(noise.astype(np.float32).tobytes())
                noisy_w = w + noise
                noisy_weights.append(noisy_w)
        logging.info("Added noise to weights and saved noise values to file.")
        return noisy_weights

    def evaluate(self, parameters, config):
        """Evaluate the model on the client's validation data."""
        logging.info("Evaluating model with provided parameters.")
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        logging.info("Evaluation results: loss=%f, accuracy=%f", loss, accuracy)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    batch_size = context.run_config["batch-size"]
    alpha = context.run_config.get("alpha", 0.5)
    epsilon = context.run_config.get("epsilon", 1.0)
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size, alpha)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    logging.info(
        "Client node %d: partition_id=%d, local_epochs=%d",
        partition_id,
        partition_id,
        local_epochs,
    )

    return FlowerClient(
        trainloader, valloader, local_epochs, learning_rate, epsilon
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
logging.info("ClientApp created and ready to run.")
