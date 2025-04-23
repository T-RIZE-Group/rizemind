"""pytorchexample: A Flower / PyTorch app."""

import torch
import logging
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import numpy.random as rnd

from .task import Net, get_weights, load_data, set_weights, test, train

# Configure logging to log output to a file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fl_logdynamic.log", mode="a"),
        logging.StreamHandler(),
    ],
)


# Define Flower Client with Local Differential Privacy (LDP)
class FlowerClient(NumPyClient):
    def __init__(
        self,
        trainloader,
        valloader,
        local_epochs,
        learning_rate,
        epsilon=0.1,  # Fixed epsilon
        client_id=0,  # Partition id for Shapley indexing
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
        self.client_id = int(client_id)
        # Initialize PRNG with a deterministic seed
        self.prng = rnd.default_rng(seed)

    def calculate_noise_scale(self):
        """noise_scale = 0.1 / epsilon."""
        return 0.1 / self.epsilon

    def fit(self, parameters, config):
        """Train the model with LDP and return updated parameters."""
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
        # Add Gaussian noise
        noisy_weights = self.add_noise(updated_weights)
        return noisy_weights, len(self.trainloader.dataset), results

    def add_noise(self, weights):
        """Add Gaussian noise to the model weights for LDP."""
        noisy_weights = []
        for w in weights:
            # Use PRNG to generate Gaussian noise
            noise = self.prng.normal(loc=0.0, scale=self.noise_scale, size=w.shape)
            noisy_weights.append(w + noise)
        return noisy_weights

    def evaluate(self, parameters, config):
        """Evaluate the model on this client's validation data."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)

        # Grab the round from the server config
        server_round_str = config.get("server_round", "0")
        server_round = int(server_round_str)

        # The server stored shapley_values as a comma-separated string
        shapley_str = config.get("shapley_values", "")
        if shapley_str:
            shapley_list = [float(x) for x in shapley_str.split(",") if x != ""]
            # Print if in range
            if self.client_id < len(shapley_list):
                val = shapley_list[self.client_id]
                logging.info(
                    f"Client {self.client_id}: shapley contribution (Round={server_round}) = {val}"
                )

                # Change epsilon only for rounds >= 1, if Shapley >= 0.15
                # if server_round >= 1 and val >= 0.2:
                #     new_epsilon = 0.1
                #     self.epsilon = new_epsilon
                #     self.noise_scale = 0.1 / new_epsilon
                #     logging.info(
                #         f"Client {self.client_id} updated epsilon to {self.epsilon} (noise_scale={self.noise_scale})"
                #     )

                if server_round >= 1 and val >= 0.15:
                    # Linearly map val in [0.15, 0.25] to epsilon in [0.1, 0.05]
                    new_epsilon = 0.1 - ((val - 0.15) / 0.1) * (0.1 - 0.05)
                    self.epsilon = new_epsilon
                    self.noise_scale = 0.1 / new_epsilon
                    logging.info(
                        f"Client {self.client_id} updated epsilon to {self.epsilon} (noise_scale={self.noise_scale})"
                    )

                # if server_round >= 0:
                #     # Linearly map val in [0.15, 0.25] to epsilon in [0.1, 0.05]
                #     new_epsilon = 0.1
                #     self.epsilon = new_epsilon
                #     self.noise_scale = 0.1 / new_epsilon
                #     logging.info(
                #         f"Client {self.client_id} updated epsilon to {self.epsilon} (noise_scale={self.noise_scale})"
                #     )

        # Return a 3-tuple: (loss, num_examples, metrics)
        return (loss, len(self.valloader.dataset), {"loss": loss, "accuracy": accuracy})


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    # Partition info
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Run config
    batch_size = context.run_config["batch-size"]
    alpha = context.run_config.get("alpha", 0.1)
    epsilon = 0.1
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size, alpha)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    return FlowerClient(
        trainloader,
        valloader,
        local_epochs,
        learning_rate,
        epsilon,
        client_id=partition_id,
    ).to_client()


app = ClientApp(client_fn)
