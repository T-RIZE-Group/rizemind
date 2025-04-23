import numpy.random as rnd
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from .task import Net, get_weights, load_data, set_weights, test, train


# Define Flower Client with LDP
class FlowerClient(NumPyClient):
    def __init__(
        self,
        trainloader,
        valloader,
        local_epochs,
        learning_rate,
        epsilon=1.0,
        #################
        seed=42,  # Add a seed for reproducibility
        ####################333
    ):
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epsilon = epsilon
        self.noise_scale = self.calculate_noise_scale()

        ######################
        # Initialize PRNG with a deterministic seed
        self.prng = rnd.default_rng(seed)
        ######################

    def calculate_noise_scale(self):
        """
        Calculate the standard deviation of the Gaussian noise based on epsilon.
        Here, we use a reduced scaling where noise_scale = 0.1 / epsilon.
        """
        return 0.1 / self.epsilon

    def fit(self, parameters, config):
        """Train the model and return the updated parameters with noise added for LDP."""
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
        # Add Gaussian noise to each parameter
        noisy_weights = self.add_noise(updated_weights)
        return noisy_weights, len(self.trainloader.dataset), results

    def add_noise(self, weights):
        """Add Gaussian noise to the model weights for Local Differential Privacy."""
        noisy_weights = []
        for w in weights:
            ###################
            # Use PRNG to generate Gaussian noise
            noise = self.prng.normal(loc=0.0, scale=self.noise_scale, size=w.shape)
            ##########################################################
            # noise = np.random.normal(loc=0.0, scale=self.noise_scale, size=w.shape)
            noisy_w = w + noise
            noisy_weights.append(noisy_w)
        return noisy_weights

    def evaluate(self, parameters, config):
        """Evaluate the model on the client's validation data."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    # Read the node_config to fetch data partition associated with this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = context.run_config["batch-size"]
    alpha = context.run_config.get("alpha", 0.5)  # Default alpha=0.5 for Dirichlet
    epsilon = context.run_config.get("epsilon", 1.0)  # Default epsilon=1.0
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size, alpha)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    # Return Client instance
    return FlowerClient(
        trainloader, valloader, local_epochs, learning_rate, epsilon
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
