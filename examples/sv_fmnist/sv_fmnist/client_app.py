import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from .task import Net, get_weights, load_data, set_weights, test, train


# Define Flower Client for Shapley Calculation (No DP)
class FlowerClient(NumPyClient):
    def __init__(
        self,
        trainloader,
        valloader,
        local_epochs,
        learning_rate,
        client_id=0,  # Partition ID for tracking Shapley values
    ):
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.client_id = int(client_id)

    def fit(self, parameters, config):
        """Train the model and return updated parameters."""
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
        return updated_weights, len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the model on this client's validation data."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)

        # Grab the round from the server config
        server_round = int(config.get("server_round", "0"))

        # The server stored Shapley values as a comma-separated string
        shapley_str = config.get("shapley_values", "")
        if shapley_str:
            shapley_list = [float(x) for x in shapley_str.split(",") if x != ""]
            if self.client_id < len(shapley_list):
                val = shapley_list[self.client_id]
                print(
                    f"Client {self.client_id}: Shapley Contribution (Round {server_round}) = {val}"
                )

        return (loss, len(self.valloader.dataset), {"loss": loss, "accuracy": accuracy})


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    # Partition info
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Run config
    batch_size = context.run_config["batch-size"]
    alpha = context.run_config.get("alpha", 0.1)  # Default = 0.5 (Non-IID)
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size, alpha)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    return FlowerClient(
        trainloader, valloader, local_epochs, learning_rate, client_id=partition_id
    ).to_client()


app = ClientApp(client_fn)
