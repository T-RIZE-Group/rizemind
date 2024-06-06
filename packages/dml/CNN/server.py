from typing import List, Tuple

import flwr as fl

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
# from dataPreparation import data_splitting
from utils import server_args_parser


args = server_args_parser()
train_method = args.train_method
pool_size = args.pool_size
num_rounds = args.num_rounds
num_clients_per_round = args.num_clients_per_round
num_evaluate_clients = args.num_evaluate_clients
centralised_eval = args.centralised_eval

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    print('----------server side--------')
    print('accuracies: ')
    print(accuracies)
    print('Metrics')
    print(metrics)
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
strategy = FedAvg(min_fit_clients=num_clients_per_round,
                min_available_clients=pool_size,
                evaluate_metrics_aggregation_fn=weighted_average)


# Define config
config = ServerConfig(num_rounds=num_rounds)


# Flower ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)

# dataset = '/home/iman/projects/kara/Projects/T-Rise/CNN/housing.csv'
# data_root_path = '/home/iman/projects/kara/Projects/T-Rise/CNN/'
# data_splitting(4, dataset, data_root_path, 0.3)
fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )


# Legacy mode
# if __name__ == "__main__":
#     from flwr.server import start_server
    

#     start_server(
#         server_address="0.0.0.0:8080",
#         config=config,
#         strategy=strategy,
#     )
