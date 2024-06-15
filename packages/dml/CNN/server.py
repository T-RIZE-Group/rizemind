from typing import List, Tuple

import flwr as fl

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
# from dataPreparation import data_splitting
from utils import server_args_parser
import pandas as pd
from flwr.common.logger import log
from logging import INFO
import torch
from torch.utils.data import DataLoader, TensorDataset
from server_utils import (
    eval_config,
    fit_config,
    evaluate_metrics_aggregation,
    get_evaluate_fn,
)


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

if centralised_eval:
    print('server side: centralised eval is activated')
    dataframes = []
    for i in range(args.pool_size):
        file_paths = f'/home/iman/projects/kara/Projects/T-Rize/archive/City_data/subset_{i}.csv'
        df = pd.read_csv(file_paths)
        sampled_data = df.sample(frac=0.3)
        city = df['City'].iloc[0]
        dataframes.append(sampled_data)

    dataset = pd.concat(dataframes, ignore_index=True)
    unique_cities = dataset['City'].unique()
    print(f'central test dataset includes these cities: {unique_cities}')
    dataset = dataset.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code', 'Latitude', 'Longitude'])
    dataset = dataset.dropna()
    log(INFO, "Loading centralised test set...")
    test_set = dict()
    test_set['inputs'] = dataset.drop(columns=['Price'])
    test_set['label'] = dataset['Price']

    # Prepare features and target
    X = test_set['inputs'].values
    y = test_set['label'].values

    # Reshape features for CNN input
    num_samples, num_features = X.shape
    X_reshaped = X.reshape((num_samples, 1, num_features))  # Reshape to 3D array for Conv1d (1 channel)

    # Convert to PyTorch tensors
    X_test = torch.tensor(X_reshaped, dtype=torch.float32)
    y_test = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # Create DataLoader
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_dmatrix = test_loader
# Define strategy
strategy = FedAvg(min_fit_clients=num_clients_per_round,
                min_available_clients=pool_size,
                evaluate_metrics_aggregation_fn=weighted_average,
                evaluate_fn=get_evaluate_fn(test_dmatrix, num_features) if centralised_eval else None,
                min_evaluate_clients=num_evaluate_clients if not centralised_eval else 0,
                # fraction_evaluate=1.0 if not centralised_eval else 0.0,
                # on_evaluate_config_fn=eval_config,
                # on_fit_config_fn=fit_config
)


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
