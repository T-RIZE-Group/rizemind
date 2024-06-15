import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression, LinearRegression
from typing import Dict
from utils import server_args_parser
import pandas as pd
from flwr_datasets import FederatedDataset
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LinearRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    # fds = FederatedDataset(dataset="mnist", partitioners={"train": 10})
    # dataset = fds.load_split("test").with_format("numpy")
    
    # X_test, y_test = dataset["image"].reshape((len(dataset), -1)), dataset["label"]
    
    
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
    
    X_test = np.array(dataset.drop(columns='Price'))
    
    y_test = np.array(dataset["Price"])

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        
        # loss = log_loss(y_test, model.predict(X_test))
        print('mse calculation on the server side')
        predictions = model.predict(X_test)
        loss = mean_squared_error(y_test, predictions)
        print(loss)
        # accuracy = model.score(X_test, y_test)
        accuracy = r2_score(y_test, predictions)
        return loss, {"loss": loss, "accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LinearRegression()
    utils.set_initial_params(model)
    args = server_args_parser()
    train_method = args.train_method
    pool_size = args.pool_size
    num_rounds = args.num_rounds
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=pool_size,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
    )
