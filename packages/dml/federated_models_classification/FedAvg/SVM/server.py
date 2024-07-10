import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from typing import Dict

from flwr_datasets import FederatedDataset
import argparse
import pandas as pd
from logging import INFO
from flwr.common.logger import log
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score




parser = argparse.ArgumentParser()

parser.add_argument(
    "--pool-size", default=2, type=int, help="Number of total clients."
)
parser.add_argument(
    "--num-rounds", default=5, type=int, help="Number of FL rounds."
)
args = parser.parse_args()

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: SVC):
    """Return an evaluation function for server-side evaluation."""
    print()
    print('get evaluate function on server side')
    print()
    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    # fds = FederatedDataset(dataset="mnist", partitioners={"train": 10})
    # dataset = fds.load_split("test").with_format("numpy")
    # X_test, y_test = dataset["image"].reshape((len(dataset), -1)), dataset["label"]
    dataframes= [] 
    for i in range(args.pool_size):
        file_paths = f'/home/iman/projects/kara/Projects/T-Rize/archive/City_data_classification/subset_{i}.csv'
        df = pd.read_csv(file_paths)
        sampled_data = df.sample(frac=0.3)
        city = df['City'].iloc[0]
        dataframes.append(sampled_data)
    
    dataset = pd.concat(dataframes, ignore_index=True)
   
    unique_cities = dataset['City'].unique()

    print(f'central test dataset includes these cities: {unique_cities}')
    dataset = dataset.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code', 'Latitude', 'Longitude', 'Price'])
    dataset = dataset.dropna()
    log(INFO, "Loading centralised test set...")


    X_test = dataset.drop(columns='price_label')
    y_test = dataset["price_label"]
    
    print()
    print(f'number of features in global test data on the server side is {X_test.shape[1]}')
    print()
    
    X_test = X_test.to_numpy()


    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        y_pred = model.predict(X_test)
        loss = mean_squared_error(y_pred, y_test)
        # loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = r2_score(y_pred, y_test)
        # accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=args.pool_size,
        min_evaluate_clients=args.pool_size,
        min_available_clients=args.pool_size,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    )
