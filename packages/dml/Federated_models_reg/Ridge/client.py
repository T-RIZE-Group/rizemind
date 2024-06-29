import argparse
import warnings

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import log_loss
from utils import client_args_parser
import flwr as fl
import utils
from flwr_datasets import FederatedDataset
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    N_CLIENTS = 2
    
    args = client_args_parser()

    
    # args = parser.parse_args()
    partition_id = args.partition_id

    # Load the partition data

    # fds = FederatedDataset(dataset="mnist", partitioners={"train": N_CLIENTS})

    # dataset = fds.load_partition(partition_id, "train").with_format("numpy")
    # X, y = dataset["image"].reshape((len(dataset), -1)), dataset["label"]
    # # Split the on edge data: 80% train, 20% test
    # X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    # y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]

    dataset = pd.read_csv(f'/home/iman/projects/kara/Projects/T-Rize/archive/City_data/subset_{args.partition_id}.csv')
    print(f'client number {args.partition_id} holds the data of {dataset["City"].iloc[0]}')
    print()
    dataset = dataset.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code', 'Latitude', 'Longitude'])
    dataset = dataset.dropna()
   

    X = dataset.drop(columns='Price')
    y = dataset['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15,
                                                                                random_state = 42)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    # Create LogisticRegression Model
    # model = LogisticRegression(
    #     penalty="l2",
    #     max_iter=1,  # local epoch
    #     warm_start=True,  # prevent refreshing weights when fitting
    # )
    
    model = Ridge()
    
    # model = Ridge(alpha=1.0)

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)

    # Define Flower client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            print('get params on the client side')
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
                
            print(f"Training finished for round {config['server_round']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            
            utils.set_model_params(model, parameters)
            predictions = model.predict(X_test)
            loss = mean_squared_error(y_test, predictions)
            accuracy = r2_score(y_test, predictions)
            print('mse= ', loss, 'r2 = ', accuracy)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_client(
        server_address="0.0.0.0:8080", client=MnistClient().to_client()
    )
