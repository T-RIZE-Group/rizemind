import argparse
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.svm import SVR
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer



import flwr as fl
import utils
from flwr_datasets import FederatedDataset

if __name__ == "__main__":
    N_CLIENTS = 10

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition-id",
        type=int,
        default=0,
        choices=range(0, N_CLIENTS),
        required=True,
        help="Specifies the artificial data partition",
    )
    args = parser.parse_args()
    partition_id = args.partition_id

    # Load the partition data

    dataset = pd.read_csv(f'/home/iman/projects/kara/Projects/T-Rize/archive/City_data/subset_{partition_id}.csv')
    print(f'client number {partition_id} holds the data of {dataset["City"].iloc[0]}')
    print()
    city  = dataset['City'].unique()
    print(f'city for {partition_id} is {city}')
    dataset = dataset.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code', 'Latitude', 'Longitude'])

    dataset = dataset.dropna()
    
    inputs = dataset.drop(columns="Price")
    print(f'train data on client number {partition_id}', inputs.shape[1])
    labels = dataset["Price"]
    inputs = inputs.to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size = 0.15,
                                                                            random_state = 42)

    
    model = SVR(
        kernel='rbf'
    )
    model.fit(X_train, y_train)

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model, args.partition_id)

    # Define Flower client
    class SVRClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
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
            y_pred = model.predict(X_test)
            loss = mean_squared_error(y_pred, y_test)
            # loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = r2_score(y_pred, y_test)
            
            print()
            print(f'accuracy on the client numebr {args.partition_id} is {accuracy}')
            print()
            # accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    # Myclient = MnistClient()
    # config=fl.server.ServerConfig(num_rounds=5)
    # params = Myclient.get_parameters()
    # utils.set_model_params(model, params)
    # Myclient.fit(params, config)
    # Start Flower client
    fl.client.start_client(
        server_address="0.0.0.0:8080", client=SVRClient().to_client()
    )
