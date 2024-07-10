from typing import Dict, Optional, Tuple
from pathlib import Path

import flwr as fl
import tensorflow as tf
from flwr.common.logger import log
from flwr_datasets import FederatedDataset
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, LSTM
from keras.utils import to_categorical
import argparse
import pandas as pd
from logging import INFO
from sklearn.preprocessing import StandardScaler




def create_mlp_model(input_dim, num_classes):
    
    
    model = Sequential([
    Conv1D(64, kernel_size=3, padding='same', activation='relu', input_shape=(input_dim, 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Conv1D(256, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')])
    
    return model

def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    # model = tf.keras.applications.EfficientNetB0(
    #     input_shape=(32, 32, 3), weights=None, classes=10
    # )
    
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--train-method",
    #     default="bagging",
    #     type=str,
    #     choices=["bagging", "cyclic"],
    #     help="Training methods selected from bagging aggregation or cyclic training.",
    # )
    parser.add_argument(
        "--pool-size", default=2, type=int, help="Number of total clients."
    )
    parser.add_argument(
        "--num-rounds", default=5, type=int, help="Number of FL rounds."
    )
    args = parser.parse_args()
    print("number of client in simulation: ", args.pool_size)
    num_classes = 11
    input_dim = 11
    model = create_mlp_model(input_dim, num_classes) 
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=args.pool_size,
        min_evaluate_clients=args.pool_size,
        min_available_clients=args.pool_size,
        evaluate_fn=get_evaluate_fn(model, args.pool_size),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        certificates=(
            Path("/home/iman/projects/kara/Projects/T-Rize/federated_models_classification/FedAvg/CNN/.cache/certificates/ca.crt").read_bytes(),
            Path("/home/iman/projects/kara/Projects/T-Rize/federated_models_classification/FedAvg/CNN/.cache/certificates/server.pem").read_bytes(),
            Path("/home/iman/projects/kara/Projects/T-Rize/federated_models_classification/FedAvg/CNN/.cache/certificates/server.key").read_bytes(),
        ),
    )


def get_evaluate_fn(model, pool_size):
    """Return an evaluation function for server-side evaluation."""

    # Load data here to avoid the overhead of doing it in `evaluate` itself
    
    print('server side: centralised eval is activated')
    dataframes = []
    for i in range(pool_size):
        file_paths = f'/home/iman/projects/kara/Projects/T-Rize/archive/City_data_classification/subset_{i}.csv'
        df = pd.read_csv(file_paths)
        sampled_data = df.sample(frac=0.3)
        city = df['City'].iloc[0]
        dataframes.append(sampled_data)

    dataset = pd.concat(dataframes, ignore_index=True)
    unique_cities = dataset['City'].unique()
    print()
    print(f'central test dataset includes these cities: {unique_cities}')
    print()
    dataset = dataset.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code', 'Latitude', 'Longitude', 'Price'])
    dataset = dataset.dropna()
    log(INFO, "Loading centralised test set...")
    city_num = dataset['city_num'].unique()
    print()
    print('unique city num in the gobal test datset: ')
    print(city_num)
    print()
    
    x_test = dataset.drop(columns=['price_label']).values
    y_test = dataset['price_label']
    y_test_categorical = to_categorical(y_test, num_classes=11)
    
    scaler = StandardScaler()
    x_test = scaler.fit_transform(x_test)
    
    print()
    print('global test data on the server side:')
    print(x_test[0])
    print()
    
    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_test, y_test_categorical)
        print()
        print('accuracy evaluation on the server side:')
        print("accuracy: ", accuracy)
        print()
        model_path = f"/home/iman/projects/kara/Projects/T-Rize/federated_models_classification/FedAvg/CNN/global_models/global_model_{server_round}.h5"
        model.save(model_path)
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 50 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()
