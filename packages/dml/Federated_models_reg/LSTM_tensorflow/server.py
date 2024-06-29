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
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
from LSTM_dataPreparation import CityPriceDataset
from sklearn.model_selection import train_test_split





class LSTMNet(tf.keras.Model):
    def __init__(self, input_size, hidden_size, num_layers, dropout, seq_length):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = LSTM(hidden_size, return_sequences=False, input_shape=(seq_length, input_size), dropout=dropout)
        self.fc1 = Dense(128)
        self.bn1 = BatchNormalization()
        self.fc2 = Dense(1)

    def call(self, x):
        x = self.lstm(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        return x
    
    def get_weights(self):
        return self.trainable_weights

    def set_weights(self, parameters):
        self.lstm.set_weights(parameters)

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
    input_dim = 10
   
    
    model = LSTMNet(input_dim, 72, 3,
                0.273, seq_length=12)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.006),
              loss='mse')

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
            Path("/home/iman/projects/kara/Projects/T-Rize/federated_models_classification/MLP/.cache/certificates/ca.crt").read_bytes(),
            Path("/home/iman/projects/kara/Projects/T-Rize/federated_models_classification/MLP/.cache/certificates/server.pem").read_bytes(),
            Path("/home/iman/projects/kara/Projects/T-Rize/federated_models_classification/MLP/.cache/certificates/server.key").read_bytes(),
        ),
    )


def get_evaluate_fn(model, pool_size):
    """Return an evaluation function for server-side evaluation."""

    # Load data here to avoid the overhead of doing it in `evaluate` itself
    
    print('server side: centralised eval is activated')
    data_dir = "/home/iman/projects/kara/Projects/T-Rize/archive/City_data"
    data_files = [f'subset_{i}.csv' for i in range(pool_size)]

    X = []
    y = []

    # Define scalers
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Fit the scalers on the entire dataset
    for data_file in data_files:
        full_path = os.path.join(data_dir, data_file)
        df = pd.read_csv(full_path)
        df = df.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code'])
        df = df.dropna()
        feature_scaler.partial_fit(df.drop(columns='Price').values)
        target_scaler.partial_fit(df["Price"].values.reshape(-1, 1))

    # Load the data using the fitted scalers
    for data_file in data_files:
        full_path = os.path.join(data_dir, data_file)
        dataset = CityPriceDataset(full_path, feature_scaler, target_scaler, seq_length=12)  # Adjust seq_length as needed
        X_data, y_data = dataset[:]
        X.append(X_data)
        y.append(y_data)

    X = np.concatenate(X)
    y = np.concatenate(y)

    # Get input size for the LSTM model
    input_size = X.shape[2]  # Adjust based on the shape of X

    # Split data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    x_test = X
    y_test = y
    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        dummy_input = np.zeros((1, 12, x_test.shape[2])) 
        model(dummy_input)
        model.set_weights(parameters)  # Update model with the latest parameters
        # loss, accuracy = model.evaluate(x_test, y_test)
        y_pred = model.predict(x_test)
        loss = mean_squared_error(y_test, y_pred)
        accuracy = r2_score(y_test, y_pred)
        print()
        print('accuracy evaluation on the server side:')
        print("accuracy: ", accuracy)
        print()
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
