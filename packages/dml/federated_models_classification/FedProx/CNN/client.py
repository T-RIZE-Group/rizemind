import argparse
import os
from pathlib import Path
import pandas as pd
import tensorflow as tf

import flwr as fl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from flwr_datasets import FederatedDataset
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def load_partition(idx: int, num_classes):
    """"""
    # Download and partition dataset
    dataset_path = f'/home/iman/projects/kara/Projects/T-Rize/archive/City_data_classification/subset_{idx}.csv'
   
    dataset = pd.read_csv(dataset_path)
    print()
    unique_city = dataset['City'].unique()
    print(f'Client number {idx} holds the data of {unique_city}')
    print()
    dataset = dataset.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code', 'Latitude', 'Longitude', 'Price'])
    dataset = dataset.dropna()
    city_num = dataset.iloc[0]['city_num']
    print()
    print(f'city code in the partition number {idx} is: {city_num}')
    print()
    partition = dataset
    
    # Divide data on each node: 80% train, 20% test
    partition_train, partition_test = train_test_split(partition, test_size=0.2)
    x_train = partition_train.drop(columns='price_label').values
    y_train = partition_train['price_label']
    x_test = partition_test.drop(columns='price_label').values
    y_test = partition_test['price_label']
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    

    
    
    # num_classes = len(dataset["price_label"].unique())
    
    print()
    print(f'number of classes in data of client {idx} is {num_classes}')
    print()

    y_train_categorical = to_categorical(y_train, num_classes=num_classes)
    y_test_categorical = to_categorical(y_test, num_classes=num_classes)
    
    return x_train, y_train_categorical, x_test, y_test_categorical


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



# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        print()
        print('accuracy on the client side after fit: ')
        print('accuracy: ', results['val_accuracy'])
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        print()
        print('accuracy on the client side in evaluate function')
        print('accuracy: ')
        print(accuracy)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--client-id",
        type=int,
        default=0,
        choices=range(0, 10),
        required=True,
        help="Specifies the artificial data partition of CIFAR10 to be used. "
        "Picks partition 0 by default",
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Set to true to quicky run the client using only 10 datasamples. "
        "Useful for testing purposes. Default: False",
    )
    args = parser.parse_args()

    # Load and compile Keras model
    input_dim = 11
    num_classes = 11
    x_train, y_train, x_test, y_test = load_partition(args.client_id, num_classes)
    
    model = create_mlp_model(input_dim, num_classes) 
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    # Load a subset of CIFAR-10 to simulate the local data partition
   

    # if args.toy:
    #     x_train, y_train = x_train[:10], y_train[:10]
    #     x_test, y_test = x_test[:10], y_test[:10]

    # Start Flower client
    client = CifarClient(model, x_train, y_train, x_test, y_test).to_client()

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client,
        root_certificates=Path("/home/iman/projects/kara/Projects/T-Rize/federated_models_classification/FedProx/CNN/.cache/certificates/ca.crt").read_bytes(),
    )




if __name__ == "__main__":
    main()
