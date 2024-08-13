import numpy as np
import flwr as fl
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Partition the data into NUM_CLIENTS parts
NUM_CLIENTS = 5
x_split = np.split(x_train, NUM_CLIENTS)
y_split = np.split(y_train, NUM_CLIENTS)

x_trains, y_trains, x_tests, y_tests = {}, {}, {}, {}
num_data_in_split = x_split[0].shape[0]
for idx, (client_x, client_y) in enumerate(zip(x_split, y_split)):
    train_end_idx = int(0.8 * num_data_in_split)
    x_trains[str(idx)] = client_x[:train_end_idx]
    y_trains[str(idx)] = client_y[:train_end_idx]
    x_tests[str(idx)] = client_x[train_end_idx:]
    y_tests[str(idx)] = client_y[train_end_idx:]


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.model.build((32, 28, 28, 1))
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model.set_weights(parameters)
        self.model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, batch_size=32, verbose=0)
        return loss, len(self.X_test), {"accuracy": accuracy}


def create_client(cid, model_class):
    """Create a Flower client representing a single organization."""
    model = model_class()
    return FlowerClient(model, x_trains[cid], y_trains[cid], x_tests[cid], y_tests[cid])


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": int(sum(accuracies)) / int(sum(examples))}
