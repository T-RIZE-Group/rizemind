"""signedupdates: A Flower / TensorFlow app."""

import os

import keras
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from keras import layers

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model(learning_rate: float = 0.001):
    # Define a simple CNN for CIFAR-10 and set Adam optimizer
    model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions):
    # Download and partition dataset
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2)
    x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
    x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]

    return x_train, y_train, x_test, y_test


def evaluate_fn(
    server_round: int, parameters, config: dict
) -> tuple[float, dict[str, float]]:
    """
    Evaluation function for Flower federated learning.

    Parameters:
        server_round (int): The current round of federated learning.
        parameters (NDArrays): List of model weights as numpy arrays.
        config (dict): A configuration dict which may contain keys such as "partition_id" and "num_partitions".

    Returns:
        A tuple containing:
            - loss (float): The evaluation loss.
            - metrics (dict): A dictionary containing evaluation metrics, e.g. accuracy.
    """
    # Create a new instance of the model and set its weights
    model = load_model()
    model.set_weights(parameters)

    # Get partitioning parameters from config if available, else use defaults
    partition_id = config.get("partition_id", 0)
    num_partitions = config.get("num_partitions", 1)

    # Load test data
    _, _, x_test, y_test = load_data(partition_id, num_partitions)

    # Evaluate the model on test data (using verbose=0 to suppress output)
    loss, accuracy = model.evaluate(x_test, y_test, verbose="auto")

    # Return the loss and a dictionary of metrics
    return loss, {"accuracy": accuracy}
