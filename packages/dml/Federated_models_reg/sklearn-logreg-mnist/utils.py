import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
import argparse

from flwr.common import NDArrays


def get_model_parameters(model: LinearRegression) -> NDArrays:
    """Returns the parameters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_.reshape(-1),
            model.intercept_,
        ]
    else:
        params = [
            model.coef_.reshape(-1),
        ]
    return params


def set_model_params(model: LinearRegression, params: NDArrays) -> LinearRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    # model.coef_ = params[0]
    # if model.fit_intercept:
    #     model.intercept_ = params[1]
    
    
    # model.coef_ = np.array(params[0]).reshape(1, -1)
    # if model.fit_intercept:
    #     model.intercept_ = np.array(params[1])
    
    coef_shape = model.coef_.shape
    model.coef_ = np.array(params[0]).reshape(coef_shape)  # Reshape to the original shape
    if model.fit_intercept:
        model.intercept_ = np.array(params[1])
    return model


def set_initial_params(model: LinearRegression):
    """Sets initial parameters as zeros Required since model params are uninitialized
    until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    n_classes = 1  # MNIST has 10 classes
    n_features = 10  # Number of features in dataset
    # model.classes_ = np.array([i for i in range(n_classes)])
    model.classes_ = np.array([0, 1]) 

    model.coef_ = np.zeros((1, n_features))
    print('inside set initial params')
    print('length of coeff')
    print(model.coef_)
    print(len(model.coef_))
    
    if model.fit_intercept:
        # model.intercept_ = np.zeros((n_classes,))
        model.intercept_ = np.zeros((1,))

def client_args_parser():
    """Parse arguments to define experimental settings on client side."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-method",
        default="bagging",
        type=str,
        choices=["bagging", "cyclic"],
        help="Training methods selected from bagging aggregation or cyclic training.",
    )
    parser.add_argument(
        "--num-partitions", default=10, type=int, help="Number of partitions."
    )
    parser.add_argument(
        "--partitioner-type",
        default="uniform",
        type=str,
        choices=["uniform", "linear", "square", "exponential"],
        help="Partitioner types.",
    )
    parser.add_argument(
        "--partition-id",
        default=0,
        type=int,
        help="Partition ID used for the current client.",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed used for train/test splitting."
    )
    parser.add_argument(
        "--test-fraction",
        default=0.2,
        type=float,
        help="Test fraction for train/test splitting.",
    )
    parser.add_argument(
        "--centralised-eval",
        default = False,
        action="store_true",
        help="Conduct evaluation on centralised test set (True), or on hold-out data (False).",
    )
    parser.add_argument(
        "--scaled-lr",
        action="store_true",
        help="Perform scaled learning rate based on the number of clients (True).",
    )

    args = parser.parse_args()
    return args


def server_args_parser():
    """Parse arguments to define experimental settings on server side."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-method",
        default="bagging",
        type=str,
        choices=["bagging", "cyclic"],
        help="Training methods selected from bagging aggregation or cyclic training.",
    )
    parser.add_argument(
        "--pool-size", default=2, type=int, help="Number of total clients."
    )
    parser.add_argument(
        "--num-rounds", default=5, type=int, help="Number of FL rounds."
    )
    parser.add_argument(
        "--num-clients-per-round",
        default=2,
        type=int,
        help="Number of clients participate in training each round.",
    )
    parser.add_argument(
        "--num-evaluate-clients",
        default=2,
        type=int,
        help="Number of clients selected for evaluation.",
    )
    parser.add_argument(
        "--centralised-eval",
        # default = True,
        action="store_true",
        help="Conduct centralised evaluation (True), or client evaluation on hold-out data (False).",
    )

    args = parser.parse_args()
    return args
