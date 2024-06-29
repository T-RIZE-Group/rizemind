import numpy as np
from sklearn.linear_model import LogisticRegression

from flwr.common import NDArrays
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd


# def get_model_parameters(model: LogisticRegression) -> NDArrays:
#     """Returns the parameters of a sklearn LogisticRegression model."""
#     if model.fit_intercept:
#         params = [
#             model.coef_,
#             model.intercept_,
#         ]
#     else:
#         params = [
#             model.coef_,
#         ]
#     return params

def get_model_parameters(model: SVR) -> NDArrays:
    """Returns the parameters of a sklearn SVC model."""
    if hasattr(model, 'support_'):
        params = [
            # model.support_,
            # model.support_vectors_,
            # model.dual_coef_,
            model.coef0,
            model.intercept_,
        ]
    else:
        params = []
    return params


# def set_model_params(model: LogisticRegression, params: NDArrays) -> LogisticRegression:
#     """Sets the parameters of a sklean LogisticRegression model."""
#     model.coef_ = params[0]
#     if model.fit_intercept:
#         model.intercept_ = params[1]
#     return model


def set_model_params(model: SVR, params: NDArrays) -> SVR:
    """Sets the parameters of a sklearn SVC model."""
    if len(params) > 0:
        # model.support_ = params[0]
        # model.support_vectors_ = params[1]
        # model.dual_coef_ = params[0]
        model.coef0 = params[0]
        model.intercept_ = params[1]
    return model

def set_initial_params(model: SVR, partition_id):
    """Sets initial parameters as zeros Required since model params are uninitialized
    until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    # dummy_x = np.random.rand(10,10)
    # dummy_y = np.random.rand(10,1)
    # model.fit(dummy_x, dummy_y)
    
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
    model.fit(X_train, y_train)
    # n_classes = 10  # MNIST has 10 classes
    # n_features = 784  # Number of features in dataset
    # model.classes_ = np.array([i for i in range(10)])

    # model.coef_ = np.zeros((n_classes, n_features))
    # if model.fit_intercept:
    #     model.intercept_ = np.zeros((n_classes,))
