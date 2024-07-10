import numpy as np
from sklearn.linear_model import LogisticRegression

from flwr.common import NDArrays
from sklearn.svm import SVR, SVC


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

def get_model_parameters(model: SVC) -> NDArrays:
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


def set_model_params(model: SVC, params: NDArrays) -> SVC:
    """Sets the parameters of a sklearn SVC model."""
    if len(params) > 0:
        # model.support_ = params[0]
        # model.support_vectors_ = params[1]
        # model.dual_coef_ = params[0]
        model.coef0 = params[0]
        model.intercept_ = params[1]
    return model

def set_initial_params(model: SVC):
    """Sets initial parameters as zeros Required since model params are uninitialized
    until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    dummy_x = np.zeros((100,11))
    dummy_y = np.random.randint(0,11, 100)
    model.fit(dummy_x, dummy_y)
    # pass
    
    
    
    # n_classes = 11  # MNIST has 10 classes
    # n_features = 10  # Number of features in dataset
    # model.classes_ = np.array([i for i in range(10)])

    # model.coef_ = np.zeros((n_classes, n_features))
    # if model.fit_intercept:
    #     model.intercept_ = np.zeros((n_classes,))
