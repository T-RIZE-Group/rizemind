from collections import OrderedDict
from typing import cast

import numpy as np
import polars as pl
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from tabpfn import TabPFNRegressor


def load_model(
    sample_X: pd.DataFrame,
    sample_y: np.ndarray,
):
    model = TabPFNRegressor(
        average_before_softmax=True,
        device="auto",
        inference_precision="auto",
        fit_mode="fit_preprocessors",
    )
    model.fit(sample_X, sample_y)
    return model


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def test(model: TabPFNRegressor, X_test, y_test):
    y_pred = model.predict(y_test)

    rmse = cast(float, root_mean_squared_error(y_test, y_pred))
    mae = cast(float, mean_absolute_error(y_test, y_pred))
    r2 = cast(float, r2_score(y_test, y_pred))

    return r2, rmse, mae


def load_data(
    path: str,
    label_name: str,
):
    df: pl.DataFrame = pl.DataFrame()
    df = pl.read_csv(path)
    X, y = df.drop(label_name), df.select(pl.col(label_name)).to_numpy()
    y = y.ravel()
    return X.to_pandas(), y
