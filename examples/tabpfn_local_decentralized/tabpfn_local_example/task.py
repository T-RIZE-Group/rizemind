from collections import OrderedDict
from typing import cast

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from tabpfn import TabPFNRegressor

from rizemind.configuration.toml_config import TomlConfig
from flwr_datasets.partitioner import IidPartitioner
from datasets import Dataset


def load_model(
    X_sample: pd.DataFrame,
    y_sample: np.ndarray,
):
    model = TabPFNRegressor(
        average_before_softmax=True,
        device="auto",
        inference_precision="auto",
        fit_mode="fit_preprocessors",
    )
    model.fit(X_sample, y_sample)
    return model


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def test(model: TabPFNRegressor, X_test, y_test):
    y_pred = model.predict(X_test)

    rmse = cast(float, root_mean_squared_error(y_test, y_pred))
    mae = cast(float, mean_absolute_error(y_test, y_pred))
    r2 = cast(float, r2_score(y_test, y_pred))

    return r2, rmse, mae


# def load_data(
#     path: str,
#     label_name: str,
# ):
#     df: pl.DataFrame = pl.DataFrame()
#     df = pl.read_csv(path)
#     df = df.sample(100)
#     X, y = df.drop(label_name), df.select(pl.col(label_name)).to_numpy()
#     y = y.ravel()
#     return X.to_pandas(), y

partitioner = None  # Cache partitioner
i = 0


def load_data(partition_id: int, num_partitions: int):
    load_dotenv()
    config = TomlConfig("./pyproject.toml")

    global partitioner
    if partitioner is None:
        dataset_path = cast(str, config.get("tool.dataset.config.dataset_path"))
        df = pl.read_csv(dataset_path)
        dataset = Dataset.from_polars(df)
        partitioner = IidPartitioner(num_partitions)
        partitioner.dataset = dataset

    global i
    partition = partitioner.load_partition((partition_id + i) % num_partitions)
    i += 1
    df = cast(pl.DataFrame, partition.to_polars())
    label_name = cast(str, config.get("tool.dataset.config.label_name"))

    X, y = df.drop(label_name), df.select(pl.col(label_name)).to_numpy()
    y = y.ravel()
    return X.to_pandas(), y
