from typing import OrderedDict, cast

import pandas as pd
import torch
import torch.nn as nn
from flwr.common import NDArrays, Scalar
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from tabpfn import TabPFNRegressor
from tabpfn.base import PerFeatureTransformer


def get_weights(net: nn.Module) -> NDArrays:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(
    *,
    model: TabPFNRegressor,
    parameters: NDArrays,
) -> TabPFNRegressor:
    params_dict = zip(model.model_.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.model_.load_state_dict(state_dict)
    return model


def load_weights_into_model(
    *,
    model: PerFeatureTransformer,
    parameters: NDArrays,
    config: dict[str, Scalar],
) -> PerFeatureTransformer:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict)
    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    global fds
    if fds is None:
        train_partitioner = IidPartitioner(num_partitions=num_partitions)
        test_partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="wwydmanski/blog-feedback",
            partitioners={"train": train_partitioner, "test": test_partitioner},
        )
    train_df = cast(pd.DataFrame, fds.load_partition(partition_id, "train").to_pandas())
    test_df = cast(pd.DataFrame, fds.load_partition(partition_id, "test").to_pandas())

    return train_df, test_df
