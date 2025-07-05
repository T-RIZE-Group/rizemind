from pathlib import Path
from typing import cast

import pandas as pd
import torch
from flwr.client import NumPyClient
from flwr.common import (
    Context,
    NDArrays,
    Scalar,
)
from tabpfn.model.loading import load_model_criterion_config
from tabpfn.regressor import TabPFNRegressor
from examples.tabpfn_centralized.tabpfn_training.finetuning.constant_utils import (
    Metrics,
)
from tabpfn_centralized.tabpfn_training.finetuning.constant_utils import (
    SupportedDevice,
    SupportedValidationMetric,
    TaskType,
)
from tabpfn_centralized.tabpfn_training.finetuning.finetune_main import (
    fine_tune_tabpfn,
    get_metric,
)
from tabpfn_centralized.tabpfn_training.task import (
    get_weights,
    load_data,
    load_weights_into_model,
    set_weights,
)
from sklearn.model_selection import train_test_split


class FlowerClient(NumPyClient):
    def __init__(
        self,
        *,
        train_dataset: pd.DataFrame,
        test_dataset: pd.DataFrame,
        base_model_path: str,
        model_path: str,
        batch_size: int,
    ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.base_model_path = base_model_path
        self.model_path = model_path
        self.batch_size = batch_size

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]):
        model, criterion, checkpoint_config = load_model_criterion_config(
            model_path=self.base_model_path,
            check_bar_distribution_criterion=True,
            cache_trainset_representation=False,
            which="regressor",
            version="v2",
            download=True,
            model_seed=42,
        )
        model = load_weights_into_model(
            model=model, parameters=parameters, config=config
        )
        model.criterion = criterion
        torch.save(
            dict(
                state_dict=model.state_dict(),
                config=checkpoint_config.__dict__,
            ),
            str(self.model_path),
        )
        metrics: Metrics = fine_tune_tabpfn(
            path_to_base_model=Path(self.model_path),
            save_path_to_fine_tuned_model=Path(self.model_path),
            time_limit=60,
            random_seed=42,
            finetuning_config={"learning_rate": 0.00001, "batch_size": self.batch_size},
            validation_metric=SupportedValidationMetric.R2,
            X_train=self.train_dataset.drop("target", axis=1, inplace=False),
            y_train=self.train_dataset["target"],
            X_val=self.test_dataset.drop("target", axis=1, inplace=False),
            y_val=self.test_dataset["target"],
            categorical_features_index=None,
            device=SupportedDevice.GPU,
            task_type=TaskType.REGRESSION,
            show_training_curve=False,
            logger_level=21,
            use_wandb=False,
        )
        model, _, _ = load_model_criterion_config(
            model_path=self.base_model_path,
            check_bar_distribution_criterion=True,
            cache_trainset_representation=False,
            which="regressor",
            version="v2",
            download=True,
            model_seed=42,
        )
        return (
            get_weights(model),
            self.train_dataset.shape[0],
            metrics.model_dump(),
        )

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]):
        model = TabPFNRegressor(model_path=self.model_path)
        Xy_train, Xy_test = train_test_split(self.test_dataset, train_size=0.1)
        # Model must be initialized before calling predict
        model.fit(Xy_train.drop("target", axis=1, inplace=False), Xy_train["target"])
        model = set_weights(model=model, parameters=parameters)
        y_pred = model.predict(Xy_test.drop("target", axis=1, inplace=False))
        validation_metric = get_metric(
            metric=SupportedValidationMetric.R2, problem_type=TaskType.REGRESSION
        )
        r2 = validation_metric(Xy_test["target"], y_pred)
        loss = validation_metric.convert_score_to_error(r2)
        return loss, self.test_dataset.shape[0], {"r2": cast(Scalar, r2)}


def client_fn(context: Context):
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    train_dataset, test_dataset = load_data(
        partition_id=partition_id, num_partitions=num_partitions
    )

    base_model_path = str(context.node_config["base-model-path"])
    unique_model_path = f"{base_model_path}-{partition_id}.ckpt"
    batch_size = int(context.run_config["batch-size"])

    return FlowerClient(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        base_model_path=base_model_path,
        model_path=unique_model_path,
        batch_size=batch_size,
    )
