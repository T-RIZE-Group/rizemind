import os
import tempfile
from logging import WARNING
from typing import cast

import mlflow
import numpy as np
import pandas as pd
from flwr.common import (
    Parameters,
    Scalar,
    log,
    parameters_to_ndarrays,
)
from mlflow.entities import RunStatus, ViewType

from rizemind.logging import (
    TRAIN_METRIC_HISTORY_KEY,
    FitMetricHistory,
)
from rizemind.logging.base_metric_storage import BaseMetricStorage


class MLFLowMetricStorage(BaseMetricStorage):
    """A concrete implementation of `BaseMetricStorage` that logs metrics and models to an MLflow tracking server.

    This class integrates Flower federated learning with MLflow, enabling centralized
    tracking of experiments, metrics, and model artifacts. Upon initialization, it
    connects to a specified MLflow tracking URI, sets up an experiment, and creates
    a new run to store all subsequent data.

    Attributes:
        experiment_name: The name of the MLflow experiment.
        run_name: The name of the MLflow run.
        mlflow_uri: The URI for the MLflow tracking server.
        mlflow_client: The MLflow client for interacting with the API.
        run_id: The unique ID of the MLflow run created for this session.
    """

    def __init__(self, experiment_name: str, run_name: str, mlflow_uri: str):
        """Initializes the MLFLowMetricStorage and sets up the MLflow run.

        This constructor connects to the MLflow tracking server, ensures the specified
        experiment exists, and starts a new run. The run ID is stored for logging
        metrics and artifacts throughout the federated learning process.

        Args:
            experiment_name: The name of the experiment in MLflow. If it
                doesn't exist, it will be created.
            run_name: The name assigned to the run within the experiment.
            mlflow_uri: The connection URI for the MLflow tracking server.
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.mlflow_uri = mlflow_uri
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.mlflow_client = mlflow.MlflowClient()
        mlflow.set_experiment(experiment_name=self.experiment_name)
        run = mlflow.start_run(run_name=self.run_name)
        self.run_id: str = run.info.run_id
        mlflow.end_run()

        self._best_loss = np.inf
        self._current_round_model = Parameters(tensors=[], tensor_type="")

    def write_fit_metrics(self, server_round: int, metrics: dict[str, Scalar]):
        """Logs fit_metrics that are serialized by `TRAIN_METRIC_HISTORY_KEY` flag to mlflow.

        This method deserializes the value of `TRAIN_METRIC_HISTORY_KEY` key in metrics dictionary.
        This method is used to log metrics of each SuperNodes to SuperLink's mlflow server.
        This allows the server to check logs of each client for better troubleshooting.

        Args:
            server_round: The current round of federated learning, used as the 'step' in Mlflow.
            metrics: A dictionary mapping containing `TRAIN_METRIC_HISTORY_KEY`.
        """
        for key, value in metrics.items():
            # Skip non-TRAIN_METRIC_HISTORY_KEY metrics
            if not key.startswith(TRAIN_METRIC_HISTORY_KEY):
                log(
                    WARNING,
                    f"Non-{TRAIN_METRIC_HISTORY_KEY} key ({key}) was found in fit metrics. Skipping logging key.",
                )
                continue

            # Get client's run and experiment
            client_id = key.lstrip(TRAIN_METRIC_HISTORY_KEY)
            client_experiment_name = f"{self.experiment_name}_client_{client_id}"
            mlflow.set_experiment(client_experiment_name)

            # Get the last epoch and run_id
            runs_df = cast(
                pd.DataFrame,
                mlflow.search_runs(
                    experiment_names=[client_experiment_name],
                    filter_string=f"tags.mlflow.runName = '{self.run_name}'",
                    run_view_type=ViewType.ALL,
                    order_by=["attributes.end_time DESC"],
                    max_results=1,
                ),
            )
            epochs_passed = 0
            run_id = ""
            if runs_df.empty:
                # If a previous run doesn't exist
                # start a run with the given name
                mlflow.start_run(run_name=self.run_name)
            else:
                # If a previous run exists
                # update the number of epochs passed
                epochs_passed = int(cast(int, runs_df.loc[0, "metrics.epochs"]))
                # continue the run
                run_id = cast(str, runs_df.loc[0, "run_id"])
                mlflow.start_run(run_id=run_id)

            train_metric_history = FitMetricHistory.deserialize(
                serialized_train_metric_history=cast(str, value)
            )
            epochs_this_round = 0
            for metric, phases in train_metric_history.model_dump().items():
                for phase, values in phases.items():
                    for step, metric_value in enumerate(values):
                        mlflow.log_metric(
                            key=f"{phase}_{metric}",
                            value=metric_value,
                            step=step + epochs_passed,
                        )
                    epochs_this_round = max(epochs_this_round, len(values))

            epochs_passed += epochs_this_round
            mlflow.log_metric(key="epochs", value=epochs_passed)
            mlflow.end_run(status=RunStatus.to_string(RunStatus.FINISHED))

    def write_eval_metrics(self, server_round: int, metrics: dict[str, Scalar]):
        """Logs a dictionary of metrics to the MLflow run for a specific server round.

        This method iterates through the provided metrics and logs each one to the
        active MLflow run, using the server round as the step.

        Args:
            server_round: The current round of federated learning, used as the
                'step' in MLflow.
            metrics: A dictionary mapping metric names (e.g., "accuracy")
                to their scalar values.
        """
        mlflow.set_experiment(experiment_name=self.experiment_name)
        for k, v in metrics.items():
            self.mlflow_client.log_metric(
                run_id=self.run_id, key=k, value=float(v), step=server_round
            )

    def update_current_round_model(self, parameters: Parameters):
        """Temporarily stores the model parameters for the current round in memory.

        This method holds the latest model parameters so they can be saved as an
        MLflow artifact later by `update_best_model` if this model proves to be
        the best one based on its loss.

        Args:
            parameters: The model parameters from the current round.
        """
        self._current_round_model = parameters

    def update_best_model(self, server_round: int, loss: float):
        """Saves the current model as an MLflow artifact if its loss is the lowest seen so far.

        It compares the provided loss with its internally tracked best loss.
        If the new loss is lower, it updates the best loss and serializes the
        in-memory model parameters to a temporary `.npz` file. This file is then
        uploaded as an artifact to the MLflow run. It also logs the best round
        and loss as metrics.

        Args:
            server_round: The server round that produced this model.
            loss: The loss value of the current model, used to determine
                if it is the new best model.
        """
        if loss < self._best_loss:
            self._best_loss = loss
            with tempfile.TemporaryDirectory() as tmp:
                ndarray_params = parameters_to_ndarrays(self._current_round_model)
                path = os.path.join(tmp, "weights.npz")
                np.savez(path, *ndarray_params)
                self.mlflow_client.log_artifact(
                    run_id=self.run_id,
                    local_path=path,
                    artifact_path="flwr_best_model_params",
                )
                self.mlflow_client.log_metric(
                    run_id=self.run_id, key="best_round", value=server_round
                )
                self.mlflow_client.log_metric(
                    run_id=self.run_id, key="avg_loss", value=loss, step=server_round
                )
