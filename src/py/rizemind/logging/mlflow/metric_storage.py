import os
import tempfile

import mlflow
import numpy as np
from flwr.common import (
    Parameters,
    Scalar,
    parameters_to_ndarrays,
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

    def write_metrics(self, server_round: int, metrics: dict[str, Scalar]):
        """Logs a dictionary of metrics to the MLflow run for a specific server round.

        This method iterates through the provided metrics and logs each one to the
        active MLflow run, using the server round as the step.

        Args:
            server_round: The current round of federated learning, used as the
                'step' in MLflow.
            metrics: A dictionary mapping metric names (e.g., "accuracy")
                to their scalar values.
        """
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
