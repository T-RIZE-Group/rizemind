import os
import tempfile

import mlflow
import numpy as np
from flwr.common import (
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from rizemind.logging.base_metrics_storage import BaseMetricsStorage


class MLFLowMetricStorage(BaseMetricsStorage):
    def __init__(self, experiment_name: str, run_name: str, mlflow_uri: str):
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
        for k, v in metrics.items():
            self.mlflow_client.log_metric(
                run_id=self.run_id, key=k, value=float(v), step=server_round
            )

    def update_current_round_model(self, parameters: Parameters):
        self._current_round_model = parameters

    def update_best_model(self, server_round: int, loss: float):
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
