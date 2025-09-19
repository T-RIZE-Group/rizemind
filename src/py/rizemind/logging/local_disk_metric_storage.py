import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from flwr.common import Parameters, Scalar, parameters_to_ndarrays
from flwr.common.typing import UserConfigValue

from rizemind.logging.base_metric_storage import BaseMetricStorage


class LocalDiskMetricStorage(BaseMetricStorage):
    """A concrete implementation of `BaseMetricStorage` that saves metrics and models to the local disk.

    This class provides a straightforward way to persist federated learning
    artifacts by writing them to a structured directory on the local filesystem.
    For each run, it creates a unique timestamped subdirectory within the specified
    application folder.

    The generated directory structure is as follows:
    - <dir>/
      - <app_name>/
        - <YYYY-MM-DD-HH-MM-SS>/
          - config.json  (Federated learning configuration settings)
          - metrics.csv  (Round-by-round metrics)
          - weights.npz  (Parameters of the best model)

    Attributes:
        dir: The root directory for storing all run artifacts.
        config_file: The path to the JSON file for storing configuration.
        metrics_file: The path to the CSV file for storing metrics.
        weights_file: The path to the .npz file for storing the best model's weights.
    """

    def __init__(self, dir: Path, app_name: str) -> None:
        """Initializes the LocalDiskMetricStorage and sets up the directory structure.

        This creates the necessary directories and files for the current run,
        including the main metrics CSV file with its headers.

        Args:
            dir (Path): The base directory where the application-specific folder
                will be created.
            app_name (str): The name of the application or experiment, used to
                create a subdirectory within `dir`.
        """
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%m-%S")
        self.dir = dir.joinpath(app_name, current_time)
        self.dir.mkdir(parents=True, exist_ok=True)

        self.config_file = self.dir.joinpath("config.json")
        self.metrics_file = self.dir.joinpath("metrics.csv")
        self.weights_file = self.dir.joinpath("weights.npz")

        headers = ["server_round", "metric", "value"]

        with open(self.metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

        self._best_loss = np.inf
        self._current_round_model = Parameters(tensors=[], tensor_type="")

    def write_config(self, config: dict[str, UserConfigValue]):
        """Writes configuration parameters to a JSON file.

        If the config file already exists and contains data, the new
        configuration is merged with the existing content.

        Args:
            config (dict[str, UserConfigValue]): A dictionary of configuration
                settings to be saved.
        """
        new_config_df = pl.from_dicts([config])

        file_exists_and_has_content = (
            os.path.exists(self.config_file) and os.path.getsize(self.config_file) > 0
        )
        if file_exists_and_has_content is True:
            config_df = pl.read_json(self.config_file)
            merged_df = pl.concat([config_df, new_config_df], how="diagonal_relaxed")
            new_config_df = merged_df
        new_config_df.write_json(self.config_file)

    def write_metrics(self, server_round: int, metrics: dict[str, Scalar]):
        """Appends a dictionary of metrics to the metrics.csv file.

        Each key-value pair in the metrics dictionary is written as a new row
        in the CSV file, along with the server round.

        Args:
            server_round (int): The current round of federated learning.
            metrics (dict[str, Scalar]): A dictionary mapping metric names
                (e.g., "accuracy") to their scalar values.
        """
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            for metric, value in metrics.items():
                csv.writer(f).writerow([server_round, metric, value])

    def update_current_round_model(self, parameters: Parameters):
        """Temporarily stores the model parameters for the current round in memory.

        This method holds the latest model parameters so they can be saved to
        disk later by `update_best_model` if this model proves to be the best one.

        Args:
            parameters (Parameters): The model parameters from the current round.
        """
        self._current_round_model = parameters

    def update_best_model(self, server_round: int, loss: float):
        """Saves the current model to disk if its loss is the lowest seen so far.

        It compares the provided loss with its internally tracked best loss.
        If the new loss is lower, it updates the best loss and serializes the
        in-memory model parameters (from `update_current_round_model`) to a
        'weights.npz' file, overwriting any previously saved best model.

        Args:
            server_round (int): The server round that produced this model.
                (Not used in this implementation but required by the interface).
            loss (float): The loss value of the current model.
        """
        if loss < self._best_loss:
            self._best_loss = loss
            ndarray_params = parameters_to_ndarrays(self._current_round_model)
            np.savez(self.weights_file, ndarray_params)
