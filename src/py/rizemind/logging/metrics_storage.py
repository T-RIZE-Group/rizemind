import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from flwr.common import Parameters, Scalar, parameters_to_ndarrays
from flwr.common.typing import UserConfigValue
from rizemind.logging.base_metrics_storage import BaseMetricsStorage


class MetricsStorage(BaseMetricsStorage):
    def __init__(self, dir: Path, app_name: str) -> None:
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
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            for metric, value in metrics.items():
                csv.writer(f).writerow([server_round, metric, value])

    def update_current_round_model(self, parameters: Parameters):
        self._current_round_model = parameters

    def update_best_model(self, server_round: int, loss: float):
        if loss < self._best_loss:
            self._best_loss = loss
            ndarray_params = parameters_to_ndarrays(self._current_round_model)
            np.savez(self.weights_file, ndarray_params)
