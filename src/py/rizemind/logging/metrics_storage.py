import csv
import json
import os
from datetime import datetime
from logging import ERROR
from pathlib import Path

from flwr.common import Scalar, log
from flwr.common.typing import UserConfigValue


class MetricsStorage:
    def __init__(self, dir: Path, app_name: str) -> None:
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%m-%S")
        self.dir = dir.joinpath(app_name, current_time)
        self.dir.mkdir(parents=True, exist_ok=True)

        self.config_file = self.dir.joinpath("config.json")
        self.metrics_file = self.dir.joinpath("metrics.csv")

        headers = ["round", "metric", "value"]

        with open(self.metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def write_config(self, config: dict[str, UserConfigValue]):
        file_exists_and_has_content = (
            os.path.exists(self.config_file) and os.path.getsize(self.config_file) > 0
        )
        if file_exists_and_has_content is True:
            with open(self.config_file) as f:
                existing_config: dict = json.load(f)
                for key, value in existing_config.items():
                    config[key] = value
        try:
            with open(self.config_file, mode="w") as f:
                json.dump(config, f)
        except OSError as e:
            log(ERROR, f"Error writing config JSON to {self.config_file}: {e}")

    def write_metrics(self, round: int, metrics: dict[str, Scalar]):
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            for metric, value in metrics.items():
                csv.writer(f).writerow([round, metric, value])
