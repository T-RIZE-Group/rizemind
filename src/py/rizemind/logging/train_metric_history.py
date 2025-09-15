from typing import Literal

from flwr.common import Scalar
from pydantic import BaseModel

TRAIN_METRIC_HISTORY_KEY = "rizemind.logging.train_metric_history"


class TrainMetricHistory(BaseModel):
    history: dict[str, list[float]]

    def __init__(self, history: dict[str, list[float]] = {}):
        super().__init__(history=history)

    def append(self, metrics: dict[str, Scalar], is_eval: bool):
        phase: Literal["eval", "train"] = "eval" if is_eval else "train"

        for k, v in metrics.items():
            metric = f"{k}_{phase}"
            if metric not in self.history:
                self.history[metric] = []
            self.history[metric].append(float(v))

    def items(self):
        return self.history.items()

    def serialize(self) -> dict[str, str]:
        return {TRAIN_METRIC_HISTORY_KEY: self.model_dump_json()}

    @classmethod
    def deserialize(cls, serialized_train_metric_history: str) -> "TrainMetricHistory":
        return TrainMetricHistory.model_validate_json(serialized_train_metric_history)
