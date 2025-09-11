from typing import Literal

from flwr.common import Scalar
from pydantic import BaseModel


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
