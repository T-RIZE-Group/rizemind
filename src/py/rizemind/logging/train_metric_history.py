from typing import Literal

from flwr.common import Scalar
from pydantic import BaseModel

TRAIN_METRIC_HISTORY_KEY = "rizemind.logging.train_metric_history"


class TrainMetricHistory(BaseModel):
    """Standard class to log training metrics for clients.

    The `TrainMetricHistory` class is used to log metrics that are gathered during the training phase.
    It is the standard class that is used in mods that perform logging.
    It has the capability to distinguish metrics for evaluation and training (during training phase).
    """

    _history: dict[str, list[float]]

    def __init__(self, history: dict[str, list[float]] = {}):
        super().__init__(history=history)

    def append(self, metrics: dict[str, Scalar], is_eval: bool):
        """Append a dictionary of metrics to the history.

        Adds a suffix '_eval' or '_train' to the metric name based on the is_eval flag.

        Args:
            metrics: A dictionary mapping metric names (str) to their scalar values.
            is_eval: Indicates if the metrics are from an evaluation phase.
        """
        phase: Literal["eval", "train"] = "eval" if is_eval else "train"

        for k, v in metrics.items():
            metric = f"{k}_{phase}"
            if metric not in self._history:
                self._history[metric] = []
            self._history[metric].append(float(v))

    def items(self):
        """Return an iterable of the history's items (key-value pairs).

        The `items` is used as the equivalent for `dict.items`.

        Returns:
            A view object displaying a list of the dictionary's key-value tuple pairs.
        """
        return self._history.items()

    def serialize(self) -> dict[str, str]:
        """Serialize the TrainMetricHistory instance into its json representation.

        The instance is converted to a JSON string and stored under a predefined key.
        This serialization ensures `TrainMetricHistory` compatibility with `metrics`
        as a `[str, Scalar]` type.

        Returns:
            A dictionary containing the serialized TrainMetricHistory instance as a JSON string.
        """
        return {TRAIN_METRIC_HISTORY_KEY: self.model_dump_json()}

    @classmethod
    def deserialize(cls, serialized_train_metric_history: str) -> "TrainMetricHistory":
        """Deserialize a JSON string into a TrainMetricHistory instance.

        Args:
            serialized_train_metric_history: The JSON string representation of a TrainMetricHistory.

        Returns:
            A new instance of the TrainMetricHistory class.
        """
        return TrainMetricHistory.model_validate_json(serialized_train_metric_history)
