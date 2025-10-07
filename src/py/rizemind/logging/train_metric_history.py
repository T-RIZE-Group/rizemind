from typing import Literal

from flwr.common import Metrics, Scalar
from pydantic import BaseModel

TRAIN_METRIC_HISTORY_KEY = "rizemind.logging.train_metric_history"


class TrainMetricHistory(BaseModel):
    """Standard class to log training metrics for clients.

    The `TrainMetricHistory` class is used to log metrics that are gathered during the training phase.
    It is the standard class that is used in mods that perform logging.
    It has the capability to distinguish metrics for evaluation and training (during training phase).
    """

    history: dict[str, list[float]]

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
            if metric not in self.history:
                self.history[metric] = []
            self.history[metric].append(float(v))

    def items(self):
        """Return an iterable of the history's items (key-value pairs).

        The `items` is used as the equivalent for `dict.items`.

        Returns:
            A view object displaying a list of the dictionary's key-value tuple pairs.
        """
        return self.history.items()

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


def fit_metric_history_aggregation_fn(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Returns the serialized FitMetricHistories for each single client

    Typical usage:
    >>> fedavg_strategy = FedAvg(
    >>>    fraction_fit=float(context.run_config["fraction-fit"]),
    >>>    fraction_evaluate=float(context.run_config["fraction-evaluate"]),
    >>>    min_available_clients=int(context.run_config["min-available-clients"]),
    >>>    min_evaluate_clients=int(context.run_config["min-evaluate-clients"]),
    >>>    initial_parameters=parameters,
    >>>    fit_metrics_aggregation_fn=fit_metric_history_aggregation_fn,
    >>> )
    """
    serialized_metrics: dict[str, Scalar] = {}
    for _, client_metrics_dict in metrics:
        for key, value in client_metrics_dict.items():
            if key.startswith(TRAIN_METRIC_HISTORY_KEY):
                serialized_metrics[key] = value
    return serialized_metrics
