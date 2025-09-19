from typing import Any

from flwr.common import Context

from rizemind.configuration.base_config import BaseConfig
from rizemind.configuration.transform import unflatten

MLFLOW_CONFIG_KEY = "rizemind.mlflow.config"


class MLFlowConfig(BaseConfig):
    """A data class for holding MLflow configuration parameters.

    This class provides a structured way to manage and access MLflow-specific
    settings, such as the experiment name, run name, and tracking URI. It
    inherits from a base configuration class and includes a factory method to
    conveniently load the configuration from a Flower context.

    Attributes:
        experiment_name: The name of the MLflow experiment to use for logging.
        run_name: The name to assign to the MLflow run.
        mlflow_uri: The URI of the MLflow tracking server.
    """

    experiment_name: str
    run_name: str
    mlflow_uri: str

    @staticmethod
    def from_context(ctx: Context) -> "MLFlowConfig | None":
        """Loads MLflow configuration from the context.

        This static method acts as a factory to create an `MLFlowConfig` instance
        by extracting records from the provided `Context` object. It looks for a
        specific key (`MLFLOW_CONFIG_KEY`) within the context's state.

        Args:
            ctx: The Flower context object which may contain the
                MLflow configuration records.

        Returns:
            MLFlowConfig | None: An instance of `MLFlowConfig` if the configuration
            is found in the context, otherwise `None`.
        """
        if MLFLOW_CONFIG_KEY in ctx.state.config_records:
            records: Any = ctx.state.config_records[MLFLOW_CONFIG_KEY]
            return MLFlowConfig(**unflatten(records))
        return None
