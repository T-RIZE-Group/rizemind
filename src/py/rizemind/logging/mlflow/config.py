from typing import Any

from flwr.common import Context
from rizemind.configuration.base_config import BaseConfig
from rizemind.configuration.transform import unflatten

MLFLOW_CONFIG_KEY = "rizemind.mlflow.config"


class MLFlowConfig(BaseConfig):
    experiment_name: str
    run_name: str
    mlflow_uri: str

    @staticmethod
    def from_context(ctx: Context) -> "MLFlowConfig | None":
        if MLFLOW_CONFIG_KEY in ctx.state.config_records:
            records: Any = ctx.state.config_records[MLFLOW_CONFIG_KEY]
            return MLFlowConfig(**unflatten(records))
        return None
