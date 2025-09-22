from flwr.common import Context
from flwr.common.record.configrecord import ConfigRecord
from pydantic import BaseModel

from rizemind.configuration.transform import flatten, to_config_record


class BaseConfig(BaseModel):
    def to_config_record(self) -> ConfigRecord:
        return to_config_record(flatten(self.model_dump()))

    def _store_in_context(self, context: Context, state_key: str) -> None:
        context.state.config_records[state_key] = self.to_config_record()
