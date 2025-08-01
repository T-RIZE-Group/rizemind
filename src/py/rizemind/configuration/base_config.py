from flwr.common.record.configrecord import ConfigRecord
from pydantic import BaseModel

from rizemind.configuration.transform import to_config_record


class BaseConfig(BaseModel):
    def to_config_record(self) -> ConfigRecord:
        return to_config_record(self.model_dump())
