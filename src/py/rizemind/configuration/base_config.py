from flwr.common.record.configrecord import ConfigRecord
from pydantic import BaseModel


class BaseConfig(BaseModel):
    def to_config_record(self) -> ConfigRecord:
        config_dict = {k: v for k, v in self.model_dump().items() if v is not None}
        return ConfigRecord(config_dict)
