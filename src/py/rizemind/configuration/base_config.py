from flwr.common.record.configrecord import ConfigRecord
from pydantic import BaseModel

from rizemind.configuration.transform import to_config_record


class BaseConfig(BaseModel):
    """Base configuration model with conversion to a Flower `ConfigRecord`.

    Extends pydantic's `BaseModel` and adds `to_config_record()` to integrate
    with Flower's configuration interface.
    """

    def to_config_record(self) -> ConfigRecord:
        """Convert this configuration into a Flower `ConfigRecord`.

        Returns:
            The Flower `ConfigRecord` representing this configuration.
        """
        return to_config_record(self.model_dump())
