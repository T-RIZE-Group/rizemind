from flwr.common import Context
from flwr.common.record.configrecord import ConfigRecord
from pydantic import BaseModel

from rizemind.configuration.transform import flatten, to_config_record


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
        return to_config_record(flatten(self.model_dump()))

    def _store_in_context(self, context: Context, state_key: str) -> None:
        """Store the configuration in the context.

        Args:
            context: The context to store the configuration in.
            state_key: The key to store the configuration in.
        """
        context.state.config_records[state_key] = self.to_config_record()
