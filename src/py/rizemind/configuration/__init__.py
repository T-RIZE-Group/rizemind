"""Configuration management utilities for Rizemind.

This module provides tools for loading, transforming, and managing configuration
data for Rizemind applications. It includes support for TOML configuration files,
Pydantic-based configuration models, and integration with Flower's configuration
system through data transformation utilities.

The module offers two main configuration approaches:

- `TomlConfig`: For loading configuration from TOML files with environment
  variable substitution
- `BaseConfig`: A Pydantic-based model for structured configuration with
  validation and type safety

Additionally, it provides a comprehensive set of transformation functions to
convert between different configuration formats used by the Flower federated
learning framework.

Typical usage example:
    >>> # Load configuration from TOML file
    >>> config = TomlConfig("config.toml")
    >>> database_host = config.get("database.host", "localhost")
    >>> # Use Pydantic-based configuration
    >>> class MyConfig(BaseConfig):
    >>>   host: str = "localhost"
    >>>   port: int = 8080
    >>> my_config = MyConfig(host="example.com")
    >>> flower_config = my_config.to_config_record()

    >>> # Transform configuration data
    >>> nested_config = {"db": {"host": "localhost", "port": 5432}}
    >>> flat_config = flatten(
    ...     nested_config
    ... )  # {"db.host": "localhost", "db.port": 5432}
"""

from rizemind.configuration.base_config import BaseConfig
from rizemind.configuration.toml_config import TomlConfig
from rizemind.configuration.transform import (
    concat,
    flatten,
    from_config,
    from_properties,
    normalize,
    to_config,
    to_config_record,
    to_properties,
    unflatten,
)

__all__ = [
    "BaseConfig",
    "TomlConfig",
    "normalize",
    "to_config_record",
    "flatten",
    "to_config",
    "to_properties",
    "unflatten",
    "from_config",
    "from_properties",
    "concat",
]
from hexbytes import HexBytes
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


def _hexbytes_schema(cls, source, handler: GetCoreSchemaHandler):
    def _validate(v):
        if isinstance(v, HexBytes):
            return v

        raise TypeError(f"Cannot parse {type(v)} as HexBytes")

    # validate to HexBytes in Python; serialize to a hex string in JSON
    return core_schema.no_info_plain_validator_function(
        _validate,
        serialization=core_schema.plain_serializer_function_ser_schema(
            lambda v: v,
            return_schema=core_schema.bytes_schema(),
            when_used="json",
        ),
    )


# apply once at import time
HexBytes.__get_pydantic_core_schema__ = classmethod(_hexbytes_schema)  # pyright: ignore[reportAttributeAccessIssue]
