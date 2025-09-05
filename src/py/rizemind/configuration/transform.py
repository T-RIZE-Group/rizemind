from typing import Any, cast

from flwr.common import Properties
from flwr.common.record.configrecord import ConfigRecord
from flwr.common.typing import Config, ConfigRecordValues, Scalar


def normalize(data: dict[str, Any]) -> dict[str, ConfigRecordValues]:
    def convert(value: Any) -> ConfigRecordValues:
        # Scalars
        if isinstance(value, int | float | str | bytes | bool):
            return value

        # Lists of valid scalars
        if isinstance(value, list):
            if all(isinstance(i, int | float) for i in value):
                return cast(ConfigRecordValues, value)
            if all(isinstance(i, str | bytes | bool) for i in value):
                return cast(ConfigRecordValues, value)
            # Mixed or unsupported list contents
            raise TypeError(f"Unsupported list element types in value: {value}")

        if isinstance(value, dict):
            return normalize(value)  # type: ignore

        return str(value)

    return {k: convert(v) for k, v in data.items() if v is not None}


def to_config_record(d: dict[str, Any]) -> ConfigRecord:
    return ConfigRecord(cast(dict[str, ConfigRecordValues], flatten(normalize(d))))


def flatten(d: dict[str, Any], prefix: str = "") -> Config:
    flattened: Config = {}
    for key, value in d.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened = flattened | flatten(value, f"{new_key}")
        else:
            flattened[new_key] = value
    return flattened


def to_config(d: dict[str, Any], prefix: str = "") -> Config:
    normalized = normalize(d)
    return flatten(normalized, prefix)


def to_properties(d: dict[str, Any], prefix: str = "") -> Properties:
    normalized = normalize(d)
    return flatten(normalized, prefix)


def unflatten(flat_dict: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for compound_key, value in flat_dict.items():
        keys = compound_key.split(".")
        d = result
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return result


def from_config(config: Config) -> dict[str, Any]:
    return unflatten(config)


def from_properties(properties: Properties) -> dict[str, Any]:
    return unflatten(properties)


def _to_plain_dict(conf: Config | dict[str, Any] | ConfigRecord) -> dict[str, Any]:
    """Return a regular ``dict[str, Any]`` from any supported config-like object."""
    if isinstance(conf, ConfigRecord):
        return dict(conf)
    return cast(dict[str, Any], conf)


def concat(
    conf_a: Config | dict[str, Scalar] | ConfigRecord,
    conf_b: Config | dict[str, Scalar] | ConfigRecord,
) -> dict[str, Scalar]:
    """
    Merge two configuration objects into a flattened Flower ``ConfigRecord``.

    """
    dict_a = _to_plain_dict(conf_a)
    dict_b = _to_plain_dict(conf_b)
    return dict_a | dict_b
