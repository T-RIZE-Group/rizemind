from typing import Any, cast

from flwr.common import Properties
from flwr.common.record.configrecord import ConfigRecord
from flwr.common.typing import Config, ConfigRecordValues, Scalar


def normalize(data: dict[str, Any]) -> dict[str, ConfigRecordValues]:
    """Normalize nested values to the limited set supported by Flower.

    Converts values to scalars (int, float, str, bytes, bool), lists of
    scalars, or nested dicts. Values equal to `None` are dropped.
    Unsupported list element mixes raise a `TypeError`. Non-list, non-dict
    values outside the supported set are converted to their string
    representation.

    Args:
        data: Arbitrary nested mapping to normalize.

    Returns:
        A new mapping with values restricted to Flower-compatible scalar types,
        lists of scalars, or nested dicts.

    Raises:
        TypeError: If a list contains mixed or unsupported element types.
    """

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
    """Build a Flower `ConfigRecord` from a nested mapping.

    Normalizes and flattens the mapping before constructing the record.

    Args:
        d: Nested mapping containing configuration values.

    Returns:
        A `ConfigRecord` with dot-delimited keys and normalized values.
    """
    return ConfigRecord(cast(dict[str, ConfigRecordValues], flatten(normalize(d))))


def flatten(d: dict[str, Any], prefix: str = "") -> Config:
    """Flatten a nested mapping using dot-delimited keys.

    Each nested dictionary path is joined with `.` to form a single key. When
    a prefix is provided, it is prepended to each generated key.

    Args:
        d: Nested mapping to flatten.
        prefix: Optional namespace to prepend to generated keys.

    Returns:
        A flat mapping suitable for Flower `Config`/`Properties`.
    """
    flattened: Config = {}
    for key, value in d.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened = flattened | flatten(value, f"{new_key}")
        else:
            flattened[new_key] = value
    return flattened


def to_config(d: dict[str, Any], prefix: str = "") -> Config:
    """Return a flat Flower `Config` from a nested mapping.

    Normalizes values and flattens keys. If `prefix` is provided, keys are
    namespaced under the given prefix.

    Args:
        d: Nested mapping to normalize and flatten.
        prefix: Optional namespace to prepend.

    Returns:
        A Flower `Config` mapping.
    """
    normalized = normalize(d)
    return flatten(normalized, prefix)


def to_properties(d: dict[str, Any], prefix: str = "") -> Properties:
    """Return a flat Flower `Properties` mapping from a nested mapping.

    Normalizes values and flattens keys. If `prefix` is provided, keys are
    namespaced under the given prefix.

    Args:
        d: Nested mapping to normalize and flatten.
        prefix: Optional namespace to prepend.

    Returns:
        A Flower `Properties` mapping.
    """
    normalized = normalize(d)
    return flatten(normalized, prefix)


def unflatten(flat_dict: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct a nested mapping from dot-delimited keys.

    Args:
        flat_dict: Mapping with keys separated by `.`.

    Returns:
        A nested mapping rebuilt from the flat representation.
    """
    result = {}
    for compound_key, value in flat_dict.items():
        keys = compound_key.split(".")
        d = result
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return result


def from_config(config: Config) -> dict[str, Any]:
    """Convert a Flower ``Config`` into a nested mapping.

    Args:
        config: Flat mapping produced by `flatten`/`to_config`.

    Returns:
        A nested mapping.
    """
    return unflatten(config)


def from_properties(properties: Properties) -> dict[str, Any]:
    """Convert Flower ``Properties`` into a nested mapping.

    Args:
        properties: Flat mapping produced by `flatten`/`to_properties`.

    Returns:
        A nested mapping.
    """
    return unflatten(properties)


def _to_plain_dict(conf: Config | dict[str, Any] | ConfigRecord) -> dict[str, Any]:
    """Return a regular dict from any supported config-like object.

    Args:
        conf: A `Config`, `ConfigRecord`, or plain dict.

    Returns:
        A new dict containing the same key/value pairs.
    """
    if isinstance(conf, ConfigRecord):
        return dict(conf)
    return cast(dict[str, Any], conf)


def concat(
    conf_a: Config | dict[str, Scalar] | ConfigRecord,
    conf_b: Config | dict[str, Scalar] | ConfigRecord,
) -> dict[str, Scalar]:
    """Merge two configuration objects into a flat dictionary.

    Both inputs may be Flower configs, plain dicts, or `ConfigRecord`s. On
    key conflicts, values from `conf_b` take precedence.

    Args:
        conf_a: First configuration source.
        conf_b: Second configuration source; overrides on conflicts.

    Returns:
        A new flat dictionary with the merged contents.
    """
    dict_a = _to_plain_dict(conf_a)
    dict_b = _to_plain_dict(conf_b)
    return dict_a | dict_b
