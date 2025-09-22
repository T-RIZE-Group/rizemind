import copy
import os
from functools import reduce
from pathlib import Path
from typing import Any, cast

import tomli


def replace_env_vars(obj: dict[str, Any] | str) -> dict[str, Any] | str:
    if isinstance(obj, str):
        # Replace placeholders with environment variable values
        return os.path.expandvars(obj)
    elif isinstance(obj, dict):
        return {key: replace_env_vars(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [replace_env_vars(item) for item in obj]
    return obj


def _safe_copy(value: Any) -> Any:
    """
    Return a deep copy of mutable objects, or the original value for immutable objects.

    :param value: The value to potentially copy
    :return: A copy of mutable objects, or the original value for immutable objects
    """
    if isinstance(value, dict | list):
        return copy.deepcopy(value)
    return value


class TomlConfig:
    """
    A class to load and manage configuration from a TOML file.
    It parses string in configurations to replaces them with env variables.

    **Example Usage:**

    .. code-block:: python

        config = TomlConfig("config.toml")
        database_host = config.get("database.host", "localhost")
        api_key = config.get(["api", "key"], "default_key")

    **Example TOML Configuration:**

    .. code-block:: toml

        [database]
        host = "127.0.0.1"
        port = 5432

        [api]
        key = "$ENV_API_KEY"

    :param path: Path to the TOML configuration file.
    :type path: str
    """

    def __init__(self, path: str | Path):
        """
        Initialize the TomlConfig instance.

        :param path: Path to the TOML configuration file.
        :type path: str
        """
        self.path = Path(path)
        self._validate_path()
        self._data = self._load_toml()

    def _validate_path(self):
        """Ensure the file exists and is readable."""
        if not self.path.exists():
            raise FileNotFoundError(f"Config file not found: {self.path}")
        if not self.path.is_file():
            raise ValueError(f"Provided path is not a file: {self.path}")

    def _load_toml(self) -> dict:
        """Load and return the TOML file as a dictionary."""
        with self.path.open("rb") as f:
            return cast(dict, replace_env_vars(tomli.load(f)))

    @property
    def data(self) -> dict:
        """
        Return a deep copy of the loaded TOML dictionary.

        :return: A deep copy of the entire TOML configuration as a dictionary.
        :rtype: dict
        """
        return _safe_copy(self._data)

    def get(self, keys: list[str] | str, default: Any | None = None) -> Any:
        """
        Retrieve a nested value from the config safely.
        Returns a deep copy of mutable objects (dicts, lists) to prevent external modification.

        :param keys: list of keys representing the path to the value.
        :param default: Default value to return if the key path does not exist.
        :return: A copy of the value at the given key path or the default value.
        """
        if isinstance(keys, str):
            keys = keys.split(".")

        result = reduce(
            lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
            keys,
            self._data,
        )

        return _safe_copy(result)
