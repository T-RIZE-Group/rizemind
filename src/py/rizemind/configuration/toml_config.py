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


class TomlConfig:
    """Load and manage configuration from a TOML file.

    On initialization, the file is parsed and all string values containing
    environment variable placeholders are expanded.

    Attributes:
        path: Path to the TOML configuration file.

    Examples:
        >>> # Example TOML Configuration File:
        >>> # [database]
        >>> # host = "127.0.0.1"
        >>> # port = 5432
        >>> # [api]
        >>> # key = "$ENV_API_KEY"
        >>> config = TomlConfig("config.toml")
        >>> database_host = config.get("database.host", "localhost")
        >>> api_key = config.get(["api", "key"], "default_key")
    """

    def __init__(self, path: str | Path):
        """Initialize a configuration instance for the given TOML file.

        Args:
            path: Path to the TOML configuration file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the path is not a regular file.
        """
        self.path = Path(path)
        self._validate_path()
        self._data = self._load_toml()

    def _validate_path(self):
        """Validate that the configuration path exists and is a file.

        Raises:
            FileNotFoundError: If the config file was not found.
            ValueError: If the provided path is not a file.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Config file not found: {self.path}")
        if not self.path.is_file():
            raise ValueError(f"Provided path is not a file: {self.path}")

    def _load_toml(self) -> dict:
        """Load the TOML file and expand environment variables in strings.

        Returns:
            A dictionary containing the parsed configuration.
        """
        with self.path.open("rb") as f:
            return cast(dict, replace_env_vars(tomli.load(f)))

    @property
    def data(self) -> dict:
        """The loaded configuration data as a dictionary."""
        return self._data

    def get(self, keys: list[str] | str, default: Any | None = None) -> Any:
        """Retrieve a nested configuration value.

        Args:
            keys: A dot-delimited string path (for example, ``"a.b.c"``) or a
            list of path segments (for example, ``["a", "b", "c"]``).
            default: Value to return if the path does not exist.

        Returns:
            The value at the given key path, or `default` if the path is not
            present.
        """
        if isinstance(keys, str):
            keys = keys.split(".")
        return reduce(
            lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
            keys,
            self._data,
        )
