import tomli
from pathlib import Path
from typing import Any, List, Optional, Union
from functools import reduce

class TomlConfig:
    def __init__(self, path: str):
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
            return tomli.load(f)

    @property
    def data(self) -> dict:
        """Return the loaded TOML dictionary."""
        return self._data

    def get(self, keys: Union[List[str], str], default: Optional[Any] = None) -> Any:
      """
      Retrieve a nested value from the config safely.

      :param keys: List of keys representing the path to the value.
      :param default: Default value to return if the key path does not exist.
      :return: The value at the given key path or the default value.
      """
      if isinstance(keys, str):
        keys = keys.split(".")
      return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys, self._data)
