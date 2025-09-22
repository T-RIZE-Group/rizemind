import os
import tempfile
from pathlib import Path

from rizemind.configuration.toml_config import TomlConfig

toml_content_with_env_var = """
[tool.web3.account]
mnemonic = "test test test test test test test test test test test junk"

env_var = "$TEST"
"""


def test_toml_config_parses_env_vars():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content_with_env_var)
        temp_path = f.name
    try:
        os.environ["TEST"] = "hello world"
        conf = TomlConfig(temp_path)
        assert (
            conf.get(["tool", "web3", "account", "mnemonic"])
            == "test test test test test test test test test test test junk"
        )
        assert (
            conf.get("tool.web3.account.mnemonic")
            == "test test test test test test test test test test test junk"
        )
        assert conf.get("tool.web3.account.env_var") == os.environ["TEST"]
    finally:
        Path(temp_path).unlink()


toml_content = """
[database]
host = "localhost"
port = 5432
users = ["admin", "user"]

[api]
key = "secret"
settings = {timeout = 30, retries = 3}
"""


def test_data_property_returns_copy():
    """Test that the data property returns a copy, not a reference."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        temp_path = f.name

    try:
        config = TomlConfig(temp_path)

        # Get the data
        data1 = config.data
        data2 = config.data

        # Modify data1
        data1["database"]["host"] = "modified"
        data1["database"]["users"].append("hacker")

        # Check that data2 is unchanged (proving they are different objects)
        assert data2["database"]["host"] == "localhost", (
            "data property should return copies"
        )
        assert len(data2["database"]["users"]) == 2, (
            "data property should return copies"
        )
        assert "hacker" not in data2["database"]["users"], (
            "data property should return copies"
        )

    finally:
        Path(temp_path).unlink()


def test_get_method_returns_copy():
    """Test that the get method returns copies for mutable objects."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        temp_path = f.name

    try:
        config = TomlConfig(temp_path)

        # Get mutable objects
        users1 = config.get("database.users")
        users2 = config.get("database.users")
        settings1 = config.get("api.settings")
        settings2 = config.get("api.settings")

        # Modify the first copies
        users1.append("hacker")
        settings1["timeout"] = 999

        # Check that the second copies are unchanged
        assert len(users2) == 2, "get method should return copies of lists"
        assert "hacker" not in users2, "get method should return copies of lists"
        assert settings2["timeout"] == 30, "get method should return copies of dicts"

        # Test that immutable values are not copied unnecessarily
        host1 = config.get("database.host")
        host2 = config.get("database.host")
        assert host1 is host2, "immutable values should not be copied"

    finally:
        Path(temp_path).unlink()
