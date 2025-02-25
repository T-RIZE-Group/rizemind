import os
from rizemind.configuration.toml_config import TomlConfig


def test_toml_config():
    os.environ["TEST"] = "hello world"
    conf = TomlConfig("./src/py/rizemind/configuration/sample_config.toml")
    assert (
        conf.get(["tool", "web3", "account", "mnemonic"])
        == "test test test test test test test test test test test junk"
    )
    assert (
        conf.get("tool.web3.account.mnemonic")
        == "test test test test test test test test test test test junk"
    )
    assert conf.get("tool.web3.account.env_var") == os.environ["TEST"]
