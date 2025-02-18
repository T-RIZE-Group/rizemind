from rize_dml.configuration.toml_config import TomlConfig


def test_toml_config():
    conf = TomlConfig("./src/py/rize_dml/configuration/sample_config.toml")
    assert (
        conf.get(["tool", "web3", "account", "mnemonic"])
        == "test test test test test test test test test test test junk"
    )
    assert (
        conf.get("tool.web3.account.mnemonic")
        == "test test test test test test test test test test test junk"
    )
