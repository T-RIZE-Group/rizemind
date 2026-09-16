import pytest
from rizemind.contracts.swarm.swarm_v1.swarm_v1_factory import SwarmV1FactoryConfig
from rizemind.web3.chains import ARC_MAINNET_CHAINID, RIZENET_TESTNET_CHAINID
from web3 import Web3

ARC_MAINNET_FACTORY = "0x721a4ebA8747eF5db299E8Eec67A2FbAe7866353"
RIZENET_TESTNET_FACTORY = "0xd66C7C89Fb97eA5c06b0b7CaF2086dF1E82b9E88"


@pytest.mark.parametrize(
    "chain_id, expected",
    [
        (ARC_MAINNET_CHAINID, ARC_MAINNET_FACTORY),
        (RIZENET_TESTNET_CHAINID, RIZENET_TESTNET_FACTORY),
    ],
)
def test_known_chains_resolve_to_their_factory(chain_id, expected):
    config = SwarmV1FactoryConfig(name="test_model")

    assert config.get_factory_deployment(chain_id).address == expected


def test_registered_addresses_are_checksummed():
    config = SwarmV1FactoryConfig(name="test_model")

    for deployment in config.factory_deployments.values():
        assert Web3.is_checksum_address(deployment.address)


def test_unknown_chain_raises():
    config = SwarmV1FactoryConfig(name="test_model")

    with pytest.raises(Exception, match="Chain ID#1 is unsupported"):
        config.get_factory_deployment(1)


def test_toml_style_override_replaces_the_default_map():
    """A `factory_deployments` override coerces TOML's string keys to ints."""
    config = SwarmV1FactoryConfig(
        name="test_model",
        factory_deployments={"5042": {"address": ARC_MAINNET_FACTORY}},
    )

    assert config.get_factory_deployment(ARC_MAINNET_CHAINID).address == (
        ARC_MAINNET_FACTORY
    )
    with pytest.raises(Exception, match="unsupported"):
        config.get_factory_deployment(RIZENET_TESTNET_CHAINID)
