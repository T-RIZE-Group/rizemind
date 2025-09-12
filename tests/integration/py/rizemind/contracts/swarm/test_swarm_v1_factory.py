import pytest
from rizemind.contracts.access_control.fl_access_control.FlAccessControl import (
    FlAccessControl,
)
from rizemind.contracts.swarm.swarm_v1.swarm_v1_factory import (
    SwarmV1Factory,
    SwarmV1FactoryConfig,
)

from tests.integration.forge_fixtures import AnvilContext
from tests.integration.forge_helper import run_script

type Deployment = tuple[FlAccessControl, str, list[str]]


@pytest.fixture(scope="module")
def factory_config(anvil: AnvilContext):
    deployer = anvil.account_conf.get_account(0)
    run_script(
        "script/deployments/selectors/SelectorFactory.s.sol",
        account=deployer.address,
        env={"SELECTOR_FACTORY_OWNER": deployer.address},
    )
    run_script(
        "script/deployments/selectors/AlwaysSampled.s.sol",
        account=deployer.address,
    )
    artifact_path = run_script(
        "script/integrations/Deploy.s.sol",
        account=deployer.address,
        env={"OWNER": deployer.address},
    )
    factory_config = SwarmV1FactoryConfig(
        name="test", local_factory_deployment_path=str(artifact_path)
    )

    return factory_config


def test_deploy(factory_config: SwarmV1FactoryConfig, anvil: AnvilContext):
    factory = SwarmV1Factory(factory_config)
    deployer = anvil.account_conf.get_account(1)
    swarm = factory.deploy(
        deployer, [anvil.account_conf.get_account(2).address], anvil.w3conf.get_web3()
    )
    assert swarm.address is not None
