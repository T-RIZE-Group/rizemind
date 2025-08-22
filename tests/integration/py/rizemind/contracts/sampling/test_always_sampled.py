import pytest
from rizemind.contracts.deployment import DeployedContract
from rizemind.contracts.erc.erc5267.erc5267 import ERC5267
from rizemind.contracts.local_deployment import load_forge_artifact
from rizemind.contracts.sampling.always_sampled import AlwaysSamplesSelectorConfig
from rizemind.contracts.sampling.selector_factory import SelectorFactoryContract

from tests.integration.forge_fixtures import AnvilContext
from tests.integration.forge_helper import run_script

type SelectorFactoryDeployment = tuple[DeployedContract, DeployedContract]


@pytest.fixture(scope="module")
def deployment(anvil: AnvilContext) -> SelectorFactoryDeployment:
    """Deploy SelectorFactory and AlwaysSampled contracts."""
    deployer = anvil.account_conf.get_account(0)

    # Deploy SelectorFactory first
    selector_factory_artifact = run_script(
        "script/deployments/selectors/SelectorFactory.s.sol",
        account=deployer.address,
        env={"SELECTOR_FACTORY_OWNER": deployer.address},
    )
    selector_factory_contract = load_forge_artifact(
        selector_factory_artifact, "SelectorFactory"
    )

    # Deploy AlwaysSampled and register it with the factory
    always_sampled_artifact = run_script(
        "script/deployments/selectors/AlwaysSampled.s.sol",
        account=deployer.address,
    )
    always_sampled_contract = load_forge_artifact(
        always_sampled_artifact, "AlwaysSampled"
    )

    return selector_factory_contract, always_sampled_contract


def test_always_sampled_selector_id_matches_onchain(
    deployment: SelectorFactoryDeployment, anvil: AnvilContext
):
    selector_factory_deployment, always_sampled_deployment = deployment
    w3 = anvil.w3conf.get_web3()
    selector_factory_contract = SelectorFactoryContract.from_address(
        address=selector_factory_deployment.address, w3=w3
    )
    always_sampled_contract = ERC5267.from_address(
        address=always_sampled_deployment.address, w3=w3
    )
    onchain_version = always_sampled_contract.get_eip712_domain().version
    selector_factory_id = selector_factory_contract.get_id(onchain_version)

    always_sampled_config = AlwaysSamplesSelectorConfig()
    always_sampled_id = always_sampled_config.get_selector_id()
    assert selector_factory_id == always_sampled_id, "lib ID doesn't match onchain ID"


def test_always_sampled_selector_can_be_created(
    deployment: SelectorFactoryDeployment, anvil: AnvilContext
):
    selector_factory_deployment, _ = deployment
    w3 = anvil.w3conf.get_web3()
    selector_factory_contract = SelectorFactoryContract.from_address(
        address=selector_factory_deployment.address,
        w3=w3,
        account=anvil.account_conf.get_account(0),
    )
    always_sampled_config = AlwaysSamplesSelectorConfig()
    always_sampled_id = always_sampled_config.get_selector_id()
    always_sampled_init_data = always_sampled_config.get_init_data()
    salt = w3.keccak(text="always-sampled")
    selector_factory_contract.create_selector(
        always_sampled_id, salt, always_sampled_init_data
    )
