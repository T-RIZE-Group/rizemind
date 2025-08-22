import pytest
from rizemind.contracts.deployment import DeployedContract
from rizemind.contracts.erc.erc5267.erc5267 import ERC5267
from rizemind.contracts.local_deployment import load_forge_artifact
from rizemind.contracts.sampling.random_sampling import RandomSamplingSelectorConfig
from rizemind.contracts.sampling.selector_factory import SelectorFactoryContract

from tests.integration.forge_fixtures import AnvilContext
from tests.integration.forge_helper import run_script

type SelectorFactoryDeployment = tuple[DeployedContract, DeployedContract]


@pytest.fixture(scope="module")
def deployment(anvil: AnvilContext) -> SelectorFactoryDeployment:
    """Deploy SelectorFactory and RandomSampling contracts."""
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

    # Deploy RandomSampling and register it with the factory
    random_sampling_artifact = run_script(
        "script/deployments/selectors/RandomSampling.s.sol",
        account=deployer.address,
    )
    random_sampling_contract = load_forge_artifact(
        random_sampling_artifact, "RandomSampling"
    )

    return selector_factory_contract, random_sampling_contract


def test_random_sampling_selector_id_matches_onchain(
    deployment: SelectorFactoryDeployment, anvil: AnvilContext
):
    selector_factory_deployment, random_sampling_deployment = deployment
    w3 = anvil.w3conf.get_web3()
    selector_factory_contract = SelectorFactoryContract.from_address(
        address=selector_factory_deployment.address, w3=w3
    )
    random_sampling_contract = ERC5267.from_address(
        address=random_sampling_deployment.address, w3=w3
    )
    onchain_version = random_sampling_contract.get_eip712_domain().version
    selector_factory_id = selector_factory_contract.get_id(onchain_version)

    random_sampling_config = RandomSamplingSelectorConfig(ratio=0.5)
    random_sampling_id = random_sampling_config.get_selector_id()
    assert selector_factory_id == random_sampling_id, "lib ID doesn't match onchain ID"


def test_random_sampling_selector_can_be_created(
    deployment: SelectorFactoryDeployment, anvil: AnvilContext
):
    selector_factory_deployment, _ = deployment
    w3 = anvil.w3conf.get_web3()
    selector_factory_contract = SelectorFactoryContract.from_address(
        address=selector_factory_deployment.address,
        w3=w3,
        account=anvil.account_conf.get_account(0),
    )
    random_sampling_config = RandomSamplingSelectorConfig(ratio=0.5)
    random_sampling_id = random_sampling_config.get_selector_id()
    random_sampling_init_data = random_sampling_config.get_init_data()
    salt = w3.keccak(text="random-sampling")
    selector_factory_contract.create_selector(
        random_sampling_id, salt, random_sampling_init_data
    )
