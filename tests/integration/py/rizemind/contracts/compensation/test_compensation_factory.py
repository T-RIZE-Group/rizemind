import pytest
from eth_account.signers.base import BaseAccount
from rizemind.contracts.compensation.compensation_factory import (
    CompensationFactoryContract,
)
from rizemind.contracts.compensation.simple_mint_compensation.simple_mint_compensation import (
    SimpleMintCompensation,
    SimpleMintCompensationConfig,
)
from rizemind.contracts.local_deployment import load_forge_artifact
from web3 import Web3

from tests.integration.forge_fixtures import AnvilContext
from tests.integration.forge_helper import run_script

type Deployment = tuple[CompensationFactoryContract, BaseAccount, Web3]


@pytest.fixture(scope="module")
def deploy_compensation_factory(anvil: AnvilContext):
    """Deploy a CompensationFactory instance."""
    admin = anvil.account_conf.get_account(0)

    # Deploy the CompensationFactory
    artifact_path = run_script(
        "script/integrations/Deploy.s.sol:DeployAll",
        account=admin.address,
        env={"OWNER": admin.address},
    )
    contract = load_forge_artifact(artifact_path, "CompensationFactory")
    w3 = anvil.w3conf.get_web3()
    factory = CompensationFactoryContract.from_address(
        address=contract.address, w3=w3, account=admin
    )

    return factory, admin, w3


def test_factory_initialization(deploy_compensation_factory: Deployment):
    """Test that the factory is properly initialized."""
    factory, admin, w3 = deploy_compensation_factory

    # Test factory properties
    assert factory.address is not None


def test_create_compensation(
    deploy_compensation_factory: Deployment, anvil: AnvilContext
):
    """Test creating a compensation instance."""
    factory, admin, w3 = deploy_compensation_factory

    # Create config for SimpleMintCompensation
    config = SimpleMintCompensationConfig(
        token_name="SimpleMintCompensation",
        token_symbol="SMC",
        target_rewards=1000000 * 10**18,  # 1M tokens
        initial_admin=admin.address,
    )
    swarm_address = anvil.account_conf.get_account(10).address

    # Create the compensation instance
    deploy_tx = factory.create_compensation(
        compensation_id=config.get_compensation_id(),
        salt=w3.keccak(text="test-create-compensation"),
        init_data=config.get_init_data(swarm_address=swarm_address),
    )

    # Wait for transaction receipt
    w3.eth.wait_for_transaction_receipt(deploy_tx)

    # Get the deployed compensation address
    compensation_address = factory.get_deployed_compensation(deploy_tx)

    # Verify the compensation was created
    assert compensation_address is not None
    assert compensation_address != factory.address

    # Verify the compensation contract works
    compensation = SimpleMintCompensation.from_address(
        address=compensation_address, w3=w3, account=admin
    )
    assert compensation.name() == "SimpleMintCompensation"
    assert compensation.has_role(compensation.default_admin_role(), admin.address)


def test_get_deployed_compensation(
    deploy_compensation_factory: Deployment, anvil: AnvilContext
):
    """Test getting deployed compensation address."""
    factory, admin, w3 = deploy_compensation_factory

    # Create a compensation instance
    config = SimpleMintCompensationConfig(
        token_name="SimpleMintCompensation",
        token_symbol="SMC",
        target_rewards=1000000 * 10**18,  # 1M tokens
        initial_admin=admin.address,
    )
    swarm_address = anvil.account_conf.get_account(1).address

    deploy_tx = factory.create_compensation(
        compensation_id=config.get_compensation_id(),
        salt=w3.keccak(text="test-get-deployed"),
        init_data=config.get_init_data(swarm_address=swarm_address),
    )

    # Wait for transaction receipt
    w3.eth.wait_for_transaction_receipt(deploy_tx)

    # Get the deployed address
    compensation_address = factory.get_deployed_compensation(deploy_tx)

    # Verify it's a valid address
    assert compensation_address is not None
    assert len(compensation_address) == 42  # Ethereum address length


def test_compensation_removal(
    deploy_compensation_factory: Deployment, anvil: AnvilContext
):
    """Test removing a compensation implementation."""
    factory, admin, w3 = deploy_compensation_factory
    config = SimpleMintCompensationConfig(
        token_name="SimpleMintCompensation",
        token_symbol="SMC",
        target_rewards=1000000 * 10**18,  # 1M tokens
        initial_admin=admin.address,
    )
    implementation_id = config.get_compensation_id()
    tx_hash = factory.remove_compensation_implementation(
        compensation_id=implementation_id
    )

    w3.eth.wait_for_transaction_receipt(tx_hash)

    # Verify it's no longer registered
    assert not factory.is_compensation_registered(implementation_id)


def test_compensation_id_generation(deploy_compensation_factory: Deployment):
    """Test compensation ID generation."""
    factory, admin, w3 = deploy_compensation_factory

    # Create config
    config = SimpleMintCompensationConfig(
        token_name="SimpleMintCompensation",
        token_symbol="SMC",
        target_rewards=1000000 * 10**18,  # 1M tokens
        initial_admin=admin.address,
    )

    # Get the compensation ID
    compensation_id = config.get_compensation_id()

    # Verify it's a valid ID (should be a hash)
    assert compensation_id is not None
    assert len(compensation_id) == 32  # 32 bytes for keccak256 hash
