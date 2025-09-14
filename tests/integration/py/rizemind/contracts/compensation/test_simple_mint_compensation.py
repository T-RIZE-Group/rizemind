import pytest
from eth_account.signers.base import BaseAccount
from eth_typing import ChecksumAddress
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

type Deployment = tuple[SimpleMintCompensation, BaseAccount, ChecksumAddress, Web3]


@pytest.fixture(scope="module")
def deploy_simple_mint_compensation(anvil: AnvilContext):
    """Deploy a SimpleMintCompensation instance using the factory pattern."""
    admin = anvil.account_conf.get_account(0)
    swarm_address = anvil.account_conf.get_account(1).address  # Mock swarm address

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

    target_rewards = 1000000 * 10**18
    # Create config for SimpleMintCompensation
    config = SimpleMintCompensationConfig(
        token_name="SimpleMintCompensation",
        token_symbol="SMC",
        target_rewards=target_rewards,  # 1M tokens
        initial_admin=admin.address,
        minter=admin.address,
    )

    # Create the compensation instance
    deploy_tx = factory.create_compensation(
        compensation_id=config.get_compensation_id(),
        salt=w3.keccak(text="test-simple-mint-compensation"),
        init_data=config.get_init_data(swarm_address=swarm_address),
    )

    # Wait for transaction receipt
    w3.eth.wait_for_transaction_receipt(deploy_tx)

    # Get the deployed compensation address
    compensation_address = factory.get_deployed_compensation(deploy_tx)

    # Create the compensation contract instance
    compensation = SimpleMintCompensation.from_address(
        address=compensation_address, w3=w3, account=admin
    )

    return compensation, admin, swarm_address, w3


def test_compensation_initialization(deploy_simple_mint_compensation: Deployment):
    """Test that the compensation contract is properly initialized."""
    compensation, admin, swarm_address, w3 = deploy_simple_mint_compensation

    # Test ERC20 basic properties
    assert compensation.name() == "SimpleMintCompensation"
    assert compensation.symbol() == "SMC"
    assert compensation.decimals() == 18
    assert compensation.total_supply() == 0

    # Test that admin has proper role
    assert compensation.has_role(compensation.default_admin_role(), admin.address)
    assert compensation.has_role(compensation.get_role("MINTER_ROLE"), admin.address)


def test_get_compensation_params(anvil: AnvilContext):
    """Test getting compensation parameters."""
    admin = anvil.account_conf.get_account(0)

    # Create config
    config = SimpleMintCompensationConfig(
        token_name="SimpleMintCompensation",
        token_symbol="SMC",
        target_rewards=1000000 * 10**18,  # 1M tokens
        initial_admin=admin.address,
    )
    swarm_address = anvil.account_conf.get_account(1).address

    # Get parameters
    params = config.get_compensation_params(swarm_address=swarm_address)

    # Verify parameters
    assert params.id == config.get_compensation_id()
    assert params.init_data is not None


def test_erc20_functionality(
    deploy_simple_mint_compensation: Deployment, anvil: AnvilContext
):
    """Test basic ERC20 token functionality."""
    compensation, admin, swarm_address, w3 = deploy_simple_mint_compensation

    # Get test accounts
    user1 = anvil.account_conf.get_account(1).address
    user2 = anvil.account_conf.get_account(2).address

    # Test initial balances
    assert compensation.balance_of(admin.address) == 0
    assert compensation.balance_of(user1) == 0
    assert compensation.balance_of(user2) == 0

    # Test allowance
    assert compensation.allowance(admin.address, user1) == 0


def test_mint_and_transfer(
    deploy_simple_mint_compensation: Deployment, anvil: AnvilContext
):
    """Test minting tokens and transferring them."""
    compensation, admin, swarm_address, w3 = deploy_simple_mint_compensation
    target_rewards = 1000000 * 10**18

    # Get test accounts
    user1 = anvil.account_conf.get_account(21)
    contribution = 10**6  # all tokens

    # Mint tokens to user1
    tx_hash = compensation.distribute(
        round_id=1,
        recipients=[user1.address],
        contributions=[contribution],
    )
    w3.eth.wait_for_transaction_receipt(tx_hash)

    # Check balance
    assert compensation.balance_of(user1.address) == target_rewards
    assert compensation.total_supply() == target_rewards

    # Test transfer from user1 to user2
    user2 = anvil.account_conf.get_account(22)
    transfer_amount = int(target_rewards / 2)
    # Connect user1 account to the contract
    compensation.connect(user1)

    tx_hash = compensation.transfer(user2.address, transfer_amount)
    w3.eth.wait_for_transaction_receipt(tx_hash)

    # Check balances after transfer
    assert compensation.balance_of(user1.address) == target_rewards - transfer_amount
    assert compensation.balance_of(user2.address) == transfer_amount


def test_role_management(
    deploy_simple_mint_compensation: Deployment, anvil: AnvilContext
):
    """Test role-based access control functionality."""
    compensation, admin, swarm_address, w3 = deploy_simple_mint_compensation

    # Get test accounts
    compensation.connect(admin)
    user1 = anvil.account_conf.get_account(31).address
    minter_role = compensation.get_role("MINTER_ROLE")

    # Test that user1 doesn't have minter role initially
    assert not compensation.has_role(minter_role, user1)

    # Grant minter role to user1
    tx_hash = compensation.grant_role(minter_role, user1)
    w3.eth.wait_for_transaction_receipt(tx_hash)

    # Check that user1 now has minter role
    assert compensation.has_role(minter_role, user1)

    # Revoke minter role from user1
    tx_hash = compensation.revoke_role(minter_role, user1)
    w3.eth.wait_for_transaction_receipt(tx_hash)

    # Check that user1 no longer has minter role
    assert not compensation.has_role(minter_role, user1)


def test_eip712_domain(deploy_simple_mint_compensation: Deployment):
    """Test EIP712 domain information."""
    compensation, admin, swarm_address, w3 = deploy_simple_mint_compensation

    domain = compensation.get_eip712_domain()

    assert domain.name == "SimpleMintCompensation"
    assert domain.version == "simple-mint-compensation-v1.0.0"
    assert domain.verifyingContract == compensation.address


def test_distribute_compensation(deploy_simple_mint_compensation: Deployment):
    """Test distributing compensation to multiple recipients."""
    compensation, admin, swarm_address, w3 = deploy_simple_mint_compensation
    target_rewards = 1000000 * 10**18
    # Distribute with empty recipients (should not fail but do nothing)
    initial_supply = compensation.total_supply()

    tx_hash = compensation.distribute(
        round_id=4,
        recipients=[],
        contributions=[],
    )
    w3.eth.wait_for_transaction_receipt(tx_hash)

    # Total supply should remain unchanged
    assert compensation.total_supply() == initial_supply

    # Get test accounts
    recipients = [w3.eth.accounts[i] for i in range(1, 4)]  # users 1, 2, 3
    contributions = [100 * 10**6, 200 * 10**6, 300 * 10**6]  # 100, 200, 300 tokens

    # Distribute compensation
    tx_hash = compensation.distribute(
        round_id=3,
        recipients=recipients,
        contributions=contributions,
    )
    w3.eth.wait_for_transaction_receipt(tx_hash)

    # Check balances
    for i, recipient in enumerate(recipients):
        assert compensation.balance_of(recipient) > 0
