import pytest
from eth_typing import ChecksumAddress
from hexbytes import HexBytes
from rizemind.contracts.contribution.calculator_factory import (
    CalculatorFactoryContract,
)
from rizemind.contracts.contribution.contribution_calculator.contribution_calculator import (
    ContributionCalculator,
    ContributionCalculatorConfig,
)
from rizemind.contracts.local_deployment import load_forge_artifact
from web3 import Web3

from tests.integration.forge_fixtures import AnvilContext
from tests.integration.forge_helper import run_script

type Deployment = tuple[ContributionCalculator, ChecksumAddress, ChecksumAddress, Web3]


@pytest.fixture(scope="module")
def deploy_contribution_calculator(anvil: AnvilContext):
    """Deploy a ContributionCalculator instance using the factory pattern."""
    admin = anvil.account_conf.get_account(0)
    swarm_address = anvil.account_conf.get_account(1).address  # Mock swarm address

    # Deploy the CalculatorFactory
    artifact_path = run_script(
        "script/integrations/Deploy.s.sol:DeployAll",
        account=admin.address,
        env={"OWNER": admin.address},
    )
    contract = load_forge_artifact(artifact_path, "CalculatorFactory")
    w3 = anvil.w3conf.get_web3()
    factory = CalculatorFactoryContract.from_address(
        address=contract.address, w3=w3, account=admin
    )

    # Create config for ContributionCalculator
    config = ContributionCalculatorConfig(
        initial_admin=admin.address,
        initial_num_samples=10,
    )

    # Create the calculator instance
    deploy_tx = factory.create_calculator(
        calculator_id=config.get_calculator_id(),
        salt=w3.keccak(text="test-contribution-calculator"),
        init_data=config.get_init_data(swarm_address=swarm_address),
    )

    # Get the deployed calculator address
    calculator_address = factory.get_deployed_calculator(deploy_tx)
    calculator = ContributionCalculator.from_address(address=calculator_address, w3=w3)
    calculator.connect(admin)

    return calculator, admin.address, swarm_address, w3


def test_calculator_initialization(deploy_contribution_calculator: Deployment):
    """Test that the calculator is properly initialized."""
    calculator, admin_address, swarm_address, w3 = deploy_contribution_calculator

    # Test that we can get evaluations required
    evaluations_required = calculator.get_evaluations_required(
        round_id=0, number_of_players=3
    )
    assert evaluations_required == 10  # Should match initial_num_samples

    # Test that we can get total evaluations
    total_evaluations = calculator.get_total_evaluations(
        round_id=0, number_of_players=3
    )
    assert total_evaluations == 8  # 2^3 = 8 for 3 players


def test_register_result(deploy_contribution_calculator: Deployment):
    """Test registering evaluation results."""
    calculator, admin_address, swarm_address, w3 = deploy_contribution_calculator

    # Register a result
    round_id = 1
    sample_id = 0
    number_of_players = 3
    set_id = calculator.get_mask(
        round_id=round_id, i=sample_id, number_of_players=number_of_players
    )
    model_hash = HexBytes(Web3.keccak(text=str(set_id)))
    result = 100

    tx_hash = calculator.register_result(
        round_id=round_id,
        sample_id=sample_id,
        set_id=set_id,
        model_hash=model_hash,
        result=result,
        number_of_players=number_of_players,
    )

    assert tx_hash is not None
    w3.eth.wait_for_transaction_receipt(tx_hash)

    # Verify the result was stored
    stored_result = calculator.get_result(round_id=round_id, set_id=set_id)
    assert stored_result == result


def test_calculate_contribution(deploy_contribution_calculator: Deployment):
    """Test calculating Shapley values."""
    calculator, admin_address, swarm_address, w3 = deploy_contribution_calculator

    round_id = 2
    number_of_players = 2
    tx_hash = calculator.set_evaluations_required(
        round_id=round_id, evaluations_required=2 * number_of_players
    )
    w3.eth.wait_for_transaction_receipt(tx_hash)

    sample_id_0 = 0
    set_id_0 = calculator.get_mask(
        round_id=round_id, i=sample_id_0, number_of_players=number_of_players
    )
    model_hash_0 = HexBytes(Web3.keccak(text=f"model_{set_id_0}"))
    tx_hash = calculator.register_result(
        round_id=round_id,
        sample_id=sample_id_0,
        set_id=set_id_0,
        model_hash=model_hash_0,
        result=0,
        number_of_players=number_of_players,
    )
    w3.eth.wait_for_transaction_receipt(tx_hash)

    sample_id_1 = 1
    set_id_1 = calculator.get_mask(
        round_id=round_id, i=sample_id_1, number_of_players=number_of_players
    )
    model_hash_1 = HexBytes(Web3.keccak(text=f"model_{set_id_1}"))
    tx_hash = calculator.register_result(
        round_id=round_id,
        sample_id=sample_id_1,
        set_id=set_id_1,
        model_hash=model_hash_1,
        result=50,
        number_of_players=number_of_players,
    )
    w3.eth.wait_for_transaction_receipt(tx_hash)

    sample_id_2 = 2
    set_id_2 = calculator.get_mask(
        round_id=round_id, i=sample_id_2, number_of_players=number_of_players
    )
    model_hash_2 = HexBytes(Web3.keccak(text=f"model_{set_id_2}"))
    tx_hash = calculator.register_result(
        round_id=round_id,
        sample_id=sample_id_2,
        set_id=set_id_2,
        model_hash=model_hash_2,
        result=30,
        number_of_players=number_of_players,
    )
    w3.eth.wait_for_transaction_receipt(tx_hash)

    sample_id_3 = 3
    set_id_3 = calculator.get_mask(
        round_id=round_id, i=sample_id_3, number_of_players=number_of_players
    )
    model_hash_3 = HexBytes(Web3.keccak(text=f"model_{set_id_3}"))
    tx_hash = calculator.register_result(
        round_id=round_id,
        sample_id=sample_id_3,
        set_id=set_id_3,
        model_hash=model_hash_3,
        result=100,
        number_of_players=number_of_players,
    )
    w3.eth.wait_for_transaction_receipt(tx_hash)

    # Calculate contributions
    contribution_0 = calculator.calculate_contribution(
        round_id=round_id, trainer_index=0, number_of_trainers=number_of_players
    )
    contribution_1 = calculator.calculate_contribution(
        round_id=round_id, trainer_index=1, number_of_trainers=number_of_players
    )

    assert isinstance(contribution_0, int)
    assert isinstance(contribution_1, int)


def test_get_mask(deploy_contribution_calculator: Deployment):
    """Test mask generation for different rounds and samples."""
    calculator, admin_address, swarm_address, w3 = deploy_contribution_calculator

    round_id = 3
    number_of_players = 3

    # Get masks for different sample indices
    mask_0 = calculator.get_mask(
        round_id=round_id, i=0, number_of_players=number_of_players
    )
    mask_1 = calculator.get_mask(
        round_id=round_id, i=1, number_of_players=number_of_players
    )
    mask_2 = calculator.get_mask(
        round_id=round_id, i=2, number_of_players=number_of_players
    )

    # Masks should be different for different sample indices
    assert mask_0 != mask_1
    assert mask_1 != mask_2
    assert mask_0 != mask_2

    # Masks should be within valid range (0 to 2^number_of_players - 1)
    max_mask = (1 << number_of_players) - 1
    assert 0 <= mask_0 <= max_mask
    assert 0 <= mask_1 <= max_mask
    assert 0 <= mask_2 <= max_mask


def test_admin_role_management(deploy_contribution_calculator: Deployment):
    """Test admin role management functions."""
    calculator, admin_address, swarm_address, w3 = deploy_contribution_calculator

    # Get a new account for testing role management
    new_admin = Web3.to_checksum_address(
        "0x1234567890123456789012345678901234567890"
    )  # Mock address

    # Grant admin role to new account
    tx_hash = calculator.grant_admin_role(new_admin)
    assert tx_hash is not None
    w3.eth.wait_for_transaction_receipt(tx_hash)

    # Revoke admin role from new account
    tx_hash = calculator.revoke_admin_role(new_admin)
    assert tx_hash is not None
    w3.eth.wait_for_transaction_receipt(tx_hash)


def test_set_evaluations_required(deploy_contribution_calculator: Deployment):
    """Test setting the number of evaluations required for a round."""
    calculator, admin_address, swarm_address, w3 = deploy_contribution_calculator

    round_id = 4
    new_evaluations_required = 20

    # Set new evaluations required
    tx_hash = calculator.set_evaluations_required(
        round_id=round_id, evaluations_required=new_evaluations_required
    )
    assert tx_hash is not None
    w3.eth.wait_for_transaction_receipt(tx_hash)

    # Verify the change
    evaluations_required = calculator.get_evaluations_required(
        round_id=round_id, number_of_players=3
    )
    assert evaluations_required == new_evaluations_required


def test_supports_interface(deploy_contribution_calculator: Deployment):
    """Test interface support."""
    calculator, admin_address, swarm_address, w3 = deploy_contribution_calculator

    # Test that it supports the IERC165 interface
    # This is a basic interface ID that most contracts should support
    supports_erc165 = calculator.supports_interface("0x01ffc9a7")
    assert supports_erc165


def test_eip712_domain(deploy_contribution_calculator: Deployment):
    """Test EIP712 domain information."""
    calculator, admin_address, swarm_address, w3 = deploy_contribution_calculator

    domain = calculator.get_eip712_domain()

    # Verify the contract name and version
    assert domain.name == "ContributionCalculator"
    assert domain.version == "contribution-calculator-v1.0.0"
    assert domain.verifyingContract == calculator.contract.address


def test_get_result_or_throw(deploy_contribution_calculator: Deployment):
    """Test get_result_or_throw with existing and non-existing results."""
    calculator, admin_address, swarm_address, w3 = deploy_contribution_calculator

    round_id = 5
    sample_id = 0
    number_of_players = 3
    set_id = calculator.get_mask(
        round_id=round_id, i=sample_id, number_of_players=number_of_players
    )
    model_hash = HexBytes(Web3.keccak(text=f"model_{set_id}"))
    result = 42

    # First, register a result
    tx_hash = calculator.register_result(
        round_id=round_id,
        sample_id=sample_id,
        set_id=set_id,
        model_hash=model_hash,
        result=result,
        number_of_players=number_of_players,
    )
    w3.eth.wait_for_transaction_receipt(tx_hash)

    # Should return the result without throwing
    retrieved_result = calculator.get_result_or_throw(round_id=round_id, set_id=set_id)
    assert retrieved_result == result

    # Should throw for non-existing result
    with pytest.raises(Exception):  # The contract should revert
        calculator.get_result_or_throw(round_id=round_id, set_id=999)
