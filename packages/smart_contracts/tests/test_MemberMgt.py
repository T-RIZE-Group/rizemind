import ape
import pytest
from eth_account import Account
from eth_hash.auto import keccak
from eth_account.messages import encode_defunct
import json

# Fixture to use the first four accounts
@pytest.fixture
def accounts_fixture(accounts):
    return accounts[:4]  # Use first four accounts

# Fixture for the primary account
@pytest.fixture
def account(accounts_fixture):
    return accounts_fixture[0]

# Fixture for a new member account
@pytest.fixture
def new_member(accounts_fixture):
    return accounts_fixture[3]

# Fixture for the initial member accounts
@pytest.fixture
def members(accounts_fixture):
    return accounts_fixture[:3]  # First three accounts as initial members

# Fixture for the new model data to be submitted
@pytest.fixture
def new_model_data():
    model_data = {
        "name": "Model1",
        "parameters": {
            "param1": 0.1,
            "param2": 100,
            "param3": "abc"
        }
    }
    # Serialize model_data to JSON and encode as bytes
    model_data_bytes = json.dumps(model_data).encode('utf-8')
    return model_data_bytes

# Fixture to deploy the MemberManagement contract
@pytest.fixture
def memberManagementContract(account, project, members):
    contract = account.deploy(project.MemberManagement, members, 3)
    return contract

# Function to sign data with a private key
def sign_data(private_key, data):
    message_hash = keccak(data)
    signed_message = Account.sign_message(encode_defunct(message_hash), private_key)
    return signed_message.signature

# Test function for submitting a model update and verifying the model count
def test_submitModelUpdate(account, memberManagementContract, new_model_data, members):
    contract = memberManagementContract

    # Call the getModelCount method to verify it exists and get the current count
    initial_model_count = contract.getModelCount()
    print(f"Initial model count: {initial_model_count}")

    # Use a valid private key
    private_key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"  # Replace with an actual private key
    signature = sign_data(private_key, new_model_data)

    # Submit the model update
    contract.submitModelUpdate(new_model_data, signature, sender=account)

    # Verify the model count after submission
    updated_model_count = contract.getModelCount()
    print(f"Updated model count: {updated_model_count}")

    # Assert that the model count has increased by 1
    assert updated_model_count == initial_model_count + 1, "Model count should increment by 1"

    
