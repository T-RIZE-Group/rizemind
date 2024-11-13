import pytest
import numpy as np
from web3 import Web3
from tensorflow import keras
import torch
from eth_account import Account
from eth_account.datastructures import SignedMessage
from .signature import hash_weights, hash_tf_model, hash_torch_model, sign_torch_model, prepare_eip712_message, sign_tf_model, recover_model_signer
from keras import layers

def test_hash_weights():
    # Create a mock weight array
    weights = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    # Manually hash the byte representation using Keccak-256
    expected_hash = Web3.keccak(weights.tobytes()).hex()

    # Verify that sign_weights produces the correct hash
    assert hash_weights(weights) == expected_hash, "sign_weights did not produce the expected Keccak-256 hash"

def test_hash_tf_model():
    # Create a simple TensorFlow model
    model = keras.Sequential([
        keras.Input(shape=(10, 1)),
        keras.layers.Dense(5)
    ])

    # Get the weights directly from the model
    weights = np.concatenate([layer.flatten() for layer in model.get_weights()]).astype(np.float32)

    # Manually hash the weights for comparison
    expected_hash = Web3.keccak(weights.tobytes()).hex()

    # Verify that sign_tf_model produces the correct hash
    assert hash_tf_model(model) == expected_hash, "sign_tf_model did not produce the expected Keccak-256 hash"

def test_hash_torch_model():
    # Create a simple PyTorch model
    model = torch.nn.Linear(10, 5)

    # Get the weights directly from the model
    weights = np.concatenate([param.detach().numpy().flatten() for param in model.parameters()]).astype(np.float32)

    # Manually hash the weights for comparison
    expected_hash = Web3.keccak(weights.tobytes()).hex()

    # Verify that sign_torch_model produces the correct hash
    assert hash_torch_model(model) == expected_hash, "sign_torch_model did not produce the expected Keccak-256 hash"

@pytest.fixture
def torch_model():
    # Create a simple PyTorch model for testing
    model = torch.nn.Linear(10, 5)
    return model
@pytest.fixture
def eth_account():
    # Create a test Ethereum account
    return Account.create()


def test_sign_torch_model(torch_model, eth_account):
    chain_id = 1 
    contract_address = "0xCcCCccccCCCCcCCCCCCcCcCccCcCCCcCcccccccC"  
    app_name = "TestApp"
    round_number = 1

    signed_message = sign_torch_model(eth_account, torch_model, chain_id, contract_address, app_name, round_number)
    print(type(signed_message))
    assert isinstance(signed_message, SignedMessage), "should return a SignedMessage"

    message = prepare_eip712_message(chain_id, contract_address, app_name, round_number, hash_torch_model(torch_model))
    address = Account.recover_message(message, [signed_message.v, signed_message.r, signed_message.s])
    assert address == eth_account.address, "recovered address doesn't match"

def test_sign_tf_model(eth_account):
    model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile()
    chain_id = 1 
    contract_address = "0xCcCCccccCCCCcCCCCCCcCcCccCcCCCcCcccccccC"  
    app_name = "TestApp"
    round_number = 1
    signed_message = sign_tf_model(eth_account, model, chain_id, contract_address, app_name, round_number)
    recovered_address = recover_model_signer(model.get_weights(), chain_id, contract_address, app_name, round_number, [signed_message.v, signed_message.r, signed_message.s])
    assert recovered_address == eth_account.address, "recovered address doesn't match signer"