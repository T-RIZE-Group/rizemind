import pytest
from eth_account import Account
from eth_account.datastructures import SignedMessage
from rizemind.authentication.signatures.model import (
    Parameters,
    hash_parameters,
    recover_model_signer,
    sign_parameters_model,
)


@pytest.fixture
def eth_account():
    # Create a test Ethereum account
    return Account.create()


@pytest.fixture
def test_params():
    return Parameters(tensors=[bytes(0x87231), bytes(0x5423)], tensor_type="float32")


@pytest.fixture
def signing_params():
    return {
        "version": "1.0.0",
        "chain_id": 1,
        "contract_address": "0xCcCCccccCCCCcCCCCCCcCcCccCcCCCcCcccccccC",
        "app_name": "TestApp",
        "round_number": 1,
    }


def test_hash_parameters(test_params):
    """Test that parameter hashing produces expected output"""
    hash_result = hash_parameters(test_params)
    assert isinstance(hash_result, str), "hash should be a hex string"
    assert hash_result.startswith("0x"), "hash should start with 0x"
    assert len(hash_result) == 66, "hash should be 32 bytes (66 chars including 0x)"


def test_sign_parameters_model(eth_account, test_params, signing_params):
    """Test that model parameter signing works correctly"""
    signed_message = sign_parameters_model(
        eth_account,
        signing_params["version"],
        test_params,
        signing_params["chain_id"],
        signing_params["contract_address"],
        signing_params["app_name"],
        signing_params["round_number"],
    )
    assert isinstance(signed_message, SignedMessage), "should return a SignedMessage"
    assert hasattr(signed_message, "signature"), "should have signature attribute"
    assert hasattr(signed_message, "message_hash"), "should have message_hash attribute"


def test_recover_model_signer(eth_account, test_params, signing_params):
    """Test that we can recover the correct signer from a signature"""
    # First sign the message
    signed_message = sign_parameters_model(
        eth_account,
        signing_params["version"],
        test_params,
        signing_params["chain_id"],
        signing_params["contract_address"],
        signing_params["app_name"],
        signing_params["round_number"],
    )

    # Extract v, r, s from signature
    sig = signed_message.signature
    v, r, s = sig[-1], sig[:32], sig[32:64]

    # Recover signer
    recovered_address = recover_model_signer(
        test_params,
        signing_params["version"],
        signing_params["chain_id"],
        signing_params["contract_address"],
        signing_params["app_name"],
        signing_params["round_number"],
        (v, r, s),
    )

    assert recovered_address == eth_account.address, (
        "recovered address should match signer"
    )


def test_different_parameters_different_hashes():
    """Test that different parameters produce different hashes"""
    params1 = Parameters(tensors=[bytes(0x1)], tensor_type="float32")
    params2 = Parameters(tensors=[bytes(0x2)], tensor_type="float32")

    hash1 = hash_parameters(params1)
    hash2 = hash_parameters(params2)

    assert hash1 != hash2, "different parameters should produce different hashes"


def test_same_parameters_same_hash(test_params):
    """Test that same parameters produce same hash"""
    hash1 = hash_parameters(test_params)
    hash2 = hash_parameters(test_params)

    assert hash1 == hash2, "same parameters should produce same hash"
