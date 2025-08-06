import pytest
from eth_account import Account
from rizemind.authentication.notary.model.model_signature import (
    Parameters,
    hash_parameters,
    recover_model_signer,
    sign_parameters_model,
)
from rizemind.authentication.signatures.eip712 import (
    EIP712DomainRequiredFields,
    EIP712DomainStruct,
)
from rizemind.authentication.signatures.signature import Signature
from web3 import Web3


@pytest.fixture
def eth_account():
    # Create a test Ethereum account
    return Account.create()


@pytest.fixture
def test_params():
    return Parameters(tensors=[bytes(0x87231), bytes(0x5423)], tensor_type="float32")


@pytest.fixture
def domain() -> EIP712DomainRequiredFields:
    return EIP712DomainStruct(
        name="test",
        version="1.0.0",
        chainId=1,
        verifyingContract=Web3.to_checksum_address(
            "0xCcCCccccCCCCcCCCCCCcCcCccCcCCCcCcccccccC"
        ),
    )


def test_hash_parameters(test_params):
    """Test that parameter hashing produces expected output"""
    hash_result = hash_parameters(test_params)
    assert isinstance(hash_result, bytes), "hash should be a hex string"
    assert len(hash_result) == 32, "hash should be 32 bytes"


def test_sign_parameters_model(eth_account, test_params, domain):
    """Test that model parameter signing works correctly"""
    signed_message = sign_parameters_model(
        account=eth_account, domain=domain, round=1, parameters=test_params
    )
    assert isinstance(signed_message, Signature), "should return a SignedMessage"


def test_recover_model_signer(eth_account, test_params, domain):
    """Test that we can recover the correct signer from a signature"""
    round = 1
    signature = sign_parameters_model(
        account=eth_account, domain=domain, round=round, parameters=test_params
    )

    # Recover signer
    recovered_address = recover_model_signer(
        model=test_params, round=round, domain=domain, signature=signature
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
