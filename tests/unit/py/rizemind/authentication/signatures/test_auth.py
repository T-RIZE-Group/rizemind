import pytest
from eth_account import Account
from rizemind.authentication.signatures.auth import (
    recover_auth_signer,
    sign_auth_message,
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
def test_nonce() -> str:
    return Web3.to_hex(Web3.keccak(text="test-nonce"))


@pytest.fixture
def test_round_id() -> int:
    return 1


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


def test_sign_auth_message(
    eth_account, test_nonce, test_round_id, domain: EIP712DomainRequiredFields
):
    """Test that authentication message signing works correctly"""
    signed_message = sign_auth_message(
        account=eth_account, nonce=test_nonce, round=test_round_id, domain=domain
    )

    assert isinstance(signed_message, Signature), "should return a Signature"
    assert hasattr(signed_message, "data"), "should have data attribute"


def test_recover_auth_signer(
    eth_account, test_nonce, test_round_id, domain: EIP712DomainRequiredFields
):
    """Test that we can recover the correct signer from an auth signature"""
    # First sign the message
    signature = sign_auth_message(
        account=eth_account, nonce=test_nonce, round=test_round_id, domain=domain
    )

    # Recover signer
    recovered_address = recover_auth_signer(
        round=test_round_id, nonce=test_nonce, domain=domain, signature=signature
    )

    assert recovered_address == eth_account.address, (
        "recovered address should match signer"
    )


def test_different_nonce_different_signatures(
    eth_account, test_round_id, domain: EIP712DomainRequiredFields
):
    """Test that different nonces produce different signatures"""
    nonce1 = Web3.keccak(text="nonce1")
    nonce2 = Web3.keccak(text="nonce2")

    signed1 = sign_auth_message(
        account=eth_account, nonce=nonce1, round=test_round_id, domain=domain
    )

    signed2 = sign_auth_message(
        account=eth_account, nonce=nonce2, round=test_round_id, domain=domain
    )

    assert str(signed1) != str(signed2), (
        "different nonces should produce different signatures"
    )


def test_same_nonce_same_signature(
    eth_account, test_nonce, test_round_id, domain: EIP712DomainRequiredFields
):
    """Test that same inputs produce same signature"""
    signed1 = sign_auth_message(
        account=eth_account, nonce=test_nonce, round=test_round_id, domain=domain
    )

    signed2 = sign_auth_message(
        account=eth_account, nonce=test_nonce, round=test_round_id, domain=domain
    )

    assert str(signed1) == str(signed2), "same inputs should produce same signature"
