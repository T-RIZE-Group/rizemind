import pytest
from eth_account.messages import SignableMessage
from rizemind.authentication.signatures.eip712 import (
    EIP712DomainABI,
    EIP712DomainRequiredFields,
    EIP712DomainStruct,
    EIP712DomainTypeName,
    prepare_eip712_message,
)
from web3 import Web3


@pytest.fixture
def domain_params():
    return {
        "chainid": 1,
        "version": "1.0.0",
        "contract": "0xCcCCccccCCCCcCCCCCCcCcCccCcCCCcCcccccccC",
        "name": "TestDomain",
    }


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


@pytest.fixture
def test_type():
    return {
        "Test": [
            {"name": "testField", "type": "string"},
            {"name": "testNumber", "type": "uint256"},
        ]
    }


@pytest.fixture
def test_message():
    return {
        "testField": "test value",
        "testNumber": 123,
    }


def test_prepare_eip712_message(domain, test_type, test_message):
    """Test EIP-712 message preparation"""

    encoded_message = prepare_eip712_message(
        domain,
        "Test",
        test_message,
        test_type,
    )

    assert isinstance(encoded_message, SignableMessage)


def test_eip712_domain_abi_structure():
    """Test EIP-712 domain ABI structure"""
    assert len(EIP712DomainABI) == 4

    required_fields = {"name", "version", "chainId", "verifyingContract"}
    abi_fields = {field["name"] for field in EIP712DomainABI}

    assert abi_fields == required_fields, "required fields don't match"


def test_message_preparation_with_empty_types(domain, test_type):
    """Test message preparation with empty types dictionary"""

    with pytest.raises(ValueError):
        prepare_eip712_message(
            domain,
            "Test",
            {"field": "value"},
        )

    with pytest.raises(ValueError):
        prepare_eip712_message(
            domain,
            "Test",
            {"field": "value"},
            test_type,
        )


def test_domain_type_constant():
    """Test EIP-712 domain type name constant"""
    assert EIP712DomainTypeName == "EIP712Domain"
    assert isinstance(EIP712DomainTypeName, str)
