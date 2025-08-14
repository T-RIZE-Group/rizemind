import pytest
from rizemind.authentication.notary.model.config import (
    MODEL_NOTARY_PREFIX,
    ModelNotaryConfig,
    parse_model_notary_config,
    prepare_model_notary_config,
)
from rizemind.authentication.signatures.signature import Signature
from rizemind.contracts.erc.erc5267.typings import EIP712Domain, EIP712DomainMinimal
from rizemind.exception.parse_exception import ParseException
from web3 import Web3


@pytest.fixture
def test_round_id() -> int:
    """Create a test round ID."""
    return 1


@pytest.fixture
def test_model_hash() -> bytes:
    """Create a test model hash."""
    return Web3.keccak(text="test-model-hash")


@pytest.fixture
def domain() -> EIP712Domain:
    """Create a test EIP712 domain."""
    return EIP712Domain(
        name="test",
        version="1.0.0",
        chainId=1,
        verifyingContract=Web3.to_checksum_address(
            "0xCcCCccccCCCCcCCCCCCcCcCccCcCCCcCcccccccC"
        ),
        fields=b"test_fields",
        salt=b"test_salt",
        extensions=[1, 2, 3],
    )


def test_prepare_model_notary_config_success(
    test_round_id: int,
    domain: EIP712Domain,
    random_signature: Signature,
    test_model_hash: bytes,
):
    """Test that model notary config preparation works correctly."""
    result = prepare_model_notary_config(
        round_id=test_round_id,
        domain=domain,
        signature=random_signature,
        model_hash=test_model_hash,
    )

    assert f"{MODEL_NOTARY_PREFIX}.domain.name" in result, "should contain domain"
    assert f"{MODEL_NOTARY_PREFIX}.round_id" in result, "should contain round_id"
    assert f"{MODEL_NOTARY_PREFIX}.model_hash" in result, "should contain model_hash"
    assert f"{MODEL_NOTARY_PREFIX}.signature.data" in result, "should contain signature"

    # Verify the values
    assert result[f"{MODEL_NOTARY_PREFIX}.round_id"] == test_round_id
    assert result[f"{MODEL_NOTARY_PREFIX}.model_hash"] == test_model_hash
    assert result[f"{MODEL_NOTARY_PREFIX}.signature.data"] == random_signature.data


def test_parse_model_notary_config_success(
    minimal_domain: EIP712DomainMinimal,
    random_signature: Signature,
    test_model_hash: bytes,
    test_round_id: int,
):
    """Test that model notary config parsing works correctly."""
    config_dict = prepare_model_notary_config(
        round_id=test_round_id,
        domain=minimal_domain,
        signature=random_signature,
        model_hash=test_model_hash,
    )

    result = parse_model_notary_config(config_dict)

    assert isinstance(result, ModelNotaryConfig), "should return a ModelNotaryConfig"
    assert result.domain == minimal_domain, "should have correct domain"
    assert result.round_id == test_round_id, "should have correct round_id"
    assert result.model_hash == test_model_hash, "should have correct model_hash"
    assert result.signature == random_signature, "should have correct signature"


def test_parse_model_notary_config_missing_rizemind_key():
    """Test that missing rizemind key raises ParseException."""
    config_dict = {"invalid": "data"}

    with pytest.raises(ParseException):
        parse_model_notary_config(config_dict)


def test_parse_model_notary_config_missing_notary_key():
    """Test that missing notary key raises ParseException."""
    config_dict = {f"{MODEL_NOTARY_PREFIX}.invalid": "data"}

    with pytest.raises(ParseException):
        parse_model_notary_config(config_dict)


def test_parse_model_notary_config_missing_domain(
    test_round_id: int,
    random_signature: Signature,
    test_model_hash: bytes,
    minimal_domain: EIP712DomainMinimal,
):
    """Test that missing field raises ParseException."""
    config_dict = prepare_model_notary_config(
        round_id=test_round_id,
        domain=minimal_domain,
        signature=random_signature,
        model_hash=test_model_hash,
    )
    del config_dict[f"{MODEL_NOTARY_PREFIX}.round_id"]

    with pytest.raises(ParseException):
        parse_model_notary_config(config_dict)


def test_different_values_produce_different_configs(
    domain: EIP712Domain, random_signature: Signature
):
    """Test that different values produce different configs."""
    round_id1 = 1
    round_id2 = 2
    model_hash1 = Web3.keccak(text="model-hash-1")
    model_hash2 = Web3.keccak(text="model-hash-2")

    config1 = prepare_model_notary_config(
        round_id=round_id1,
        domain=domain,
        signature=random_signature,
        model_hash=model_hash1,
    )

    config2 = prepare_model_notary_config(
        round_id=round_id2,
        domain=domain,
        signature=random_signature,
        model_hash=model_hash2,
    )

    assert config1 != config2, "different values should produce different configs"


def test_same_values_produce_same_config(
    test_round_id: int,
    domain: EIP712Domain,
    random_signature: Signature,
    test_model_hash: bytes,
):
    """Test that same values produce same config."""
    config1 = prepare_model_notary_config(
        round_id=test_round_id,
        domain=domain,
        signature=random_signature,
        model_hash=test_model_hash,
    )

    config2 = prepare_model_notary_config(
        round_id=test_round_id,
        domain=domain,
        signature=random_signature,
        model_hash=test_model_hash,
    )

    assert config1 == config2, "same values should produce same config"
