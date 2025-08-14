import os

import pytest
from eth_account import Account
from eth_account.messages import encode_defunct
from flwr.common import GetPropertiesIns, GetPropertiesRes
from flwr.common.typing import Code
from pydantic import ValidationError
from rizemind.authentication.signatures.signature import Signature
from rizemind.authentication.train_auth import (
    TRAIN_AUTH_PREFIX,
    RoundAuthResponseConfig,
    TrainAuthInsConfig,
    parse_train_auth_ins,
    parse_train_auth_res,
    prepare_train_auth_ins,
    prepare_train_auth_res,
)
from rizemind.contracts.erc.erc5267.typings import EIP712DomainMinimal
from rizemind.exception.parse_exception import ParseException
from web3 import Web3


@pytest.fixture(scope="module")
def domain() -> EIP712DomainMinimal:
    return EIP712DomainMinimal(
        name="RizeMind",
        version="1",
        chainId=1,
        verifyingContract=Web3.to_checksum_address(
            "0x0000000000000000000000000000000000000000"
        ),
    )


@pytest.fixture(scope="module")
def round_id() -> int:
    return 5


@pytest.fixture(scope="module")
def nonce() -> bytes:
    return os.urandom(16)


@pytest.fixture(scope="module")
def signed_message():
    acct = Account.create()
    msg = encode_defunct(text="pytest-train-auth")
    return Account.sign_message(msg, private_key=acct.key)


@pytest.fixture(scope="module")
def signature(signed_message) -> Signature:
    return Signature(data=signed_message.signature)


def test_prepare_and_parse_train_auth_ins_roundtrip(domain, round_id, nonce):
    ins: GetPropertiesIns = prepare_train_auth_ins(
        domain=domain, round_id=round_id, nonce=nonce
    )

    # Every produced key must start with the expected prefix
    assert all(k.startswith(TRAIN_AUTH_PREFIX) for k in ins.config)

    parsed: TrainAuthInsConfig = parse_train_auth_ins(ins)

    assert parsed.round_id == round_id
    assert parsed.nonce == nonce
    assert parsed.domain.model_dump() == domain.model_dump()


def test_prepare_and_parse_train_auth_res_roundtrip(signature):
    res: GetPropertiesRes = prepare_train_auth_res(signature)

    # Status object set correctly
    assert res.status.code == Code.OK
    assert res.status.message == "auth signed"

    parsed: RoundAuthResponseConfig = parse_train_auth_res(res)
    assert parsed.signature.data == signature.data  # identical bytes


def test_train_auth_ins_validation_error_on_bad_nonce(domain):
    with pytest.raises(ValidationError):
        # nonce must be bytes, int will NOT coerce
        prepare_train_auth_ins(domain=domain, round_id=1, nonce=123)  # type: ignore[arg-type]


def test_train_auth_ins_validation_error_bad_round_id(domain, nonce):
    with pytest.raises(ValidationError):
        prepare_train_auth_ins(domain=domain, round_id="not-int", nonce=nonce)  # type: ignore[arg-type]


def test_train_auth_ins_parse_missing_prefix_raises():
    # Deliberately omit the expected "rizemind.train_auth" prefix
    bad_ins = GetPropertiesIns(config={"foo.bar": 1})
    with pytest.raises(ParseException):
        parse_train_auth_ins(bad_ins)


def test_round_auth_response_validation_error_bad_signature():
    # Passing *anything* other than a Signature instance should fail
    with pytest.raises(ValidationError):
        prepare_train_auth_res(signature="not-a-signature")  # type: ignore[arg-type]


def test_signature_field_validation_error_invalid_bytes():
    # Signature of incorrect length → fails when the model is built
    with pytest.raises(ValidationError):
        RoundAuthResponseConfig(signature=Signature(data=b"\x11" * 64))  # 64 ≠ 65
