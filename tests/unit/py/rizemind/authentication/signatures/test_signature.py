import os

import pytest
from eth_account import Account
from eth_account.datastructures import SignedMessage
from eth_account.messages import encode_defunct
from eth_typing import HexStr
from rizemind.authentication.signatures.signature import Signature
from web3 import Web3

# --------------------------------------------------------------------------- #
#                                FIXTURES                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def signed_message() -> SignedMessage:
    """
    Create a real ECDSA signature once per test session.

    We sign a simple defunct message with a randomly-generated key so that all
    tests share the same, valid `SignedMessage` instance.
    """
    acct = Account.create()
    message = encode_defunct(text="pytest-signature")
    return Account.sign_message(message, private_key=acct.key)


@pytest.fixture(scope="module")
def sig_bytes(signed_message: SignedMessage) -> bytes:
    """65-byte RSV signature (raw bytes)."""
    return signed_message.signature  # type: ignore[attr-defined]


@pytest.fixture(scope="module")
def rsv(signed_message: SignedMessage) -> tuple[HexStr, HexStr, int]:
    """Return hex-padded r, s and integer v suitable for Signature.from_rsv."""
    r_hex = Web3.to_hex(signed_message.r.to_bytes(32, "big"))
    s_hex = Web3.to_hex(signed_message.s.to_bytes(32, "big"))
    return r_hex, s_hex, signed_message.v


# --------------------------------------------------------------------------- #
#                              HAPPY-PATH TESTS                               #
# --------------------------------------------------------------------------- #


def test_signature_accepts_real_bytes(sig_bytes: bytes):
    sig = Signature(data=sig_bytes)
    assert sig.data == sig_bytes


def test_r_s_v_properties_match_signed_message(
    sig_bytes: bytes, signed_message: SignedMessage
):
    sig = Signature(data=sig_bytes)

    exp_r = Web3.to_hex(signed_message.r.to_bytes(32, "big")).lower()
    exp_s = Web3.to_hex(signed_message.s.to_bytes(32, "big")).lower()
    exp_v = signed_message.v

    assert sig.r == exp_r
    assert sig.s == exp_s
    assert sig.v == exp_v


def test_from_hex_roundtrip(sig_bytes: bytes, signed_message: SignedMessage):
    hex_str = Web3.to_hex(sig_bytes)
    sig = Signature.from_hex(hex_str)

    assert sig.data == sig_bytes
    assert sig.to_hex() == hex_str.lower()
    # Compare against original SignedMessage to guarantee consistency
    assert sig.to_hex() == Web3.to_hex(signed_message.signature).lower()


def test_from_rsv_constructs_identical_signature(rsv, sig_bytes: bytes):
    r_hex, s_hex, v = rsv
    sig = Signature.from_rsv(r_hex, s_hex, v)

    assert sig.data == sig_bytes
    # extra sanity: tuple layout equals eth-account expectation (v, r, s)
    assert sig.to_tuple() == (v, sig_bytes[:32], sig_bytes[32:64])


def test_to_tuple_and_str(sig_bytes: bytes, signed_message: SignedMessage):
    sig = Signature(data=sig_bytes)
    v, r_bytes, s_bytes = sig.to_tuple()

    assert v == signed_message.v
    assert r_bytes == signed_message.r.to_bytes(32, "big")
    assert s_bytes == signed_message.s.to_bytes(32, "big")
    # __str__ delegates to to_hex()
    assert str(sig) == sig.to_hex() == Web3.to_hex(sig_bytes).lower()


# --------------------------------------------------------------------------- #
#                              ERROR-PATH TESTS                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("length", [0, 1, 32, 64, 66, 100])
def test_invalid_signature_lengths_rejected(length: int):
    with pytest.raises(ValueError, match="Signature must be exactly 65 bytes"):
        Signature(data=os.urandom(length))


@pytest.mark.parametrize("bad_v", [0, 1, 29, 255])
def test_from_rsv_rejects_bad_v(rsv, bad_v: int):
    r_hex, s_hex, _ = rsv
    with pytest.raises(ValueError, match="v must be either 27 or 28"):
        Signature.from_rsv(r_hex, s_hex, bad_v)


def test_from_rsv_rejects_short_r(rsv):
    # Take padded s from fixture but truncate r to 30 bytes to trigger error
    _, s_hex, v = rsv
    short_r_hex = Web3.to_hex(b"20")
    with pytest.raises(ValueError, match="r and s must be 32 bytes each"):
        Signature.from_rsv(short_r_hex, s_hex, v)
