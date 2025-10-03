import os
from unittest.mock import MagicMock

import pytest
from eth_account import Account
from eth_typing import ChecksumAddress
from flwr.common import GetPropertiesRes
from flwr.common.typing import Code, Status
from rizemind.authentication import AuthenticatedClientProperties
from rizemind.authentication.can_train_criterion import CanTrainCriterion
from rizemind.authentication.signatures.auth import sign_auth_message
from rizemind.authentication.train_auth import (
    TRAIN_AUTH_PREFIX,
    prepare_train_auth_ins,
    prepare_train_auth_res,
)
from rizemind.contracts.erc.erc5267.typings import EIP712Domain
from web3 import Web3


# --------------------------------------------------------------------------- #
# Test scaffolding
# --------------------------------------------------------------------------- #
class _DummySwarm:
    """Minimal stub of the on-chain Swarm contract façade."""

    last_call: tuple[ChecksumAddress, int] | None

    def __init__(self, domain: EIP712Domain, result: bool) -> None:
        self._domain = domain
        self._result = result
        self.last_call = None  # (signer, round_id)

    # API expected by CanTrainCriterion
    def get_eip712_domain(self) -> EIP712Domain:
        return self._domain

    def can_train(self, trainer: ChecksumAddress, round_id: int) -> bool:
        self.last_call = (trainer, round_id)
        return self._result

    def can_evaluate(self, evaluator: ChecksumAddress, round_id: int) -> bool:
        raise NotImplementedError()


@pytest.fixture(autouse=True)
def nonce(monkeypatch) -> bytes:
    """
    Force `os.urandom` to return a reproducible 32-byte nonce so the test can
    pre-compute the very same instruction object that the SUT will generate.
    """
    nonce = b"\x01" * 32
    monkeypatch.setattr(os, "urandom", lambda n: nonce)
    return nonce


@pytest.fixture
def domain() -> EIP712Domain:
    return EIP712Domain(
        name="TestDomain",
        version="1",
        chainId=1,
        verifyingContract=Web3.to_checksum_address(
            "0xCcCCccccCCCCcCCCCCCcCcCccCcCCCcCcccccccC"
        ),
        fields=b"02",
        salt=b"03",
        extensions=[],
    )


@pytest.fixture
def signer_account():
    return Account.create()


@pytest.mark.parametrize("swarm_allows_training", [True, False])
def test_select_propagates_swarm_decision(
    swarm_allows_training: bool,
    nonce: bytes,
    domain: EIP712Domain,
    signer_account,
):
    rnd = 42
    swarm = _DummySwarm(domain, result=swarm_allows_training)

    client = MagicMock(spec_set=["properties"])
    client.properties = {}
    auth_prop = AuthenticatedClientProperties(trainer_address=signer_account.address)
    auth_prop.tag_client(client)
    criterion = CanTrainCriterion(round_id=rnd, swarm=swarm)
    outcome = criterion.select(client)

    assert outcome is swarm_allows_training

    # `swarm.can_train` should have been invoked with the *recovered* signer
    # and the same round id
    assert swarm.last_call is not None
    signer, called_round = swarm.last_call
    assert called_round == rnd
    assert signer == signer_account.address


def test_select_returns_false_when_parsing_exception(domain: EIP712Domain):
    """
    If the Flower client returns a malformed GetPropertiesRes such that
    `parse_train_auth_res` raises, `select` must *not* propagate the exception
    but instead return False.
    """
    rnd = 13

    swarm = _DummySwarm(domain, result=True)

    client = MagicMock(spec_set=["properties"])
    client.properties = {}

    criterion = CanTrainCriterion(round_id=rnd, swarm=swarm)
    outcome = criterion.select(client)

    assert outcome is False, "select() should return False on parse failures"
    assert swarm.last_call is None, (
        "`swarm.can_train` should not be called when parsing fails"
    )
