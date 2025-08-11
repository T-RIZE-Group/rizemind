import os
from unittest.mock import Mock, patch

import pytest
from eth_account import Account
from eth_account.signers.base import BaseAccount
from eth_tester import EthereumTester
from eth_tester.backends.mock import MockBackend
from flwr.common import ConfigRecord, Context, RecordDict
from flwr.common.typing import UserConfig
from rizemind.authentication.config import AccountConfig
from rizemind.authentication.signatures.signature import Signature
from rizemind.contracts.erc.erc5267.typings import EIP712DomainMinimal
from rizemind.web3.config import Web3Config
from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider


@pytest.fixture
def mnemonic() -> str:
    return "test test test test test test test test test test test junk"


@pytest.fixture
def account(mnemonic) -> BaseAccount:
    """Create an account for testing."""
    Account.enable_unaudited_hdwallet_features()
    return Account.from_mnemonic(
        mnemonic,
        account_path="m/44'/60'/0'/0/0",
    )


@pytest.fixture
def account_config(mnemonic):
    """Create an AccountConfig instance."""
    return AccountConfig(
        mnemonic=mnemonic,
        default_account_index=0,
    )


@pytest.fixture
def minimal_domain():
    """Create a sample EIP712 domain."""
    return EIP712DomainMinimal(
        name="Test Swarm",
        version="1.0.0",
        chainId=1,
        verifyingContract=Web3.to_checksum_address(
            "0x1234567890123456789012345678901234567890"
        ),
    )


@pytest.fixture
def context() -> Context:
    return Context(
        run_id=1,
        node_id=1,
        node_config=UserConfig(),
        state=RecordDict(),
        run_config=UserConfig(),
    )


@pytest.fixture
def random_signature() -> Signature:
    return Signature(data=os.urandom(65))


@pytest.fixture(scope="session")
def eth_tester():
    return EthereumTester(backend=MockBackend())


@pytest.fixture(scope="session")
def w3(eth_tester):
    provider = EthereumTesterProvider(eth_tester)
    w3 = Web3(provider)
    return w3


@pytest.fixture(scope="session")
def accounts(w3):
    return w3.eth.accounts


@pytest.fixture
def web3_config(w3: Web3):
    """Create a mock Web3Config that returns the w3 instance."""

    mock_config = Mock(spec=Web3Config)
    mock_config.get_web3.return_value = w3
    mock_config.to_config_record.return_value = ConfigRecord({})
    with patch("rizemind.web3.config.Web3Config", return_value=mock_config):
        yield mock_config
