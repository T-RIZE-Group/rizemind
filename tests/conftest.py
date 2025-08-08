from eth_account.signers.base import BaseAccount
from flwr.common import Context, RecordDict
from flwr.common.typing import UserConfig
import pytest
from eth_account import Account
from rizemind.authentication.config import AccountConfig
from rizemind.contracts.erc.erc5267.typings import EIP712DomainMinimal
from web3 import Web3


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
