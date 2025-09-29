from dataclasses import dataclass

import pytest
from pydantic import HttpUrl
from rizemind.authentication import AccountConfig
from rizemind.web3 import Web3Config

from .forge_helper import start_anvil


@dataclass
class AnvilContext:
    w3conf: Web3Config
    account_conf: AccountConfig


MNEMONIC = "test test test test test test test test test test test junk"


@pytest.fixture(scope="session")
def anvil():
    proc = start_anvil(mnemonic=MNEMONIC)
    w3conf = Web3Config(url=HttpUrl(url="http://127.0.0.1:8545"))
    account_conf = AccountConfig(mnemonic=MNEMONIC)
    yield AnvilContext(w3conf=w3conf, account_conf=account_conf)
    proc.terminate()
