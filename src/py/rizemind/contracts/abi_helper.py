import json
from pathlib import Path
from typing import Any

from web3 import Web3

from rizemind.exception.base_exception import RizemindException


class AbiNotFoundError(RizemindException):
    def __init__(self, path: Path):
        super().__init__(code="abi_not_found", message=f"ABI not found: {path}")


type Abi = Any


def load_abi(path: Path) -> Abi:
    """
    Load and validate an ABI
    """
    if not path.exists():
        raise AbiNotFoundError(path)

    with open(path, encoding="utf-8") as f:
        abi = json.load(f)

    Web3().eth.contract(abi=abi)

    return abi
