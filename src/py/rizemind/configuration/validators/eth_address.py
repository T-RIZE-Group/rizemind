from typing import Annotated, Any

from eth_typing import ChecksumAddress
from pydantic import BeforeValidator
from pydantic.functional_validators import AfterValidator
from web3 import Web3


def _validate_eth_address(addr: str) -> ChecksumAddress:
    """
    Ensure `addr` is a valid Ethereum address and
    always return it in EIP-55 checksum form.
    """
    if not isinstance(addr, str):
        raise TypeError("Address must be a string")

    if not Web3.is_address(addr):
        raise ValueError("Invalid Ethereum address")

    return Web3.to_checksum_address(addr)


EthereumAddress = Annotated[str, AfterValidator(_validate_eth_address)]


def to_checksum_or_none(v: Any) -> ChecksumAddress | None:
    if v is None:
        return None
    if not isinstance(v, str):
        return None
    if not Web3.is_address(v):
        return None
    return Web3.to_checksum_address(v)


EthereumAddressOrNone = Annotated[str | None, BeforeValidator(to_checksum_or_none)]
