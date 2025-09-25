from typing import Annotated

from eth_typing import ChecksumAddress
from pydantic.functional_validators import AfterValidator
from web3 import Web3


def _validate_eth_address(addr: str) -> ChecksumAddress:
    """Ensure `addr` is a valid Ethereum address and always return it in EIP-55 checksum form.

    Raises:
        TypeError: If the addr is not a string.
        ValueError: If the string is not an Ethereum address.
    """
    if not isinstance(addr, str):
        raise TypeError("Address must be a string")

    if not Web3.is_address(addr):
        raise ValueError("Invalid Ethereum address")

    return Web3.to_checksum_address(addr)


EthereumAddress = Annotated[str, AfterValidator(_validate_eth_address)]
"""A validated Ethereum address type.

This type alias represents a string that has been validated to be a proper
Ethereum address and is automatically converted to EIP-55 checksum format.
When used in Pydantic models or function parameters, any string input will
be validated and normalized to the standard checksum format.

The validation ensures that:
- The input is a string type
- The string represents a valid Ethereum address (20 bytes, hex-encoded)
- The address is returned in EIP-55 checksum format for consistency

Example:
    >>> from rizemind.configuration.validators.eth_address import EthereumAddress
    >>> # This will validate and convert to checksum format
    >>> address: EthereumAddress = "0xd8da6bf26964af9d7eed9e03e53415d37aa96045"
    >>> # Result: "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"

Raises:
    TypeError: If the input is not a string.
    ValueError: If the string is not a valid Ethereum address.
"""
