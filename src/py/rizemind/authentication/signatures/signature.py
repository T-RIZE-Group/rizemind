from typing import Self

from eth_typing import HexStr
from pydantic import BaseModel, Field, field_validator
from web3 import Web3


class Signature(BaseModel):
    """Represents an ECDSA signature for Ethereum.

    This model provides a structured way to handle the 65-byte signature format,
    which is a concatenation of the r, s, and v components. It includes
    properties to access individual components and class methods for convenient
    instantiation from different formats.

    Attributes:
        data: The raw 65-byte signature, concatenated as r (32 bytes) +
              s (32 bytes) + v (1 byte).
    """

    data: bytes = Field(..., description="65-byte signature (r + s + v)")

    @field_validator("data")
    @classmethod
    def validate_signature_length(cls, v: bytes) -> bytes:
        """Pydantic validator to ensure the signature data is exactly 65 bytes long.

        Args:
            v: The input byte string to validate.

        Returns:
            The validated 65-byte string.

        Raises:
            ValueError: If the length of `v` is not 65.
        """
        if len(v) != 65:
            raise ValueError("Signature must be exactly 65 bytes")
        return v

    @property
    def r(self) -> HexStr:
        """The 'r' value of the ECDSA signature.

        Returns:
            The first 32 bytes of the signature as a `HexStr`.
        """
        return Web3.to_hex(self.data[:32])

    @property
    def s(self) -> HexStr:
        """The 's' value of the ECDSA signature.

        Returns:
            The middle 32 bytes (bytes 32-64) of the signature as a `HexStr`.
        """
        return Web3.to_hex(self.data[32:64])

    @property
    def v(self) -> int:
        """The 'v' value (recovery identifier) of the ECDSA signature.

        Returns:
            The last byte (65th) of the signature as an integer.
        """
        return self.data[64]

    @classmethod
    def from_hex(cls, signature: HexStr) -> Self:
        """Creates a `Signature` instance from a hexadecimal string.

        Args:
            signature: The 65-byte signature as a hex string (e.g., '0x...').

        Returns:
            A new `Signature` instance.
        """
        return cls(data=Web3.to_bytes(hexstr=signature))

    @classmethod
    def from_rsv(cls, r: HexStr, s: HexStr, v: int) -> Self:
        """Creates a `Signature` instance from its r, s, and v components.

        Args:
            r: The 'r' value as a 32-byte hex string.
            s: The 's' value as a 32-byte hex string.
            v: The 'v' value (recovery ID), must be 27 or 28.

        Returns:
            A new `Signature` instance.

        Raises:
            ValueError: If `v` is not 27 or 28, or if `r` or `s` are not
                        32 bytes each after conversion from hex.
        """
        if v not in (27, 28):
            raise ValueError("v must be either 27 or 28")

        r_bytes = Web3.to_bytes(hexstr=r)
        s_bytes = Web3.to_bytes(hexstr=s)

        if len(r_bytes) != 32 or len(s_bytes) != 32:
            raise ValueError("r and s must be 32 bytes each")

        return cls(data=r_bytes + s_bytes + bytes([v]))

    def to_tuple(self) -> tuple[int, bytes, bytes]:
        """Converts the signature to a tuple of (v, r, s).

        This format is commonly used by Ethereum libraries like `eth-account` for
        transaction signing and public key recovery.

        Returns:
            A tuple containing the signature components: (v, r_bytes, s_bytes).
        """
        return (
            self.v,
            self.data[:32],  # r
            self.data[32:64],  # s
        )

    def to_hex(self) -> HexStr:
        """The full 65-byte signature as a hex string.

        Returns:
            The signature as a `HexStr` (e.g., '0x...').
        """
        return Web3.to_hex(self.data)

    def __str__(self) -> str:
        """The string representation of the signature, which is its hex format.

        Returns:
            The signature as a hex string.
        """
        return self.to_hex()
