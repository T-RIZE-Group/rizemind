from typing import Self

from eth_typing import HexStr
from pydantic import BaseModel, Field, field_validator
from web3 import Web3


class Signature(BaseModel):
    """
    A Pydantic model for Ethereum signatures stored as a single 65-byte value.
    Contains methods to access the r, s, v components.

    signature: 65 bytes concatenated RSV (r[32] + s[32] + v[1])
    """

    data: bytes = Field(..., description="65-byte signature (r + s + v)")

    @field_validator("data")
    @classmethod
    def validate_signature_length(cls, v: bytes) -> bytes:
        """Validate that the signature is exactly 65 bytes."""
        if len(v) != 65:
            raise ValueError("Signature must be exactly 65 bytes")
        return v

    @property
    def r(self) -> HexStr:
        """First 32 bytes of the signature."""
        return Web3.to_hex(self.data[:32])

    @property
    def s(self) -> HexStr:
        """Middle 32 bytes of the signature."""
        return Web3.to_hex(self.data[32:64])

    @property
    def v(self) -> int:
        """Last byte of the signature."""
        return self.data[64]

    @classmethod
    def from_hex(cls, signature: HexStr) -> Self:
        """Create a Signature from a hex string."""
        return cls(data=Web3.to_bytes(hexstr=signature))

    @classmethod
    def from_rsv(cls, r: HexStr, s: HexStr, v: int) -> Self:
        """Create a Signature from r, s, v components."""
        if v not in (27, 28):
            raise ValueError("v must be either 27 or 28")

        r_bytes = Web3.to_bytes(hexstr=r)
        s_bytes = Web3.to_bytes(hexstr=s)

        if len(r_bytes) != 32 or len(s_bytes) != 32:
            raise ValueError("r and s must be 32 bytes each")

        return cls(data=r_bytes + s_bytes + bytes([v]))

    def to_tuple(self) -> tuple[int, bytes, bytes]:
        """Convert to tuple format (v, r, s) used by eth_account."""
        return (
            self.v,
            self.data[:32],  # r
            self.data[32:64],  # s
        )

    def to_hex(self) -> HexStr:
        """Convert to hex string format."""
        return Web3.to_hex(self.data)

    def __str__(self) -> str:
        """String representation of the signature in hex format."""
        return self.to_hex()
