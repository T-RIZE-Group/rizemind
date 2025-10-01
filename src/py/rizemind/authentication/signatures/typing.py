from typing import Protocol

from eth_typing import ChecksumAddress


class EIP712DomainRequiredFields(Protocol):
    """Protocol defining the required fields for an EIP-712 domain.

    This protocol specifies the mandatory fields that must be present in any
    EIP-712 domain implementation to ensure compatibility with the EIP-712
    standard for typed structured data hashing and signing.

    Attributes:
        name: A human-readable name for the domain. This is typically
        the name of the DApp or protocol.
        version: The current version of the domain.
        chainId: The EIP-155 chain ID of the network where the contract is
        deployed.
        verifyingContract: The Ethereum address of the contract that will
        verify the signature.
    """

    name: str
    version: str
    chainId: int
    verifyingContract: ChecksumAddress
