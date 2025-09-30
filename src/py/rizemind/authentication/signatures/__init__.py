"""A collection of utilities for handling cryptographic signatures.

This module provides tools for creating, managing, and verifying digital signatures
focused on Ethereum-based authentication mechanisms and EIP-712 structured data signing.
"""

from rizemind.authentication.signatures.auth import (
    recover_auth_signer,
    sign_auth_message,
)
from rizemind.authentication.signatures.eip712 import (
    domain_to_dict,
    prepare_eip712_domain,
    prepare_eip712_message,
)
from rizemind.authentication.signatures.signature import Signature

__all__ = [
    "sign_auth_message",
    "recover_auth_signer",
    "prepare_eip712_domain",
    "domain_to_dict",
    "prepare_eip712_message",
    "Signature",
]
