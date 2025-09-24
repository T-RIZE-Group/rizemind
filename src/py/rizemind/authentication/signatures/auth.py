from eth_account import Account
from eth_account.signers.base import BaseAccount
from eth_typing import ChecksumAddress
from web3 import Web3

from rizemind.authentication.signatures.eip712 import (
    EIP712DomainRequiredFields,
    prepare_eip712_message,
)
from rizemind.authentication.signatures.signature import Signature

AuthTypeName = "Auth"
AuthTypeAbi = [
    {"name": "round", "type": "uint256"},
    {"name": "nonce", "type": "bytes32"},
]


def sign_auth_message(
    *,
    round: int,
    nonce: bytes,
    domain: EIP712DomainRequiredFields,
    account: BaseAccount,
) -> Signature:
    """Signs an authentication message using the EIP-712 standard.

    Args:
        round: The current round number.
        nonce: A unique nonce for this authentication.
        domain: The EIP712 required fields.
        account: The account that will sign the message.

    Returns:
        The signed authentication message.
    """
    eip712_message = prepare_eip712_message(
        domain,
        AuthTypeName,
        {"round": round, "nonce": nonce},
        {AuthTypeName: AuthTypeAbi},
    )
    signature = account.sign_message(eip712_message)
    return Signature(data=signature.signature)


def recover_auth_signer(
    *,
    round: int,
    nonce: bytes,
    domain: EIP712DomainRequiredFields,
    signature: Signature,
) -> ChecksumAddress:
    """Recovers the address that signed an authentication message.

    Args:
        round: The round number from the signed message.
        nonce: The nonce from the signed message.
        domain: The EIP712 required fields.
        signature: The signature of the message.

    Returns:
        The address that signed the message.
    """
    eip712_message = prepare_eip712_message(
        domain,
        AuthTypeName,
        {"round": round, "nonce": nonce},
        {AuthTypeName: AuthTypeAbi},
    )
    signer = Account.recover_message(eip712_message, signature=signature.data)
    return Web3.to_checksum_address(signer)
