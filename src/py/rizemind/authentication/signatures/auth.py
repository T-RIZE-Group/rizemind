from eth_account import Account
from eth_account.signers.base import BaseAccount
from eth_typing import ChecksumAddress
from rizemind.authentication.signatures.eip712 import (
    EIP712DomainRequiredFields,
    prepare_eip712_message,
)
from rizemind.authentication.signatures.signature import Signature
from web3 import Web3

AuthTypeName = "Auth"
AuthTypeAbi = [
    {"name": "round", "type": "uint256"},
    {"name": "nonce", "type": "bytes32"},
]


def sign_auth_message(
    *,
    account: BaseAccount,
    round: int,
    nonce: bytes,
    domain: EIP712DomainRequiredFields,
) -> Signature:
    """
    Signs an authentication message using the EIP-712 standard.

    Args:
        account (BaseAccount): The account that will sign the message
        version (str): The version string for the domain
        round (int): The current round number
        nonce (HexStr): A unique nonce for this authentication
        chainid (int): The blockchain network ID
        contract (ChecksumAddress): The verifying contract address
        name (str): The domain name

    Returns:
        SignedMessage: The signed authentication message
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
    """
    Recovers the address that signed an authentication message.

    Args:
        version (str): The version string for the domain
        round (int): The round number from the signed message
        nonce (bytes): The nonce from the signed message
        chainid (int): The blockchain network ID
        contract (ChecksumAddress): The verifying contract address
        name (str): The domain name
        signature (tuple): The signature components (v, r, s)

    Returns:
        ChecksumAddress: The address that signed the message
    """
    eip712_message = prepare_eip712_message(
        domain,
        AuthTypeName,
        {"round": round, "nonce": nonce},
        {AuthTypeName: AuthTypeAbi},
    )
    signer = Account.recover_message(eip712_message, signature=signature.data)
    return Web3.to_checksum_address(signer)
