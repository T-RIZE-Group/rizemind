from eth_account import Account
from eth_account.signers.base import BaseAccount
from eth_typing import ChecksumAddress
from flwr.common.typing import Parameters
from web3 import Web3

from rizemind.authentication.signatures.eip712 import (
    EIP712DomainRequiredFields,
    prepare_eip712_message,
)
from rizemind.authentication.signatures.signature import Signature

ModelTypeName = "Model"
ModelTypeAbi = [
    {"name": "round", "type": "uint256"},
    {"name": "hash", "type": "bytes32"},
]


def hash_parameters(parameters: Parameters) -> bytes:
    """Hashes the Parameters dataclass using keccak256.

    Args:
        parameters: The model parameters to hash.

    Returns:
        The keccak256 hash of the concatenated tensors and tensor type.
    """
    # Concatenate tensors and tensor type for hashing
    data = b"".join(parameters.tensors) + parameters.tensor_type.encode()
    return Web3.keccak(data)


def sign_parameters_model(
    *,
    parameters: Parameters,
    round: int,
    domain: EIP712DomainRequiredFields,
    account: BaseAccount,
) -> Signature:
    """Signs a model's parameters using the EIP-712 standard.

    @TODO -> requires double checking with domain
    Args:
        account: An Ethereum account object from which the message will be signed.
        parameters: The model parameters to sign.
        domain: The EIP712 required fields.
        round: The current round number of the federated learning.

    Returns:
        The `SignedMessage` from eth_account
    """
    parameters_hash = hash_parameters(parameters)
    eip712_message = prepare_eip712_message(
        domain,
        ModelTypeName,
        {"round": round, "hash": parameters_hash},
        {ModelTypeName: ModelTypeAbi},
    )
    signature = account.sign_message(eip712_message)
    return Signature(data=signature.signature)


def recover_model_signer(
    *,
    model: Parameters,
    round: int,
    domain: EIP712DomainRequiredFields,
    signature: Signature,
) -> ChecksumAddress:
    """Recover the address of the signed model.

    Args:
        model: The model's parameters.
        round: The current round number of the federated learning.
        domain: The EIP712 required fields.
        signature: The signature of the trainer/aggregator that sent the parameters.

    Returns:
        The hex address of the signer.
    """
    model_hash = hash_parameters(model)
    eip712_message = prepare_eip712_message(
        domain,
        ModelTypeName,
        {"round": round, "hash": model_hash},
        {ModelTypeName: ModelTypeAbi},
    )
    return Web3.to_checksum_address(
        Account.recover_message(eip712_message, signature=signature.data)
    )
