from eth_account import Account
from eth_account.signers.base import BaseAccount
from eth_typing import ChecksumAddress
from flwr.common.typing import Parameters
from rizemind.authentication.signatures.eip712 import (
    EIP712DomainRequiredFields,
    prepare_eip712_message,
)
from rizemind.authentication.signatures.signature import Signature
from web3 import Web3

ModelTypeName = "Model"
ModelTypeAbi = [
    {"name": "round", "type": "uint256"},
    {"name": "hash", "type": "bytes32"},
]


def hash_parameters(parameters: Parameters) -> bytes:
    """
    Hashes the Parameters dataclass using keccak256.

    Args:
        parameters (Parameters): The model parameters to hash.

    Returns:
        bytes: The keccak256 hash of the concatenated tensors and tensor type.
    """
    # Concatenate tensors and tensor type for hashing
    data = b"".join(parameters.tensors) + parameters.tensor_type.encode()
    return Web3.keccak(data)


def sign_parameters_model(
    *,
    account: BaseAccount,
    parameters: Parameters,
    round: int,
    domain: EIP712DomainRequiredFields,
) -> Signature:
    """
    Signs a model's parameters using the EIP-712 standard.

    Args:
        account (Account): An Ethereum account object from which the message will be signed.
        parameters (Parameters): The model parameters to sign.
        chainid (int): The ID of the blockchain network.
        contract (str): The address of the verifying contract in hexadecimal format.
        name (str): The human-readable name of the domain.
        round (int): The current round number of the model.

    Returns:
        dict: SignedMessage from eth_account
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
    """
    Recover the address of the signed model.

    Returns:
     str: hex address of the signer.
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
