from typing import Tuple
from eth_typing import HexAddress
from eth_typing import HexStr
from web3 import Web3
from eth_account.messages import encode_typed_data
from eth_account import Account
from eth_account.signers.base import BaseAccount
from eth_account.datastructures import SignedMessage
from typing import TypedDict
from flwr.common.typing import Parameters, Scalar


class EIP712DomainAttrib(TypedDict):
    name: str
    version: str
    chainId: int
    verifyingContract: str


def prepare_eip712_domain(
    chainid: int, version: str, contract: str, name: str
) -> EIP712DomainAttrib:
    """
    Prepares the EIP-712 domain object for signing typed structured data.

    Args:
        chainid (int): The ID of the blockchain network (e.g., 1 for Ethereum mainnet, 3 for Ropsten).
        contract (str): The address of the verifying contract in hexadecimal format (e.g., "0xCcCCc...").
        name (str): The human-readable name of the domain (e.g., "MyApp").

    Returns:
        dict: A dictionary representing the EIP-712 domain object with the following keys:
              - "name": The human-readable name of the domain.
              - "version": The version of the domain, which is always set to "1".
              - "chainId": The ID of the blockchain network.
              - "verifyingContract": The address of the contract that verifies the signature.
    """

    return {
        "name": name,
        "version": version,
        "chainId": chainid,
        "verifyingContract": contract,
    }


def prepare_eip712_message(
    chainid: int, version: str, contract: str, name: str, round: int, hash: str
):
    """
    Prepares the EIP-712 structured message for signing and encoding using the provided parameters.

    Args:
        chainid (int): The ID of the blockchain network (e.g., 1 for Ethereum mainnet, 3 for Ropsten).
        contract (str): The address of the verifying contract in hexadecimal format (e.g., "0xCcCCc...").
        name (str): The human-readable name of the domain (e.g., the app or contract name).
        round (int): The current round number of the model.
        hash (str): The model hash, provided as a hexadecimal string, representing a bytes32 hash.

    Returns:
        dict: A dictionary representing the EIP-712 structured message, ready for signing.
              The message includes:
              - `domain`: The EIP-712 domain object.
              - `types`: The type definitions for the domain and message fields.
              - `primaryType`: The primary data type being signed, which is "Model".
              - `message`: The actual message containing the round and the model hash.
    """
    eip712_domain = prepare_eip712_domain(chainid, version, contract, name)
    eip712_message = {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
            ],
            "Model": [
                {"name": "round", "type": "uint256"},
                {"name": "hash", "type": "bytes32"},
            ],
        },
        "domain": eip712_domain,
        "primaryType": "Model",
        "message": {
            "round": round,
            "hash": Web3.to_bytes(hexstr=HexStr(hash)),
        },
    }
    return encode_typed_data(full_message=eip712_message)


def hash_parameters(parameters: Parameters) -> str:
    """
    Hashes the Parameters dataclass using keccak256.

    Args:
        parameters (Parameters): The model parameters to hash.

    Returns:
        bytes: The keccak256 hash of the concatenated tensors and tensor type.
    """
    # Concatenate tensors and tensor type for hashing
    data = b"".join(parameters.tensors) + parameters.tensor_type.encode()
    return Web3.keccak(data).hex()


def sign_parameters_model(
    account: BaseAccount,
    version: str,
    parameters: Parameters,
    chainid: int,
    contract: str,
    name: str,
    round: int,
) -> SignedMessage:
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
        chainid, version, contract, name, round, parameters_hash
    )
    return account.sign_message(eip712_message)


def recover_model_signer(
    model: Parameters,
    version: str,
    chainid: int,
    contract: str,
    name: str,
    round: int,
    signature: Tuple[Scalar, Scalar, Scalar],
) -> HexAddress:
    """
    Recover the address of the signed model.

    Returns:
     str: hex address of the signer.
    """
    model_hash = hash_parameters(model)
    message = prepare_eip712_message(
        chainid, version, contract, name, round, model_hash
    )
    return Account.recover_message(message, signature)
