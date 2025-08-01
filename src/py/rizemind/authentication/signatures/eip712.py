from dataclasses import dataclass
from typing import Any, Protocol

from eth_account.messages import encode_typed_data
from eth_typing import ChecksumAddress


class EIP712DomainRequiredFields(Protocol):
    name: str
    version: str
    chainId: int
    verifyingContract: ChecksumAddress


@dataclass
class EIP712DomainStruct:
    name: str
    version: str
    chainId: int
    verifyingContract: ChecksumAddress


EIP712DomainTypeName = "EIP712Domain"

EIP712DomainABI = [
    {"name": "name", "type": "string"},
    {"name": "version", "type": "string"},
    {"name": "chainId", "type": "uint256"},
    {"name": "verifyingContract", "type": "address"},
]


def prepare_eip712_domain(
    chainid: int, version: str, contract: ChecksumAddress, name: str
) -> EIP712DomainRequiredFields:
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

    return EIP712DomainStruct(
        name=name,
        version=version,
        chainId=chainid,
        verifyingContract=contract,
    )


def domain_to_dict(domain: EIP712DomainRequiredFields) -> dict[str, Any]:
    return {
        "name": domain.name,
        "version": domain.version,
        "chainId": domain.chainId,
        "verifyingContract": domain.verifyingContract,
    }


def prepare_eip712_message(
    eip712_domain: EIP712DomainRequiredFields,
    primaryType: str,
    message: dict,
    types: dict = {},
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
    eip712_message = {
        "types": {
            EIP712DomainTypeName: EIP712DomainABI,
        }
        | types,
        "domain": domain_to_dict(eip712_domain),
        "primaryType": primaryType,
        "message": message,
    }
    return encode_typed_data(full_message=eip712_message)
