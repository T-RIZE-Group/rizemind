from dataclasses import dataclass
from typing import Any, Protocol

from eth_account.messages import SignableMessage, encode_typed_data
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


@dataclass
class EIP712DomainStruct:
    """Dataclass implementation of EIP-712 domain fields.

    This class provides a concrete implementation of the EIP-712 domain
    separator structure, containing all required fields for creating
    typed structured data signatures according to the EIP-712 standard.

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
    """Prepares the EIP-712 domain object for signing typed structured data.

    Args:
        chainid: The ID of the blockchain network (e.g., 1 for Ethereum mainnet, 3 for Ropsten).
        version: The current version of the domain.
        contract: The address of the verifying contract in hexadecimal format (e.g., "0xCcCCc...").
        name: The human-readable name of the domain (e.g., "MyApp").

    Returns:
        An object representing the EIP-712 domain.
    """

    return EIP712DomainStruct(
        name=name,
        version=version,
        chainId=chainid,
        verifyingContract=contract,
    )


def domain_to_dict(domain: EIP712DomainRequiredFields) -> dict[str, Any]:
    """Changes the type of the domain to python dictionary

    Args:
        domain: The `EIP712DomainRequiredFields` representation of the EIP-712 domain.

    Returns:
        The dictionary representation of the EIP-712 domain.
    """
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
) -> SignableMessage:
    """Prepares the EIP-712 structured message for signing and encoding using the provided parameters.

    Args:
        eip712_domain: The EIP-712 domain object.
        primaryType: The name of the primary message type.
        message: The structured data to be signed.
        types: A dictionary defining the custom data types used in the message,
        where keys are type names and values are their ABI definitions.

    Returns:
        An SignableMessage object ready for signing.
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
