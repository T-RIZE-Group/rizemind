from typing import List, Tuple
import numpy as np
from web3 import Web3
from eth_account.messages import encode_typed_data
from eth_account import Account
from eth_account.account import VRS
from dataclasses import dataclass

@dataclass
class Parameters:
    """Model parameters."""
    tensors: List[bytes]
    tensor_type: str

def hash_weights(weights: np.ndarray) -> str:
    """
    Converts weights to float32 then bytes and hash them using Keccak-256.
    
    Args:
        weights (np.ndarray): A NumPy array of flattened model weights.

    Returns:
        str: The Keccak-256 hash of the model weights as a hex string.
    """
    weights_bytes = weights.astype(np.float32).tobytes()
    weights_hash = Web3.keccak(weights_bytes)

    return weights_hash.hex()

def hash_numpy_arrays(arrays: list[np.ndarray]) -> str:
    """
    Flatten the list of NumPy arrays and hash using Keccak-256.
    
    Args:
        arrays: A list of NumPy arrays.

    Returns:
        str: The Keccak-256 hash of the flattened arrays.
    """
    # Flatten and concatenate all arrays into a single 1D array
    weights = np.concatenate([arr.flatten() for arr in arrays])

    return hash_weights(weights)

def hash_tf_model(model) -> str:
    """
    Flatten the TensorFlow/Keras model weights and hash using keccak256.
    
    Args:
        model: A TensorFlow/Keras model.

    Returns:
        str: The Keccak-256 hash of the model weights.
    """
    # Get the model weights as a single, flat NumPy array
    weights = np.concatenate([layer.flatten() for layer in model.get_weights()])

    return hash_weights(weights)

def hash_torch_model(model) -> str:
    """
    Flatten the PyTorch model weights, convert to np.float32, and hash using keccak256..
    
    Args:
        model: A PyTorch model.

    Returns:
        str: The Keccak-256 hash of the model weights.
    """
    # Get the model weights as a single, flat NumPy array
    weights = np.concatenate([param.detach().numpy().flatten() for param in model.parameters()])

    weights = weights.astype(np.float32)

    return hash_weights(weights)


def prepare_eip712_domain(chainid: int, contract: str, name: str):
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
      "version": "1",
      "chainId": chainid, 
      "verifyingContract": contract
  }

def prepare_eip712_message(chainid: int, contract: str, name: str, round: int, hash: str):
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
  eip712_domain = prepare_eip712_domain(chainid, contract, name)
  eip712_message = {
    "types": {
        "EIP712Domain": [
            {"name": "name", "type": "string"},
            {"name": "version", "type": "string"},
            {"name": "chainId", "type": "uint256"},
            {"name": "verifyingContract", "type": "address"}
        ],
        "Model": [
            {"name": "round", "type": "uint256"},
            {"name": "hash", "type": "bytes32"}
        ]
    },
    "domain": eip712_domain,
    "primaryType": "Model",
    "message": {
        "round": round, 
        "hash": Web3.to_bytes(hexstr=hash)
    }
  }
  return encode_typed_data(
        full_message=eip712_message
    )

def sign_tf_model(account: Account, model, chainid: int, contract: str, name: str, round: int):
  """
  Signs a TensorFlow model's weights using the EIP-712 standard.

  Args:
      account (Account): An Ethereum account object from which the message will be signed.
      model: A TensorFlow model whose weights will be hashed.
      chainid (int): The ID of the blockchain network (e.g., 1 for Ethereum mainnet, 3 for Ropsten).
      contract (str): The address of the verifying contract in hexadecimal format.
      name (str): The human-readable name of the domain (e.g., "MyApp").
      round (int): The current round number of the model.

  Returns:
      dict: The signed message object containing the signature, message hash, and other metadata.

  Example:
      ```python
      signed_message = sign_tf_model(account, model, 1, "0x1234...", "MyApp", 1)
      ```
  """
  model_hash = hash_tf_model(model)
  eip712_message = prepare_eip712_message(chainid, contract, name, round, model_hash)
  return account.sign_message(eip712_message)

def sign_torch_model(account: Account, model, chainid: int, contract: str, name: str, round: int):
  """
  Signs a PyTorch model's weights using the EIP-712 standard.

  Args:
    account (Account): An Ethereum account object from which the message will be signed.
    model: A TensorFlow model whose weights will be hashed.
    chainid (int): The ID of the blockchain network (e.g., 1 for Ethereum mainnet, 3 for Ropsten).
    contract (str): The address of the verifying contract in hexadecimal format.
    name (str): The human-readable name of the domain (e.g., "MyApp").
    round (int): The current round number of the model.

  Returns:
    dict: SignedMessage from eth_account

  Example:
    ```python
    signed_message = sign_tf_model(account, model, 1, "0x1234...", "MyApp", 1)
    ```
  """
  model_hash = hash_torch_model(model)
  eip712_message = prepare_eip712_message(chainid, contract, name, round, model_hash)
  return account.sign_message(eip712_message)

def hash_parameters(parameters: Parameters) -> bytes:
    """
    Hashes the Parameters dataclass using keccak256.

    Args:
        parameters (Parameters): The model parameters to hash.

    Returns:
        bytes: The keccak256 hash of the concatenated tensors and tensor type.
    """
    # Concatenate tensors and tensor type for hashing
    data = b''.join(parameters.tensors) + parameters.tensor_type.encode()
    return Web3.keccak(data).hex()

def sign_parameters_model(account: Account, parameters: Parameters, chainid: int, contract: str, name: str, round: int):
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
    eip712_message = prepare_eip712_message(chainid, contract, name, round, parameters_hash)
    return account.sign_message(eip712_message)

def recover_model_signer(model: list[np.ndarray], chainid: int, contract: str, name: str, round: int, signature: Tuple[VRS, VRS, VRS]):
   """
   Recover the address of the signed model.

   Returns:
    str: hex address of the signer.
   """
   model_hash = hash_numpy_arrays(model)
   message = prepare_eip712_message(chainid, contract, name, round, model_hash)
   return Account.recover_message(message, signature)