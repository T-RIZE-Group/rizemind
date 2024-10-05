import numpy as np
from web3 import Web3
from eth_account.messages import encode_typed_data
from eth_account import Account

def hash_weights(weights: np.ndarray) -> str:
    """
    Convert the flattened weights to bytes and sign them using Keccak-256.
    
    Args:
        weights (np.ndarray): A NumPy array of flattened model weights.

    Returns:
        str: The Keccak-256 hash of the model weights as a hex string.
    """
    # Convert the NumPy array to bytes
    weights_bytes = weights.tobytes()

    # Hash the weights using Keccak-256
    weights_hash = Web3.keccak(weights_bytes)

    return weights_hash.hex()

def hash_tf_model(model) -> str:
    """
    Flatten the TensorFlow/Keras model weights, convert to np.float32, and sign them.
    
    Args:
        model: A TensorFlow/Keras model.

    Returns:
        str: The Keccak-256 hash of the model weights.
    """
    # Get the model weights as a single, flat NumPy array
    weights = np.concatenate([layer.flatten() for layer in model.get_weights()])

    # Ensure consistent precision (e.g., 32-bit floats)
    weights = weights.astype(np.float32)

    # Sign the weights using Keccak-256
    return hash_weights(weights)

def hash_torch_model(model) -> str:
    """
    Flatten the PyTorch model weights, convert to np.float32, and sign them.
    
    Args:
        model: A PyTorch model.

    Returns:
        str: The Keccak-256 hash of the model weights.
    """
    # Get the model weights as a single, flat NumPy array
    weights = np.concatenate([param.detach().numpy().flatten() for param in model.parameters()])

    # Ensure consistent precision (e.g., 32-bit floats)
    weights = weights.astype(np.float32)

    # Sign the weights using Keccak-256
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
      print(signed_message.signature)
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
    dict: The signed message object containing the signature, message hash, and other metadata.

  Example:
    ```python
    signed_message = sign_tf_model(account, model, 1, "0x1234...", "MyApp", 1)
    print(signed_message.signature)
    ```
  """
  model_hash = hash_torch_model(model)
  eip712_message = prepare_eip712_message(chainid, contract, name, round, model_hash)
  return account.sign_message(eip712_message)


