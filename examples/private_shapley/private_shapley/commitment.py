# private_shapley/commitment.py
import hashlib
import secrets
from typing import Tuple


def generate_nonce() -> str:
    """Generate a secure random nonce as a hex string."""
    return secrets.token_hex(16)  # 128-bit secure random value


def create_commitment(trainer_address: str, nonce: str, round_id: int) -> str:
    """
    Create a commitment for a trainer in a specific round.

    Args:
        trainer_address: The Ethereum address of the trainer
        nonce: A random secret value
        round_id: The current round identifier

    Returns:
        A hex string representing the commitment hash
    """
    # Normalize inputs
    trainer_address = trainer_address.lower()
    if trainer_address.startswith("0x"):
        trainer_address = trainer_address[2:]

    # Convert round_id to hex string without '0x' prefix
    round_id_hex = format(round_id, "x")

    # Concatenate the values
    message = f"{trainer_address}{nonce}{round_id_hex}"

    # Hash the message using keccak256 (eth compatible)
    hash_object = hashlib.sha3_256(message.encode("utf-8"))
    return hash_object.hexdigest()


def generate_trainer_commitment(trainer_address: str, round_id: int) -> Tuple[str, str]:
    """
    Generate a nonce and create a commitment for a trainer.

    Args:
        trainer_address: The Ethereum address of the trainer
        round_id: The current round identifier

    Returns:
        A tuple containing (nonce, commitment)
    """
    nonce = generate_nonce()
    commitment = create_commitment(trainer_address, nonce, round_id)
    return nonce, commitment
