"""Commitment scheme for private Shapley value calculation."""

import os
import hashlib
from typing import Tuple, List, Dict
import secrets

def generate_nonce() -> bytes:
    """Generate a random 32-byte nonce.
    
    Returns:
        A 32-byte random value
    """
    return secrets.token_bytes(32)

def create_commitment(address: str, nonce: bytes, round_id: int) -> bytes:
    """Create a commitment for a trainer.
    
    Args:
        address: The trainer's Ethereum address
        nonce: A random 32-byte nonce
        round_id: The current round ID
        
    Returns:
        The commitment hash
    """
    # Create commitment as keccak256(address + nonce + roundId)
    commitment_data = address.encode() + nonce + round_id.to_bytes(32, 'big')
    return hashlib.sha3_256(commitment_data).digest()

def address_to_bitfield(trainer_indices: List[int], max_trainers: int = 255) -> bytes:
    """Convert a list of trainer indices to a bitfield.
    
    Args:
        trainer_indices: List of trainer indices (1-based)
        max_trainers: Maximum number of trainers (default: 255)
        
    Returns:
        A bytes32 bitfield where bit i is set if trainer i is in the coalition
    """
    # Initialize a bitfield with all bits unset (all zeros)
    bitfield = 0
    
    # Set the bits corresponding to each trainer's index
    for idx in trainer_indices:
        if idx < 1 or idx > max_trainers:
            raise ValueError(f"Trainer index {idx} out of range (1-{max_trainers})")
        
        # Set the bit at position (idx-1)
        bitfield |= (1 << (idx - 1))
    
    # Convert to bytes32
    return bitfield.to_bytes(32, 'big')

def bitfield_to_indices(bitfield: bytes) -> List[int]:
    """Convert a bitfield to a list of trainer indices.
    
    Args:
        bitfield: A bytes32 bitfield
        
    Returns:
        A list of trainer indices (1-based)
    """
    # Convert bytes to integer
    int_bitfield = int.from_bytes(bitfield, 'big')
    
    # Extract indices where bits are set
    indices = []
    for i in range(255):  # 255 is the max number of trainers
        if (int_bitfield >> i) & 1:
            indices.append(i + 1)  # Convert to 1-based index
    
    return indices

class CoalitionManager:
    """Manages coalitions for the private Shapley scheme."""
    
    def __init__(self, max_trainers: int = 255):
        """Initialize the coalition manager.
        
        Args:
            max_trainers: Maximum number of trainers
        """
        self.max_trainers = max_trainers
        self.round_nonces: Dict[int, Dict[str, bytes]] = {}  # round_id -> {address -> nonce}
        self.trainer_indices: Dict[str, int] = {}  # address -> index
    
    def register_trainer(self, address: str, index: int) -> None:
        """Register a trainer with their index.
        
        Args:
            address: The trainer's Ethereum address
            index: The trainer's unique index
        """
        if index < 1 or index > self.max_trainers:
            raise ValueError(f"Trainer index {index} out of range (1-{self.max_trainers})")
        
        self.trainer_indices[address] = index
    
    def generate_trainer_nonce(self, address: str, round_id: int) -> bytes:
        """Generate and store a nonce for a trainer in a specific round.
        
        Args:
            address: The trainer's Ethereum address
            round_id: The current round ID
            
        Returns:
            The generated nonce
        """
        if round_id not in self.round_nonces:
            self.round_nonces[round_id] = {}
        
        nonce = generate_nonce()
        self.round_nonces[round_id][address] = nonce
        return nonce
    
    def get_trainer_nonce(self, address: str, round_id: int) -> bytes:
        """Get a trainer's nonce for a specific round.
        
        Args:
            address: The trainer's Ethereum address
            round_id: The round ID
            
        Returns:
            The trainer's nonce for the given round
        """
        if round_id not in self.round_nonces or address not in self.round_nonces[round_id]:
            raise ValueError(f"No nonce found for trainer {address} in round {round_id}")
        
        return self.round_nonces[round_id][address]
    
    def create_coalition_data(self, round_id: int, coalition_addresses: List[str]) -> Tuple[bytes, bytes]:
        """Create coalition data for publishing to the contract.
        
        Args:
            round_id: The current round ID
            coalition_addresses: List of addresses in the coalition
            
        Returns:
            A tuple (bitfield, merkle_root) for the coalition
        """
        # Get indices for all addresses in the coalition
        trainer_indices = []
        coalition_members = []
        nonces = []
        
        for addr in coalition_addresses:
            if addr not in self.trainer_indices:
                raise ValueError(f"Trainer {addr} not registered")
            
            idx = self.trainer_indices[addr]
            trainer_indices.append(idx)
            coalition_members.append((addr, idx))
            nonces.append(self.get_trainer_nonce(addr, round_id))
        
        # Create bitfield
        bitfield = address_to_bitfield(trainer_indices, self.max_trainers)
        
        # Create Merkle tree
        merkle_tree = generate_coalition_merkle_tree(round_id, coalition_members, nonces)
        merkle_root = merkle_tree.get_root()
        
        return bitfield, merkle_root
    
    def generate_merkle_proof(self, round_id: int, coalition_addresses: List[str], trainer_address: str) -> List[bytes]:
        """Generate a Merkle proof for a trainer in a coalition.
        
        Args:
            round_id: The round ID
            coalition_addresses: List of addresses in the coalition
            trainer_address: The trainer's address
            
        Returns:
            The Merkle proof for the trainer
        """
        if trainer_address not in coalition_addresses:
            raise ValueError(f"Trainer {trainer_address} not in coalition")
        
        # Get trainer index in coalition
        trainer_idx = coalition_addresses.index(trainer_address)
        
        # Recreate the Merkle tree
        coalition_members = []
        nonces = []
        
        for addr in coalition_addresses:
            idx = self.trainer_indices[addr]
            coalition_members.append((addr, idx))
            nonces.append(self.get_trainer_nonce(addr, round_id))
        
        merkle_tree = generate_coalition_merkle_tree(round_id, coalition_members, nonces)
        
        # Generate proof
        return merkle_tree.get_proof(trainer_idx)