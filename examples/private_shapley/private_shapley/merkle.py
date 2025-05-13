"""Merkle tree implementation for private Shapley value calculation."""

import hashlib
from typing import List, Optional

class MerkleTree:
    """A Merkle tree implementation for coalition verification."""
    
    def __init__(self, leaves: List[bytes]):
        """Initialize a Merkle tree with the given leaf nodes.
        
        Args:
            leaves: List of leaf node values as bytes
        """
        if not leaves:
            raise ValueError("Cannot create Merkle tree with empty leaves")
        
        # Convert leaves to their hash values if they're not already hashes
        self.leaves = [self._hash(leaf) if isinstance(leaf, bytes) else leaf for leaf in leaves]
        self.layers = [self.leaves]
        self._build_tree()
    
    def _hash(self, data: bytes) -> bytes:
        """Hash the given data using keccak256."""
        return hashlib.sha3_256(data).digest()
    
    def _build_tree(self):
        """Build the Merkle tree from the leaf nodes."""
        layer = self.leaves
        
        # Continue until we reach the root
        while len(layer) > 1:
            next_layer = []
            
            # Process pairs of nodes
            for i in range(0, len(layer), 2):
                # If odd number of nodes in the layer, duplicate the last node
                if i + 1 == len(layer):
                    next_layer.append(self._hash(layer[i] + layer[i]))
                else:
                    # Hash the concatenation of the two child nodes
                    if layer[i] <= layer[i+1]:
                        next_layer.append(self._hash(layer[i] + layer[i+1]))
                    else:
                        next_layer.append(self._hash(layer[i+1] + layer[i]))
            
            self.layers.append(next_layer)
            layer = next_layer
    
    def get_root(self) -> bytes:
        """Get the Merkle root hash.
        
        Returns:
            The Merkle root as bytes
        """
        return self.layers[-1][0]
    
    def get_proof(self, index: int) -> List[bytes]:
        """Generate a Merkle proof for the leaf at the given index.
        
        Args:
            index: The index of the leaf node
            
        Returns:
            A list of sibling hashes forming the Merkle proof
        """
        if index < 0 or index >= len(self.leaves):
            raise ValueError(f"Index {index} out of range for leaves of length {len(self.leaves)}")
        
        proof = []
        idx = index
        
        for i in range(len(self.layers) - 1):
            layer = self.layers[i]
            is_odd = idx % 2 == 1
            
            if is_odd:
                # If odd, the sibling is to the left
                sibling_idx = idx - 1
            else:
                # If even, the sibling is to the right (if it exists)
                sibling_idx = idx + 1 if idx + 1 < len(layer) else idx
            
            # Add the sibling to the proof
            if sibling_idx < len(layer):
                proof.append(layer[sibling_idx])
            
            # Move to the parent index in the next layer
            idx = idx // 2
        
        return proof
    
    @staticmethod
    def verify_proof(root: bytes, leaf: bytes, proof: List[bytes]) -> bool:
        """Verify a Merkle proof for a given leaf.
        
        Args:
            root: The Merkle root to verify against
            leaf: The leaf node value
            proof: The Merkle proof (list of sibling hashes)
            
        Returns:
            True if the proof is valid, False otherwise
        """
        # Hash the leaf if it's not already a hash
        current = leaf
        
        for sibling in proof:
            if current <= sibling:
                current = hashlib.sha3_256(current + sibling).digest()
            else:
                current = hashlib.sha3_256(sibling + current).digest()
        
        return current == root


def generate_coalition_merkle_tree(round_id: int, coalition_members: List[tuple], nonces: List[bytes]) -> MerkleTree:
    """Generate a Merkle tree for a coalition.
    
    Args:
        round_id: The current round ID
        coalition_members: List of (address, trainer_index) tuples for coalition members
        nonces: List of random nonces for each member
        
    Returns:
        A MerkleTree object for the coalition
    """
    if len(coalition_members) != len(nonces):
        raise ValueError("Number of members and nonces must match")
    
    leaves = []
    
    # Create leaf nodes for each coalition member
    for (addr, _), nonce in zip(coalition_members, nonces):
        # Create leaf as keccak256(address + nonce + roundId)
        leaf_data = addr.encode() + nonce + round_id.to_bytes(32, 'big')
        leaf = hashlib.sha3_256(leaf_data).digest()
        leaves.append(leaf)
    
    return MerkleTree(leaves)