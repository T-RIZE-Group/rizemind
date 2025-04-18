# private_shapley/merkle.py
import hashlib
from typing import List, Optional, Tuple


class MerkleTree:
    def __init__(self, leaves: List[str]):
        """
        Initialize a Merkle tree with the given leaf nodes.

        Args:
            leaves: List of hex strings representing leaf node values
        """
        if not leaves:
            raise ValueError("Cannot create a Merkle tree with no leaves")

        # Normalize leaves to ensure they're all hex strings without '0x' prefix
        self.leaves = [
            leaf.lower() if leaf.startswith("0x") else leaf.lower() for leaf in leaves
        ]

        # Build the tree
        self.layers = self._build_tree()

        # The root is the first element in the last layer
        self.root = self.layers[-1][0]

    def _hash_pair(self, left: str, right: str) -> str:
        """
        Hash a pair of nodes to create their parent node.

        Args:
            left: Left child node hash
            right: Right child node hash

        Returns:
            Hash of the concatenated child nodes
        """
        # Concatenate the hashes in order (important!)
        concat = left + right
        hash_object = hashlib.sha3_256(bytes.fromhex(concat))
        return hash_object.hexdigest()

    def _build_level(self, level: List[str]) -> List[str]:
        """
        Build the next level up in the tree.

        Args:
            level: Current level of nodes

        Returns:
            Next level up in the Merkle tree
        """
        next_level = []
        for i in range(0, len(level), 2):
            # If we have an odd number of elements, duplicate the last one
            if i + 1 == len(level):
                next_level.append(self._hash_pair(level[i], level[i]))
            else:
                next_level.append(self._hash_pair(level[i], level[i + 1]))
        return next_level

    def _build_tree(self) -> List[List[str]]:
        """
        Build the entire Merkle tree.

        Returns:
            List of layers in the tree, from leaves to root
        """
        layers = [self.leaves]
        current_level = self.leaves

        # Keep building levels until we reach the root
        while len(current_level) > 1:
            current_level = self._build_level(current_level)
            layers.append(current_level)

        return layers

    def get_proof(self, leaf_index: int) -> List[Tuple[bool, str]]:
        """
        Get the Merkle proof for a specific leaf.

        Args:
            leaf_index: Index of the leaf in the original list

        Returns:
            List of (is_right, sibling_hash) tuples forming the proof
        """
        if leaf_index < 0 or leaf_index >= len(self.leaves):
            raise ValueError(
                f"Leaf index {leaf_index} out of range (0-{len(self.leaves) - 1})"
            )

        proof = []
        current_index = leaf_index

        for i in range(len(self.layers) - 1):
            current_level = self.layers[i]
            is_right_child = current_index % 2 == 1

            if is_right_child:
                # If current node is on the right, sibling is on the left
                sibling_index = current_index - 1
            else:
                # If current node is on the left, sibling is on the right
                sibling_index = current_index + 1

                # If we're at the end of an odd-length level, use the same node again
                if sibling_index >= len(current_level):
                    sibling_index = current_index

            sibling_hash = current_level[sibling_index]
            proof.append((is_right_child, sibling_hash))

            # Move up to the parent's index in the next level
            current_index = current_index // 2

        return proof

    @staticmethod
    def verify_proof(leaf: str, proof: List[Tuple[bool, str]], root: str) -> bool:
        """
        Verify a Merkle proof for a given leaf.

        Args:
            leaf: Leaf value (hash) to verify
            proof: List of (is_right_child, sibling_hash) tuples
            root: Expected Merkle root

        Returns:
            True if the proof is valid, False otherwise
        """
        # Normalize inputs
        leaf = leaf.lower() if leaf.startswith("0x") else leaf.lower()
        root = root.lower() if root.startswith("0x") else root.lower()

        current = leaf

        for is_right_child, sibling in proof:
            sibling = sibling.lower() if sibling.startswith("0x") else sibling.lower()

            if is_right_child:
                # Current node is on the right, sibling is on the left
                current = MerkleTree._hash_pair_static(sibling, current)
            else:
                # Current node is on the left, sibling is on the right
                current = MerkleTree._hash_pair_static(current, sibling)

        return current == root

    @staticmethod
    def _hash_pair_static(left: str, right: str) -> str:
        """
        Static version of _hash_pair for use in verification.

        Args:
            left: Left child node hash
            right: Right child node hash

        Returns:
            Hash of the concatenated child nodes
        """
        # Concatenate the hashes in order (important!)
        concat = left + right
        hash_object = hashlib.sha3_256(bytes.fromhex(concat))
        return hash_object.hexdigest()
