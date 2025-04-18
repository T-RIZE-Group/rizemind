# private_shapley/Aggregator.py
import hashlib
import uuid
from typing import Dict, List, Tuple

from .commitment import create_commitment, generate_trainer_commitment
from .merkle import MerkleTree


class Aggregator:
    """
    The Aggregator (Aggregator) manages the commitment collection and Merkle tree generation.
    It interfaces with the blockchain for publishing coalition roots and results.
    """

    def __init__(self, round_id: int):
        """
        Initialize a new Aggregator for a specific round.

        Args:
            round_id: The current round identifier
        """
        self.round_id = round_id
        self.coalition_trees: Dict[str, MerkleTree] = {}
        self.coalition_trainers: Dict[str, List[str]] = {}
        self.coalition_commitments: Dict[str, List[str]] = {}
        self.coalition_nonces: Dict[str, Dict[str, str]] = {}
        self.coalition_roots: Dict[str, str] = {}
        self.coalition_ids: Dict[str, str] = {}

    def create_coalition(self, trainer_addresses: List[str]) -> str:
        """
        Create a new coalition from a list of trainer addresses.

        Args:
            trainer_addresses: List of Ethereum addresses for the trainers

        Returns:
            The coalition ID
        """
        # Generate a unique coalition ID
        coalition_key = str(uuid.uuid4())

        # Store trainer addresses for this coalition
        self.coalition_trainers[coalition_key] = trainer_addresses

        # Generate commitments and nonces for each trainer
        commitments = []
        nonces = {}

        for trainer in trainer_addresses:
            nonce, commitment = generate_trainer_commitment(trainer, self.round_id)
            commitments.append(commitment)
            nonces[trainer] = nonce

        # Create a Merkle tree for this coalition
        tree = MerkleTree(commitments)

        # Store the coalition data
        self.coalition_trees[coalition_key] = tree
        self.coalition_commitments[coalition_key] = commitments
        self.coalition_nonces[coalition_key] = nonces
        self.coalition_roots[coalition_key] = tree.root

        # Generate a public coalition ID that doesn't reveal the root directly
        # In this simple implementation, we just hash the root with the coalition key
        public_id = hashlib.sha256(
            (tree.root + coalition_key).encode("utf-8")
        ).hexdigest()
        self.coalition_ids[coalition_key] = public_id

        return public_id

    def get_trainer_merkle_proof(
        self, coalition_key: str, trainer_address: str
    ) -> List[Tuple[bool, str]]:
        """
        Get the Merkle proof for a specific trainer in a coalition.

        Args:
            coalition_key: The internal coalition key
            trainer_address: The trainer's Ethereum address

        Returns:
            The Merkle proof for this trainer
        """
        if coalition_key not in self.coalition_trees:
            raise ValueError(f"Unknown coalition key: {coalition_key}")

        if trainer_address not in self.coalition_trainers[coalition_key]:
            raise ValueError(
                f"Trainer {trainer_address} not in coalition {coalition_key}"
            )

        # Find the index of this trainer's commitment
        trainer_index = self.coalition_trainers[coalition_key].index(trainer_address)

        # Get the proof from the Merkle tree
        tree = self.coalition_trees[coalition_key]
        return tree.get_proof(trainer_index)

    def get_trainer_claim_data(self, coalition_key: str, trainer_address: str) -> Dict:
        """
        Get all data needed for a trainer to claim their reward.

        Args:
            coalition_key: The internal coalition key
            trainer_address: The trainer's Ethereum address

        Returns:
            Dictionary with all data needed for the claim
        """
        if coalition_key not in self.coalition_ids:
            raise ValueError(f"Unknown coalition key: {coalition_key}")

        if trainer_address not in self.coalition_nonces[coalition_key]:
            raise ValueError(
                f"Trainer {trainer_address} not in coalition {coalition_key}"
            )

        proof = self.get_trainer_merkle_proof(coalition_key, trainer_address)

        return {
            "trainer_address": trainer_address,
            "nonce": self.coalition_nonces[coalition_key][trainer_address],
            "round_id": self.round_id,
            "merkle_proof": proof,
            "merkle_root": self.coalition_roots[coalition_key],
            "coalition_id": self.coalition_ids[coalition_key],
        }

    def get_coalition_for_tester(self, coalition_key: str) -> Dict:
        """
        Get the public coalition ID for a tester, without revealing members.

        Args:
            coalition_key: The internal coalition key

        Returns:
            Dictionary with data for the tester
        """
        if coalition_key not in self.coalition_ids:
            raise ValueError(f"Unknown coalition key: {coalition_key}")

        return {
            "coalition_id": self.coalition_ids[coalition_key],
            "round_id": self.round_id,
        }

    def get_coalition_publication_data(self, coalition_key: str) -> Dict:
        """
        Get data needed to publish the coalition root on-chain.

        Args:
            coalition_key: The internal coalition key

        Returns:
            Dictionary with data for publishing
        """
        if coalition_key not in self.coalition_ids:
            raise ValueError(f"Unknown coalition key: {coalition_key}")

        return {
            "coalition_id": self.coalition_ids[coalition_key],
            "merkle_root": self.coalition_roots[coalition_key],
        }
