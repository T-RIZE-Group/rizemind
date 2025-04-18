# tests/test_integration.py
import unittest
from ..commitment import generate_trainer_commitment
from ..merkle import MerkleTree


class TestIntegration(unittest.TestCase):
    def test_commitment_with_merkle_tree(self):
        # Create commitments for several trainers
        trainers = [
            "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            "0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199",
            "0xdD2FD4581271e230360230F9337D5c0430Bf44C0",
        ]
        round_id = 42

        # Generate commitments
        trainer_data = []
        for trainer in trainers:
            nonce, commitment = generate_trainer_commitment(trainer, round_id)
            trainer_data.append((trainer, nonce, commitment))

        # Get just the commitments
        commitments = [data[2] for data in trainer_data]

        # Create a Merkle tree with the commitments
        tree = MerkleTree(commitments)

        # For each trainer, verify they can prove their membership
        for i, (trainer, nonce, commitment) in enumerate(trainer_data):
            # Get the proof for this commitment
            proof = tree.get_proof(i)

            # Verify the proof
            is_valid = MerkleTree.verify_proof(commitment, proof, tree.root)
            self.assertTrue(is_valid, f"Proof for trainer {trainer} should be valid")

            # Store the necessary data that a trainer would need to claim their reward
            trainer_claim_data = {
                "trainer_address": trainer,
                "nonce": nonce,
                "round_id": round_id,
                "merkle_proof": proof,
                "merkle_root": tree.root,
            }

            # Print the claim data (in a real system, this would be stored)
            print(f"Trainer {i} claim data: {trainer_claim_data}")


if __name__ == "__main__":
    unittest.main()
