# private_shapley/test_commitment.py
import unittest
from ..commitment import create_commitment, generate_trainer_commitment


class TestCommitment(unittest.TestCase):
    def test_create_commitment(self):
        trainer_address = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
        nonce = "abcdef1234567890"
        round_id = 42

        commitment = create_commitment(trainer_address, nonce, round_id)

        # Test deterministic behavior
        same_commitment = create_commitment(trainer_address, nonce, round_id)
        self.assertEqual(commitment, same_commitment)

        # Test that different inputs produce different outputs
        different_nonce = create_commitment(
            trainer_address, "different_nonce", round_id
        )
        self.assertNotEqual(commitment, different_nonce)

        different_round = create_commitment(trainer_address, nonce, 43)
        self.assertNotEqual(commitment, different_round)

        different_address = create_commitment(
            "0x0000000000000000000000000000000000000000", nonce, round_id
        )
        self.assertNotEqual(commitment, different_address)

    def test_generate_trainer_commitment(self):
        trainer_address = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
        round_id = 42

        nonce, commitment = generate_trainer_commitment(trainer_address, round_id)

        # Check if the nonce is a hex string of length 32
        self.assertEqual(len(nonce), 32)

        # Check if the commitment is valid
        reconstructed = create_commitment(trainer_address, nonce, round_id)
        self.assertEqual(commitment, reconstructed)

        # Generate a new commitment and check if it's different
        new_nonce, new_commitment = generate_trainer_commitment(
            trainer_address, round_id
        )
        self.assertNotEqual(nonce, new_nonce)
        self.assertNotEqual(commitment, new_commitment)


if __name__ == "__main__":
    unittest.main()
