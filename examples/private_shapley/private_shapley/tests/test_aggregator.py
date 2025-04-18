# tests/test_aggregator.py
import unittest
from private_shapley.aggregator import Aggregator
from private_shapley.merkle import MerkleTree


class TestAggregator(unittest.TestCase):
    def setUp(self):
        self.round_id = 42
        self.aggregator = Aggregator(self.round_id)

        # Example trainer addresses
        self.trainers = [
            "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            "0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199",
            "0xdD2FD4581271e230360230F9337D5c0430Bf44C0",
        ]

    def test_create_coalition(self):
        # Create a coalition
        coalition_id = self.aggregator.create_coalition(self.trainers)

        # Check that the coalition ID is a valid hex string
        self.assertTrue(all(c in "0123456789abcdef" for c in coalition_id))

        # Check that a Merkle tree was created
        coalition_key = next(iter(self.aggregator.coalition_ids.keys()))
        self.assertIsInstance(
            self.aggregator.coalition_trees[coalition_key], MerkleTree
        )

    def test_get_trainer_merkle_proof(self):
        # Create a coalition
        coalition_id = self.aggregator.create_coalition(self.trainers)

        # Get the coalition key from the ID
        coalition_key = next(iter(self.aggregator.coalition_ids.keys()))

        # Get proof for the first trainer
        proof = self.aggregator.get_trainer_merkle_proof(
            coalition_key, self.trainers[0]
        )

        # Check that the proof is a list of tuples
        self.assertIsInstance(proof, list)
        for item in proof:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)

    def test_get_trainer_claim_data(self):
        # Create a coalition
        coalition_id = self.aggregator.create_coalition(self.trainers)

        # Get the coalition key from the ID
        coalition_key = next(iter(self.aggregator.coalition_ids.keys()))

        # Get claim data for the first trainer
        claim_data = self.aggregator.get_trainer_claim_data(
            coalition_key, self.trainers[0]
        )

        # Check that all required fields are present
        required_fields = [
            "trainer_address",
            "nonce",
            "round_id",
            "merkle_proof",
            "merkle_root",
            "coalition_id",
        ]
        for field in required_fields:
            self.assertIn(field, claim_data)

        # Verify the data is correct
        self.assertEqual(claim_data["trainer_address"], self.trainers[0])
        self.assertEqual(claim_data["round_id"], self.round_id)
        self.assertEqual(claim_data["coalition_id"], coalition_id)

    def test_get_coalition_for_tester(self):
        # Create a coalition
        coalition_id = self.aggregator.create_coalition(self.trainers)

        # Get the coalition key from the ID
        coalition_key = next(iter(self.aggregator.coalition_ids.keys()))

        # Get tester data
        tester_data = self.aggregator.get_coalition_for_tester(coalition_key)

        # Check that all required fields are present
        required_fields = ["coalition_id", "round_id"]
        for field in required_fields:
            self.assertIn(field, tester_data)

        # Verify the data is correct
        self.assertEqual(tester_data["coalition_id"], coalition_id)
        self.assertEqual(tester_data["round_id"], self.round_id)

    def test_verification_flow(self):
        # Create a coalition
        coalition_id = self.aggregator.create_coalition(self.trainers)

        # Get the coalition key from the ID
        coalition_key = next(iter(self.aggregator.coalition_ids.keys()))

        # Get claim data for all trainers
        for trainer in self.trainers:
            claim_data = self.aggregator.get_trainer_claim_data(coalition_key, trainer)

            # Verify the proof for this trainer
            commitment = self.aggregator.coalition_commitments[coalition_key][
                self.trainers.index(trainer)
            ]

            # Check that the proof is valid
            is_valid = MerkleTree.verify_proof(
                commitment, claim_data["merkle_proof"], claim_data["merkle_root"]
            )
            self.assertTrue(is_valid)


if __name__ == "__main__":
    unittest.main()
