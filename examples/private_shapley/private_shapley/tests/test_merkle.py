# tests/test_merkle.py
import unittest
from private_shapley.merkle import MerkleTree


class TestMerkleTree(unittest.TestCase):
    def test_build_tree_even_leaves(self):
        # Create tree with 4 leaves
        leaves = [
            "1111111111111111111111111111111111111111111111111111111111111111",
            "2222222222222222222222222222222222222222222222222222222222222222",
            "3333333333333333333333333333333333333333333333333333333333333333",
            "4444444444444444444444444444444444444444444444444444444444444444",
        ]
        tree = MerkleTree(leaves)

        # Check number of layers (leaves, intermediate, root)
        self.assertEqual(len(tree.layers), 3)

        # Check the first layer (leaves)
        self.assertEqual(tree.layers[0], leaves)

        # Check that the root exists
        self.assertIsNotNone(tree.root)

    def test_build_tree_odd_leaves(self):
        # Create tree with 3 leaves
        leaves = [
            "1111111111111111111111111111111111111111111111111111111111111111",
            "2222222222222222222222222222222222222222222222222222222222222222",
            "3333333333333333333333333333333333333333333333333333333333333333",
        ]
        tree = MerkleTree(leaves)

        # Check number of layers
        self.assertEqual(len(tree.layers), 3)

        # The first layer should have 3 leaves
        self.assertEqual(len(tree.layers[0]), 3)

        # The second layer should have 2 nodes
        self.assertEqual(len(tree.layers[1]), 2)

        # The last layer should have the root
        self.assertEqual(len(tree.layers[2]), 1)

    def test_get_proof(self):
        leaves = [
            "1111111111111111111111111111111111111111111111111111111111111111",
            "2222222222222222222222222222222222222222222222222222222222222222",
            "3333333333333333333333333333333333333333333333333333333333333333",
            "4444444444444444444444444444444444444444444444444444444444444444",
        ]
        tree = MerkleTree(leaves)

        # Get proof for the first leaf
        proof_0 = tree.get_proof(0)

        # A proof for a balanced tree with 4 leaves should have 2 elements
        self.assertEqual(len(proof_0), 2)

        # Get proof for the last leaf
        proof_3 = tree.get_proof(3)
        self.assertEqual(len(proof_3), 2)

    def test_verify_proof(self):
        leaves = [
            "1111111111111111111111111111111111111111111111111111111111111111",
            "2222222222222222222222222222222222222222222222222222222222222222",
            "3333333333333333333333333333333333333333333333333333333333333333",
            "4444444444444444444444444444444444444444444444444444444444444444",
        ]
        tree = MerkleTree(leaves)

        # For each leaf, get its proof and verify it
        for i, leaf in enumerate(leaves):
            proof = tree.get_proof(i)
            is_valid = MerkleTree.verify_proof(leaf, proof, tree.root)
            self.assertTrue(is_valid, f"Proof for leaf {i} should be valid")

        # Verify that an invalid leaf fails
        fake_leaf = "5555555555555555555555555555555555555555555555555555555555555555"
        proof_0 = tree.get_proof(0)
        is_valid = MerkleTree.verify_proof(fake_leaf, proof_0, tree.root)
        self.assertFalse(is_valid, "Proof should fail for an invalid leaf")

        # Verify that a tampered proof fails
        proof_0_tampered = [(not is_right, sibling) for is_right, sibling in proof_0]
        is_valid = MerkleTree.verify_proof(leaves[0], proof_0_tampered, tree.root)
        self.assertFalse(is_valid, "Tampered proof should fail verification")

    def test_single_leaf(self):
        # Edge case: create a tree with a single leaf
        leaves = ["1111111111111111111111111111111111111111111111111111111111111111"]
        tree = MerkleTree(leaves)

        # A tree with one leaf should have just one layer
        self.assertEqual(len(tree.layers), 1)

        # The root should be the leaf itself
        self.assertEqual(tree.root, leaves[0])

        # The proof should be empty
        proof = tree.get_proof(0)
        self.assertEqual(len(proof), 0)

        # Verification should pass
        is_valid = MerkleTree.verify_proof(leaves[0], proof, tree.root)
        self.assertTrue(is_valid)


if __name__ == "__main__":
    unittest.main()
