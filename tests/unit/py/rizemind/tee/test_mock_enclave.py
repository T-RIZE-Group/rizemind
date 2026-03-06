"""Tests for MockTEEEnclave — full encrypt/aggregate/decrypt cycle."""

import numpy as np
import pytest

from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import Parameters

from rizemind.tee.crypto import (
    aes_gcm_encrypt,
    derive_shared_secret,
    derive_symmetric_key,
    deserialize_public_key,
    generate_ecdh_keypair,
    serialize_public_key,
)
from rizemind.tee.mock_enclave import MockTEEEnclave
from rizemind.tee.params import deserialize_parameters, serialize_parameters


def _encrypt_for_enclave(
    trainer_weights: np.ndarray,
    tee_pubkey_bytes: bytes,
    num_examples: int,
) -> tuple[bytes, bytes, bytes, int]:
    """Helper: encrypt a trainer's weights for the TEE enclave."""
    client_key = generate_ecdh_keypair()
    client_pubkey_bytes = serialize_public_key(client_key.public_key())

    tee_pubkey = deserialize_public_key(tee_pubkey_bytes)
    shared_secret = derive_shared_secret(client_key, tee_pubkey)
    symmetric_key = derive_symmetric_key(shared_secret)

    params = ndarrays_to_parameters([trainer_weights])
    plaintext = serialize_parameters(params)
    ciphertext, nonce = aes_gcm_encrypt(symmetric_key, plaintext)

    return ciphertext, nonce, client_pubkey_bytes, num_examples


class TestMockTEEEnclave:
    def test_lifecycle(self):
        enclave = MockTEEEnclave()
        enclave.initialize()
        pubkey = enclave.get_public_key()
        assert len(pubkey) == 65  # uncompressed secp256k1
        assert pubkey[0] == 0x04

        report = enclave.get_attestation_report()
        assert report.platform == "mock"
        assert report.enclave_public_key == pubkey

        enclave.destroy()

    def test_not_initialized_raises(self):
        enclave = MockTEEEnclave()
        with pytest.raises(RuntimeError):
            enclave.get_public_key()

    def test_attestation_report(self):
        enclave = MockTEEEnclave()
        enclave.initialize()
        report = enclave.get_attestation_report()
        assert report.document == b"mock-attestation"
        assert report.timestamp > 0
        enclave.destroy()

    def test_fedavg_two_clients_equal_weights(self):
        """Two clients with equal examples → simple average."""
        enclave = MockTEEEnclave()
        enclave.initialize()
        tee_pubkey = enclave.get_public_key()

        # Client A: weights = [2.0, 4.0]
        ct_a, nonce_a, pk_a, _ = _encrypt_for_enclave(
            np.array([2.0, 4.0], dtype=np.float32), tee_pubkey, 100
        )
        # Client B: weights = [6.0, 8.0]
        ct_b, nonce_b, pk_b, _ = _encrypt_for_enclave(
            np.array([6.0, 8.0], dtype=np.float32), tee_pubkey, 100
        )

        result_bytes = enclave.aggregate(
            encrypted_updates=[(ct_a, nonce_a, pk_a), (ct_b, nonce_b, pk_b)],
            num_examples=[100, 100],
            server_round=1,
        )

        aggregated = deserialize_parameters(result_bytes)
        result = parameters_to_ndarrays(aggregated)[0]

        # FedAvg with equal weights: (2+6)/2=4, (4+8)/2=6
        np.testing.assert_array_almost_equal(result, [4.0, 6.0])
        enclave.destroy()

    def test_fedavg_weighted(self):
        """Two clients with different example counts → weighted average."""
        enclave = MockTEEEnclave()
        enclave.initialize()
        tee_pubkey = enclave.get_public_key()

        # Client A: weights = [10.0], 300 examples
        ct_a, nonce_a, pk_a, _ = _encrypt_for_enclave(
            np.array([10.0], dtype=np.float32), tee_pubkey, 300
        )
        # Client B: weights = [20.0], 100 examples
        ct_b, nonce_b, pk_b, _ = _encrypt_for_enclave(
            np.array([20.0], dtype=np.float32), tee_pubkey, 100
        )

        result_bytes = enclave.aggregate(
            encrypted_updates=[(ct_a, nonce_a, pk_a), (ct_b, nonce_b, pk_b)],
            num_examples=[300, 100],
            server_round=1,
        )

        aggregated = deserialize_parameters(result_bytes)
        result = parameters_to_ndarrays(aggregated)[0]

        # FedAvg weighted: (10*300 + 20*100) / (300+100) = 5000/400 = 12.5
        np.testing.assert_array_almost_equal(result, [12.5])
        enclave.destroy()

    def test_single_client(self):
        """Single client → aggregated result equals input."""
        enclave = MockTEEEnclave()
        enclave.initialize()
        tee_pubkey = enclave.get_public_key()

        weights = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ct, nonce, pk, _ = _encrypt_for_enclave(weights, tee_pubkey, 50)

        result_bytes = enclave.aggregate(
            encrypted_updates=[(ct, nonce, pk)],
            num_examples=[50],
            server_round=1,
        )

        aggregated = deserialize_parameters(result_bytes)
        result = parameters_to_ndarrays(aggregated)[0]

        np.testing.assert_array_equal(result, weights)
        enclave.destroy()
