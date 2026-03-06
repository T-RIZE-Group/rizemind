"""Tests for TEEAggregationStrategy integration with MockTEEEnclave."""

import numpy as np
import pytest

from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import FitRes, Parameters, Scalar, Status
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg

from rizemind.tee.attestation import MockAttestationVerifier
from rizemind.tee.crypto import (
    aes_gcm_encrypt,
    derive_shared_secret,
    derive_symmetric_key,
    deserialize_public_key,
    generate_ecdh_keypair,
    serialize_public_key,
)
from rizemind.tee.mock_enclave import MockTEEEnclave
from rizemind.tee.params import serialize_parameters
from rizemind.tee.tee_strategy import (
    TEE_CLIENT_PUBKEY_METRIC,
    TEE_ENCRYPTED_PARAMS_METRIC,
    TEE_NONCE_METRIC,
    TEE_PUBLIC_KEY_CONFIG,
    TEEAggregationStrategy,
)


class _StubClientProxy(ClientProxy):
    def __init__(self, cid: str) -> None:
        super().__init__(cid)

    def get_properties(self, ins, timeout, group_id):
        raise NotImplementedError

    def get_parameters(self, ins, timeout, group_id):
        raise NotImplementedError

    def fit(self, ins, timeout, group_id):
        raise NotImplementedError

    def evaluate(self, ins, timeout, group_id):
        raise NotImplementedError

    def reconnect(self, ins, timeout, group_id):
        raise NotImplementedError


def _make_encrypted_fit_res(
    weights: np.ndarray,
    num_examples: int,
    tee_pubkey_bytes: bytes,
) -> FitRes:
    """Create a FitRes with encrypted parameters in metrics."""
    client_key = generate_ecdh_keypair()
    client_pubkey_bytes = serialize_public_key(client_key.public_key())

    tee_pubkey = deserialize_public_key(tee_pubkey_bytes)
    shared_secret = derive_shared_secret(client_key, tee_pubkey)
    symmetric_key = derive_symmetric_key(shared_secret)

    params = ndarrays_to_parameters([weights])
    plaintext = serialize_parameters(params)
    ciphertext, nonce = aes_gcm_encrypt(symmetric_key, plaintext)

    return FitRes(
        status=Status(code=0, message="OK"),
        parameters=params,  # original params (won't be used by TEE strategy)
        num_examples=num_examples,
        metrics={
            TEE_ENCRYPTED_PARAMS_METRIC: ciphertext,
            TEE_NONCE_METRIC: nonce,
            TEE_CLIENT_PUBKEY_METRIC: client_pubkey_bytes,
        },
    )


class TestTEEAggregationStrategy:
    def test_aggregate_fit_two_clients(self):
        enclave = MockTEEEnclave()
        verifier = MockAttestationVerifier()
        base = FedAvg()
        strategy = TEEAggregationStrategy(base, enclave, verifier)

        # Initialize to get the public key
        strategy._ensure_enclave()
        tee_pubkey = enclave.get_public_key()

        # Two clients with equal examples
        res_a = _make_encrypted_fit_res(
            np.array([2.0, 4.0], dtype=np.float32), 100, tee_pubkey
        )
        res_b = _make_encrypted_fit_res(
            np.array([6.0, 8.0], dtype=np.float32), 100, tee_pubkey
        )

        results = [
            (_StubClientProxy("a"), res_a),
            (_StubClientProxy("b"), res_b),
        ]
        failures: list = []

        aggregated, metrics = strategy.aggregate_fit(1, results, failures)

        assert aggregated is not None
        assert metrics["tee_aggregated_count"] == 2
        assert len(failures) == 0

        result = parameters_to_ndarrays(aggregated)[0]
        np.testing.assert_array_almost_equal(result, [4.0, 6.0])

    def test_missing_tee_metadata_goes_to_failures(self):
        enclave = MockTEEEnclave()
        verifier = MockAttestationVerifier()
        strategy = TEEAggregationStrategy(FedAvg(), enclave, verifier)
        strategy._ensure_enclave()

        # FitRes without TEE metrics
        plain_res = FitRes(
            status=Status(code=0, message="OK"),
            parameters=Parameters(tensors=[], tensor_type=""),
            num_examples=10,
            metrics={},
        )

        results = [(_StubClientProxy("x"), plain_res)]
        failures: list = []

        aggregated, metrics = strategy.aggregate_fit(1, results, failures)

        assert aggregated is None
        assert len(failures) == 1

    def test_attestation_verification(self):
        enclave = MockTEEEnclave()
        verifier = MockAttestationVerifier()
        strategy = TEEAggregationStrategy(FedAvg(), enclave, verifier)

        # Should not raise
        strategy._ensure_enclave()
        assert strategy._attestation is not None
        assert strategy._attestation.platform == "mock"
