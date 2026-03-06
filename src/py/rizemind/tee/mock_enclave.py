"""Software mock of a TEE enclave for development and testing.

Performs all cryptographic and aggregation operations in normal process
memory — no hardware TEE required.  Provides the same interface as
``NitroTEEEnclave`` so the rest of the stack (strategy, client mod)
works identically in both modes.
"""

import json
import logging
import struct
import time

from cryptography.hazmat.primitives.asymmetric.ec import (
    ECDH,
    SECP256K1,
    EllipticCurvePrivateKey,
    EllipticCurvePublicKey,
    generate_private_key,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from flwr.common.typing import FitRes, Parameters, Status
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg

from rizemind.tee.crypto import (
    TEE_AES_AAD,
    TEE_HKDF_INFO,
    aes_gcm_decrypt,
    derive_symmetric_key,
)
from rizemind.tee.enclave import AttestationReport, TEEEnclave
from rizemind.tee.params import deserialize_parameters, serialize_parameters

log = logging.getLogger(__name__)


class _StubClientProxy(ClientProxy):
    """Minimal stub required by Flower's aggregate_fit signature."""

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


class MockTEEEnclave(TEEEnclave):
    """In-process software simulation of a TEE enclave.

    Uses the same ECDH + AES-GCM + Flower FedAvg pipeline as the real
    Nitro enclave server, but runs everything in the current process.
    """

    def __init__(self) -> None:
        self._private_key: EllipticCurvePrivateKey | None = None

    def initialize(self) -> None:
        self._private_key = generate_private_key(SECP256K1())
        log.info("MockTEEEnclave initialized")

    def get_attestation_report(self) -> AttestationReport:
        return AttestationReport(
            enclave_public_key=self.get_public_key(),
            document=b"mock-attestation",
            platform="mock",
            timestamp=time.time(),
        )

    def get_public_key(self) -> bytes:
        if self._private_key is None:
            raise RuntimeError("Enclave not initialized")
        return self._private_key.public_key().public_bytes(
            Encoding.X962, PublicFormat.UncompressedPoint
        )

    def aggregate(
        self,
        encrypted_updates: list[tuple[bytes, bytes, bytes]],
        num_examples: list[int],
        server_round: int,
    ) -> bytes:
        if self._private_key is None:
            raise RuntimeError("Enclave not initialized")

        results: list[tuple[ClientProxy, FitRes]] = []

        for i, ((ciphertext, nonce, client_pubkey_bytes), n_examples) in enumerate(
            zip(encrypted_updates, num_examples)
        ):
            # ECDH with this client's public key
            client_pubkey = EllipticCurvePublicKey.from_encoded_point(
                SECP256K1(), client_pubkey_bytes
            )
            shared_secret = self._private_key.exchange(ECDH(), client_pubkey)
            symmetric_key = derive_symmetric_key(shared_secret, info=TEE_HKDF_INFO)

            # Decrypt model parameters
            plaintext = aes_gcm_decrypt(symmetric_key, nonce, ciphertext, TEE_AES_AAD)
            parameters = deserialize_parameters(plaintext)

            fit_res = FitRes(
                status=Status(code=0, message="OK"),
                parameters=parameters,
                num_examples=n_examples,
                metrics={},
            )
            results.append((_StubClientProxy(str(i)), fit_res))

        # Run Flower's FedAvg
        strategy = FedAvg()
        aggregated, _ = strategy.aggregate_fit(server_round, results, [])

        if aggregated is None:
            raise RuntimeError("FedAvg aggregation returned None")

        log.info(
            "MockTEEEnclave aggregated %d updates for round %d",
            len(results),
            server_round,
        )
        return serialize_parameters(aggregated)

    def destroy(self) -> None:
        self._private_key = None
        log.info("MockTEEEnclave destroyed")
