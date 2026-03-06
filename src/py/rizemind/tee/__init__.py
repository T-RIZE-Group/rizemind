"""TEE-based secure aggregation for Rizemind federated learning.

Provides hardware-agnostic TEE integration for encrypted model aggregation
using ECDH key exchange and AES-256-GCM encryption.  Trainers encrypt their
model updates for the TEE enclave, which decrypts and runs Flower's native
aggregation (e.g. FedAvg) in a trusted environment.

Supported backends:
- **AWS Nitro Enclaves** (``rizemind.tee.nitro``)
- **Mock** (``MockTEEEnclave``) for development and testing
"""

from rizemind.tee.attestation import (
    AttestationVerifier,
    MockAttestationVerifier,
    NitroAttestationVerifier,
)
from rizemind.tee.crypto import ec_key_from_account
from rizemind.tee.enclave import AttestationReport, TEEEnclave
from rizemind.tee.mock_enclave import MockTEEEnclave

try:
    # Optional at import time: avoids forcing all client/server dependencies
    # when callers only need enclave lifecycle utilities.
    from rizemind.tee.tee_client_mod import tee_encryption_mod
except ImportError:  # pragma: no cover - environment-dependent optional import
    tee_encryption_mod = None  # type: ignore[assignment]

try:
    from rizemind.tee.tee_strategy import TEEAggregationStrategy
except ImportError:  # pragma: no cover - environment-dependent optional import
    TEEAggregationStrategy = None  # type: ignore[assignment]

__all__ = [
    "AttestationReport",
    "AttestationVerifier",
    "MockAttestationVerifier",
    "MockTEEEnclave",
    "NitroAttestationVerifier",
    "TEEAggregationStrategy",
    "TEEEnclave",
    "ec_key_from_account",
    "tee_encryption_mod",
]
