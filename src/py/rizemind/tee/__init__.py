"""TEE-based secure aggregation exports."""

from rizemind.tee.attestation import (
    AttestationVerifier,
    MockAttestationVerifier,
    NitroAttestationVerifier,
)
from rizemind.tee.crypto import ec_key_from_account
from rizemind.tee.enclave import AttestationReport, TEEEnclave
from rizemind.tee.mock_enclave import MockTEEEnclave
from rizemind.tee.tee_strategy import TEEAggregationStrategy

__all__ = [
    "AttestationReport",
    "AttestationVerifier",
    "MockAttestationVerifier",
    "MockTEEEnclave",
    "NitroAttestationVerifier",
    "TEEAggregationStrategy",
    "TEEEnclave",
    "ec_key_from_account",
]

try:
    from rizemind.tee.tee_client_mod import tee_encryption_mod
except Exception:
    tee_encryption_mod = None  # optional runtime import
else:
    __all__.append("tee_encryption_mod")
