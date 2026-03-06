"""Abstract TEE enclave interface and attestation report dataclass.

Provides a hardware-agnostic abstraction over Trusted Execution Environments.
Concrete implementations exist for AWS Nitro Enclaves (``nitro/``) and a
software mock (``mock_enclave.py``).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AttestationReport:
    """A TEE attestation report proving enclave identity and code integrity.

    Attributes:
        enclave_public_key: The enclave's ECDH public key (serialized X9.62).
        document: Raw platform-specific attestation evidence (e.g. CBOR for Nitro).
        platform: Identifier for the TEE platform (``"nitro"``, ``"mock"``).
        pcrs: Platform Configuration Register values keyed by index.
            For Nitro: PCR0 = enclave image, PCR1 = kernel, PCR2 = application.
        timestamp: Unix timestamp when the attestation was generated.
    """

    enclave_public_key: bytes
    document: bytes
    platform: str
    pcrs: dict[int, bytes] = field(default_factory=dict)
    timestamp: float = 0.0


class TEEEnclave(ABC):
    """Abstract interface for a TEE enclave that performs secure aggregation.

    The lifecycle is: ``initialize`` → (``get_attestation_report``, ``aggregate``) → ``destroy``.
    """

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the enclave, generating its ECDH keypair.

        For Nitro this starts the enclave process and connects via vsock.
        For the mock this generates keys in-process.
        """
        ...

    @abstractmethod
    def get_attestation_report(self) -> AttestationReport:
        """Get a remote attestation report proving enclave identity.

        The report binds the enclave's public ECDH key to the platform
        attestation so clients can verify the key belongs to a genuine TEE.
        """
        ...

    @abstractmethod
    def get_public_key(self) -> bytes:
        """Return the enclave's ECDH public key (serialized X9.62 uncompressed)."""
        ...

    @abstractmethod
    def aggregate(
        self,
        encrypted_updates: list[tuple[bytes, bytes, bytes]],
        num_examples: list[int],
        server_round: int,
    ) -> bytes:
        """Send encrypted model updates to the enclave for aggregation.

        The enclave decrypts each update using ECDH with the corresponding
        client public key, runs Flower's ``aggregate_fit``, and returns the
        serialized aggregated ``Parameters``.

        Args:
            encrypted_updates: List of ``(ciphertext, nonce, client_public_key)`` tuples.
            num_examples: Number of training examples per client (for weighted averaging).
            server_round: The current server round number.

        Returns:
            Serialized aggregated ``Parameters`` bytes.
        """
        ...

    @abstractmethod
    def destroy(self) -> None:
        """Destroy the enclave, wiping all keys and state."""
        ...
