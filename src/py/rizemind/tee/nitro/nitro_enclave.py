"""Parent-side Nitro Enclave management.

Implements ``TEEEnclave`` by managing the Nitro Enclave lifecycle
via ``nitro-cli`` and communicating with the enclave over vsock.
"""

import json
import logging
import socket
import struct
import subprocess

from rizemind.tee.enclave import AttestationReport, TEEEnclave
from rizemind.tee.vsock import ENCLAVE_VSOCK_PORT, vsock_recv, vsock_send

log = logging.getLogger(__name__)

# Message types matching enclave_server.py
MSG_GET_ATTESTATION = b"GET_ATTESTATION"
MSG_AGGREGATE = b"AGGREGATE"


class NitroTEEEnclave(TEEEnclave):
    """Manages an AWS Nitro Enclave from the parent EC2 instance.

    Args:
        eif_path: Path to the Enclave Image File (``.eif``).
        cpu_count: Number of vCPUs to allocate to the enclave.
        memory_mib: Memory in MiB to allocate to the enclave.
        debug_mode: If True, enable enclave debug console.
    """

    def __init__(
        self,
        eif_path: str,
        cpu_count: int = 2,
        memory_mib: int = 4096,
        debug_mode: bool = False,
    ) -> None:
        self._eif_path = eif_path
        self._cpu_count = cpu_count
        self._memory_mib = memory_mib
        self._debug_mode = debug_mode
        self._enclave_id: str | None = None
        self._enclave_cid: int | None = None
        self._public_key: bytes | None = None
        self._attestation: AttestationReport | None = None

    def initialize(self) -> None:
        """Start the Nitro Enclave and retrieve its attestation."""
        cmd = [
            "nitro-cli", "run-enclave",
            "--eif-path", self._eif_path,
            "--cpu-count", str(self._cpu_count),
            "--memory", str(self._memory_mib),
        ]
        if self._debug_mode:
            cmd.append("--debug-mode")

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        self._enclave_id = info["EnclaveID"]
        self._enclave_cid = info["EnclaveCID"]

        log.info(
            "Nitro Enclave started: id=%s cid=%d",
            self._enclave_id,
            self._enclave_cid,
        )

        # Request attestation from enclave
        self._fetch_attestation()

    def _fetch_attestation(self) -> None:
        """Connect to enclave over vsock and fetch attestation + public key."""
        sock = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
        try:
            sock.connect((self._enclave_cid, ENCLAVE_VSOCK_PORT))
            vsock_send(sock, MSG_GET_ATTESTATION)
            response = vsock_recv(sock)
        finally:
            sock.close()

        # Parse response: [pubkey_len: 4B][pubkey][attestation_doc]
        (pk_len,) = struct.unpack_from("!I", response, 0)
        self._public_key = response[4 : 4 + pk_len]
        attestation_doc = response[4 + pk_len :]

        self._attestation = AttestationReport(
            enclave_public_key=self._public_key,
            document=attestation_doc,
            platform="nitro",
        )
        log.info("Received attestation from enclave")

    def get_attestation_report(self) -> AttestationReport:
        if self._attestation is None:
            raise RuntimeError("Enclave not initialized")
        return self._attestation

    def get_public_key(self) -> bytes:
        if self._public_key is None:
            raise RuntimeError("Enclave not initialized")
        return self._public_key

    def aggregate(
        self,
        encrypted_updates: list[tuple[bytes, bytes, bytes]],
        num_examples: list[int],
        server_round: int,
    ) -> bytes:
        """Send encrypted updates to the enclave for aggregation via vsock."""
        payload = self._build_aggregate_payload(
            encrypted_updates, num_examples, server_round
        )

        sock = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
        try:
            sock.connect((self._enclave_cid, ENCLAVE_VSOCK_PORT))
            vsock_send(sock, MSG_AGGREGATE + payload)
            result = vsock_recv(sock)
        finally:
            sock.close()

        log.info(
            "Enclave aggregated %d updates for round %d",
            len(encrypted_updates),
            server_round,
        )
        return result

    @staticmethod
    def _build_aggregate_payload(
        encrypted_updates: list[tuple[bytes, bytes, bytes]],
        num_examples: list[int],
        server_round: int,
    ) -> bytes:
        """Build the binary payload for an aggregation request.

        Format matches what ``enclave_server._handle_aggregation`` expects.
        """
        header = json.dumps({
            "n_updates": len(encrypted_updates),
            "server_round": server_round,
        }).encode("utf-8")

        parts: list[bytes] = [struct.pack("!I", len(header)), header]

        for (ciphertext, nonce, client_pubkey), n_examples in zip(
            encrypted_updates, num_examples
        ):
            parts.append(struct.pack("!I", len(ciphertext)))
            parts.append(ciphertext)
            parts.append(struct.pack("!I", len(nonce)))
            parts.append(nonce)
            parts.append(struct.pack("!I", len(client_pubkey)))
            parts.append(client_pubkey)
            parts.append(struct.pack("!I", n_examples))

        return b"".join(parts)

    def destroy(self) -> None:
        """Terminate the Nitro Enclave."""
        if self._enclave_id:
            try:
                subprocess.run(
                    ["nitro-cli", "terminate-enclave", "--enclave-id", self._enclave_id],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                log.info("Nitro Enclave terminated: %s", self._enclave_id)
            except subprocess.CalledProcessError:
                log.exception("Failed to terminate enclave %s", self._enclave_id)
            finally:
                self._enclave_id = None
                self._enclave_cid = None
                self._public_key = None
                self._attestation = None
