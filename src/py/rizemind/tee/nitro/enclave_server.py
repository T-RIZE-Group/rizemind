"""Aggregation server that runs INSIDE an AWS Nitro Enclave.

This module is the enclave's entry point.  It:

1. Generates an ECDH keypair.
2. Requests an attestation document from the Nitro Secure Module (NSM),
   embedding the public key.
3. Listens on vsock for aggregation requests from the parent instance.
4. For each request: decrypts trainer updates, calls Flower's
   ``FedAvg.aggregate_fit``, and returns the aggregated parameters.

Build into an EIF via ``Dockerfile.enclave`` and
``nitro-cli build-enclave``.
"""

import json
import logging
import socket
import struct

from cryptography.hazmat.primitives.asymmetric.ec import (
    ECDH,
    SECP256K1,
    EllipticCurvePublicKey,
    generate_private_key,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from flwr.common.typing import FitRes, Parameters, Status
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg

from rizemind.tee.crypto import (
    aes_gcm_decrypt,
    derive_symmetric_key,
    TEE_AES_AAD,
    TEE_HKDF_INFO,
)
from rizemind.tee.params import deserialize_parameters, serialize_parameters

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("enclave")

# vsock constants
VSOCK_PORT = 5000
AF_VSOCK = getattr(socket, "AF_VSOCK", 40)
VMADDR_CID_ANY = getattr(socket, "VMADDR_CID_ANY", 0xFFFFFFFF)
HEADER_FMT = "!Q"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
RECV_CHUNK = 65536

# Message types
MSG_GET_ATTESTATION = b"GET_ATTESTATION"
MSG_AGGREGATE = b"AGGREGATE"


class _StubClientProxy(ClientProxy):
    """Minimal ClientProxy stub so we can build ``(ClientProxy, FitRes)`` tuples
    that Flower's ``FedAvg.aggregate_fit`` expects."""

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


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    parts: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(remaining, RECV_CHUNK))
        if not chunk:
            raise ConnectionError(f"Connection closed with {remaining} bytes remaining")
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def _vsock_recv(sock: socket.socket) -> bytes:
    header = _recv_exact(sock, HEADER_SIZE)
    (length,) = struct.unpack(HEADER_FMT, header)
    return _recv_exact(sock, length)


def _vsock_send(sock: socket.socket, data: bytes) -> None:
    header = struct.pack(HEADER_FMT, len(data))
    sock.sendall(header + data)


def _get_nsm_attestation(public_key_der: bytes) -> bytes:
    """Request an attestation document from the Nitro Secure Module.

    This uses the NSM device (``/dev/nsm``) which is only available
    inside a running Nitro Enclave.  Falls back to a placeholder
    if the device is not present (useful for container-level testing).
    """
    try:
        # NSM library (aws-nsm) provides Python bindings
        import aws_nsm  # type: ignore[import-untyped]

        fd = aws_nsm.open_nsm_device()
        attestation = aws_nsm.get_attestation_document(
            fd, public_key=public_key_der
        )
        aws_nsm.close_nsm_device(fd)
        return attestation
    except (ImportError, OSError):
        log.warning("NSM device not available, returning placeholder attestation")
        return b"MOCK_ATTESTATION"


def _handle_aggregation(
    payload: bytes,
    private_key,
) -> bytes:
    """Decrypt trainer updates and aggregate using Flower's FedAvg."""
    # Payload format: JSON header with metadata, then binary chunks
    # Parse: { "n_updates": int, "server_round": int }
    # followed by n_updates * (ciphertext_len, ciphertext, nonce_len, nonce, pubkey_len, pubkey, num_examples)
    offset = 0
    (header_len,) = struct.unpack_from("!I", payload, offset)
    offset += 4
    header = json.loads(payload[offset : offset + header_len])
    offset += header_len

    n_updates = header["n_updates"]
    server_round = header["server_round"]

    results: list[tuple[ClientProxy, FitRes]] = []

    for i in range(n_updates):
        # Read ciphertext
        (ct_len,) = struct.unpack_from("!I", payload, offset)
        offset += 4
        ciphertext = payload[offset : offset + ct_len]
        offset += ct_len

        # Read nonce
        (nonce_len,) = struct.unpack_from("!I", payload, offset)
        offset += 4
        nonce = payload[offset : offset + nonce_len]
        offset += nonce_len

        # Read client public key
        (pk_len,) = struct.unpack_from("!I", payload, offset)
        offset += 4
        client_pubkey_bytes = payload[offset : offset + pk_len]
        offset += pk_len

        # Read num_examples
        (num_examples,) = struct.unpack_from("!I", payload, offset)
        offset += 4

        # ECDH: derive shared secret with this client
        client_pubkey = EllipticCurvePublicKey.from_encoded_point(
            SECP256K1(), client_pubkey_bytes
        )
        shared_secret = private_key.exchange(ECDH(), client_pubkey)
        symmetric_key = derive_symmetric_key(shared_secret, info=TEE_HKDF_INFO)

        # Decrypt
        plaintext = aes_gcm_decrypt(symmetric_key, nonce, ciphertext, TEE_AES_AAD)
        parameters = deserialize_parameters(plaintext)

        # Build FitRes for Flower
        fit_res = FitRes(
            status=Status(code=0, message="OK"),
            parameters=parameters,
            num_examples=num_examples,
            metrics={},
        )
        results.append((_StubClientProxy(str(i)), fit_res))

    # Run Flower's FedAvg aggregation
    strategy = FedAvg()
    aggregated, metrics = strategy.aggregate_fit(server_round, results, [])

    if aggregated is None:
        raise RuntimeError("FedAvg aggregation returned None")

    log.info(
        "Aggregated %d updates for round %d", len(results), server_round
    )
    return serialize_parameters(aggregated)


def main() -> None:
    """Enclave entry point: generate keys, serve on vsock."""
    # Generate ECDH keypair
    private_key = generate_private_key(SECP256K1())
    public_key_bytes = private_key.public_key().public_bytes(
        Encoding.X962, PublicFormat.UncompressedPoint
    )

    log.info("Enclave started, ECDH public key generated (%d bytes)", len(public_key_bytes))

    # Listen on vsock
    sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    sock.bind((VMADDR_CID_ANY, VSOCK_PORT))
    sock.listen(5)
    log.info("Listening on vsock port %d", VSOCK_PORT)

    while True:
        conn, addr = sock.accept()
        try:
            msg = _vsock_recv(conn)

            if msg == MSG_GET_ATTESTATION:
                # Return attestation document with embedded public key
                attestation = _get_nsm_attestation(public_key_bytes)
                # Send: public_key_len + public_key + attestation
                response = (
                    struct.pack("!I", len(public_key_bytes))
                    + public_key_bytes
                    + attestation
                )
                _vsock_send(conn, response)
                log.info("Sent attestation report")

            elif msg[:len(MSG_AGGREGATE)] == MSG_AGGREGATE:
                # Aggregation request: MSG_AGGREGATE + payload
                payload = msg[len(MSG_AGGREGATE):]
                result = _handle_aggregation(payload, private_key)
                _vsock_send(conn, result)

            else:
                log.warning("Unknown message type: %s", msg[:20])

        except Exception:
            log.exception("Error handling connection")
        finally:
            conn.close()


if __name__ == "__main__":
    main()
