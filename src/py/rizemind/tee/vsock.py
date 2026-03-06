"""Vsock communication helpers for Nitro Enclave ↔ parent instance.

Provides length-framed send/recv over ``socket.AF_VSOCK`` so that
arbitrarily large model payloads can be reliably transferred.

Wire format per message::

    [payload_length: 8 bytes, big-endian uint64][payload: N bytes]
"""

import socket
import struct

ENCLAVE_VSOCK_PORT = 5000
HEADER_FMT = "!Q"  # 8-byte unsigned long long, big-endian
HEADER_SIZE = struct.calcsize(HEADER_FMT)
RECV_CHUNK = 65536


def vsock_send(sock: socket.socket, data: bytes) -> None:
    """Send a length-framed message over a vsock connection."""
    header = struct.pack(HEADER_FMT, len(data))
    sock.sendall(header + data)


def vsock_recv(sock: socket.socket) -> bytes:
    """Receive a length-framed message from a vsock connection."""
    header = _recv_exact(sock, HEADER_SIZE)
    (length,) = struct.unpack(HEADER_FMT, header)
    return _recv_exact(sock, length)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly *n* bytes from *sock*, raising on premature close."""
    parts: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(remaining, RECV_CHUNK))
        if not chunk:
            raise ConnectionError(
                f"Connection closed with {remaining} bytes remaining"
            )
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)
