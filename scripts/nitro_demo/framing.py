#!/usr/bin/env python3
"""Length-prefixed framing for socket/vsock message exchange."""

from __future__ import annotations

import socket
import struct

HEADER_SIZE = 4
MAX_FRAME_SIZE = 4 * 1024 * 1024


def send_frame(sock: socket.socket, payload: bytes) -> None:
    """Send one frame with 4-byte big-endian length prefix."""
    frame_len = len(payload)
    header = struct.pack(">I", frame_len)
    sock.sendall(header + payload)


def recv_exact(sock: socket.socket, num_bytes: int) -> bytes | None:
    """Receive exactly ``num_bytes`` or ``None`` if clean EOF at start."""
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")
    received = bytearray()
    while len(received) < num_bytes:
        chunk = sock.recv(num_bytes - len(received))
        if not chunk:
            if not received:
                return None
            raise ConnectionError("Connection closed while receiving frame")
        received.extend(chunk)
    return bytes(received)


def recv_frame(sock: socket.socket, max_frame_size: int = MAX_FRAME_SIZE) -> bytes | None:
    """Receive one frame. Returns ``None`` when peer closes cleanly."""
    if max_frame_size <= 0:
        raise ValueError("max_frame_size must be positive")
    header = recv_exact(sock, HEADER_SIZE)
    if header is None:
        return None
    frame_len = struct.unpack(">I", header)[0]
    if frame_len > max_frame_size:
        raise ValueError(f"Frame length {frame_len} exceeds limit {max_frame_size}")
    if frame_len == 0:
        return b""
    payload = recv_exact(sock, frame_len)
    if payload is None:
        raise ConnectionError("Connection closed before frame payload was read")
    return payload
