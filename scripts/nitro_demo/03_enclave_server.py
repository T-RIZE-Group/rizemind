#!/usr/bin/env python3
"""Minimal enclave-side server: decrypt number, increment, and respond."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import common  # noqa: E402
from framing import recv_frame, send_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run minimal Nitro demo enclave server over vsock.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=common.VSOCK_PORT,
        help=f"Vsock port (default: {common.VSOCK_PORT}).",
    )
    parser.add_argument(
        "--keys-path",
        default=str(common.default_keys_path()),
        help="Path to keys.json containing enclave private key.",
    )
    return parser.parse_args()


def _require_vsock() -> None:
    if not hasattr(socket, "AF_VSOCK"):
        raise RuntimeError("AF_VSOCK is unavailable in this Python runtime")


def _decode_envelope(frame: bytes) -> dict[str, Any]:
    try:
        obj = json.loads(frame.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Incoming frame is not valid UTF-8 JSON") from exc
    if not isinstance(obj, dict):
        raise ValueError("Incoming frame JSON must be an object")
    return obj


def handle_client(
    conn: socket.socket,
    enclave_private_key_hex: str,
) -> None:
    while True:
        frame = recv_frame(conn)
        if frame is None:
            return
        try:
            envelope = _decode_envelope(frame)
            number = common.decrypt_number(envelope, enclave_private_key_hex)
            response = common.build_success_response(number + 1)
        except Exception as exc:  # noqa: BLE001
            response = common.build_error_response(common.exception_message(exc))
        send_frame(conn, json.dumps(response).encode("utf-8"))


def main() -> int:
    args = parse_args()
    keys_path = Path(args.keys_path)
    try:
        _require_vsock()
        keys = common.load_key_material(keys_path)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps(common.build_error_response(common.exception_message(exc))))
        return 1

    enclave_private_key_hex = keys[common.ENCLAVE_PRIVATE_KEY_FIELD]
    enclave_public_key_hex = keys[common.ENCLAVE_PUBLIC_KEY_FIELD]

    cid_any = getattr(socket, "VMADDR_CID_ANY", 0xFFFFFFFF)
    server = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
    server.bind((cid_any, args.port))
    server.listen(16)
    print(
        json.dumps(
            {
                "ok": True,
                "status": "listening",
                "port": args.port,
                "enclave_public_key_hex": enclave_public_key_hex,
            }
        )
    )

    try:
        while True:
            conn, _ = server.accept()
            with conn:
                handle_client(conn, enclave_private_key_hex)
    except KeyboardInterrupt:
        return 0
    finally:
        server.close()


if __name__ == "__main__":
    raise SystemExit(main())
