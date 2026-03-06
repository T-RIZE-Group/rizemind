#!/usr/bin/env python3
"""Parent-side vsock client for minimal Nitro number increment demo."""

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
        description="Encrypt one number and send it to Nitro enclave over vsock.",
    )
    parser.add_argument(
        "--cid",
        type=int,
        required=True,
        help="Target enclave CID (from nitro-cli run-enclave output).",
    )
    parser.add_argument("--number", type=int, required=True, help="Integer input value.")
    parser.add_argument(
        "--port",
        type=int,
        default=common.VSOCK_PORT,
        help=f"Vsock port (default: {common.VSOCK_PORT}).",
    )
    parser.add_argument(
        "--keys-path",
        default=str(common.default_keys_path()),
        help="Path to keys.json.",
    )
    return parser.parse_args()


def _require_vsock() -> None:
    if not hasattr(socket, "AF_VSOCK"):
        raise RuntimeError("AF_VSOCK is unavailable in this Python runtime")


def _decode_response(frame: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(frame.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Response is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Response JSON must be an object")
    return payload


def main() -> int:
    args = parse_args()
    keys_path = Path(args.keys_path)
    try:
        _require_vsock()
        keys = common.load_key_material(keys_path)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps(common.build_error_response(common.exception_message(exc))))
        return 1

    envelope = common.encrypt_number(
        args.number,
        keys[common.SENDER_PRIVATE_KEY_FIELD],
        keys[common.ENCLAVE_PUBLIC_KEY_FIELD],
    )
    request_bytes = json.dumps(envelope).encode("utf-8")

    sock = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
    try:
        sock.connect((args.cid, args.port))
        send_frame(sock, request_bytes)
        frame = recv_frame(sock)
        if frame is None:
            print(json.dumps(common.build_error_response("No response from enclave")))
            return 1
        response = _decode_response(frame)
        print(json.dumps(response))
        return 0 if response.get("ok") is True else 1
    except Exception as exc:  # noqa: BLE001
        print(json.dumps(common.build_error_response(common.exception_message(exc))))
        return 1
    finally:
        sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
