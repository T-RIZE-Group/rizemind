#!/usr/bin/env python3
"""Local sender->enclave crypto round trip without Nitro runtime."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import common  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encrypt one number and locally decrypt it as enclave would.",
    )
    parser.add_argument("--number", type=int, required=True, help="Integer input value.")
    parser.add_argument(
        "--keys-path",
        default=str(common.default_keys_path()),
        help="Path to keys.json.",
    )
    parser.add_argument(
        "--tamper",
        action="store_true",
        help="Flip one ciphertext bit to prove AES-GCM auth failure.",
    )
    return parser.parse_args()


def _tamper_ciphertext(envelope: dict[str, str]) -> None:
    raw = bytearray(base64.b64decode(envelope["ciphertext_b64"]))
    if not raw:
        raise ValueError("Ciphertext is empty; cannot tamper")
    raw[-1] ^= 0x01
    envelope["ciphertext_b64"] = base64.b64encode(bytes(raw)).decode("ascii")


def main() -> int:
    args = parse_args()
    keys_path = Path(args.keys_path)
    try:
        keys = common.load_key_material(keys_path)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps(common.build_error_response(common.exception_message(exc))))
        return 1

    envelope = common.encrypt_number(
        args.number,
        keys[common.SENDER_PRIVATE_KEY_FIELD],
        keys[common.ENCLAVE_PUBLIC_KEY_FIELD],
    )
    if args.tamper:
        _tamper_ciphertext(envelope)

    try:
        decrypted = common.decrypt_number(
            envelope,
            keys[common.ENCLAVE_PRIVATE_KEY_FIELD],
        )
        response = {
            "ok": True,
            "input": decrypted,
            "result": decrypted + 1,
        }
        print(json.dumps(response))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps(common.build_error_response(common.exception_message(exc))))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
