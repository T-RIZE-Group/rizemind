#!/usr/bin/env python3
"""Initialize demo sender/enclave key material once and reuse it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import common  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate secp256k1 keys once for the Nitro ECDH demo.",
    )
    parser.add_argument(
        "--keys-path",
        default=str(common.default_keys_path()),
        help="Path to keys.json (default: data/nitro_demo/keys.json).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    keys_path = Path(args.keys_path)
    try:
        existed = keys_path.exists()
        keys = common.ensure_key_material(keys_path)
        status = "reused" if existed else "created"
        output = {
            "ok": True,
            "status": status,
            "keys_path": str(keys_path),
            "sender_public_key_hex": keys[common.SENDER_PUBLIC_KEY_FIELD],
            "enclave_public_key_hex": keys[common.ENCLAVE_PUBLIC_KEY_FIELD],
        }
        print(json.dumps(output))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps(common.build_error_response(common.exception_message(exc))))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
