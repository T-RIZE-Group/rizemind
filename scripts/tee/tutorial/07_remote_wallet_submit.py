#!/usr/bin/env python3
"""Submit one wallet-encrypted model update to remote Nitro relay."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from eth_account import Account
from flwr.common.parameter import ndarrays_to_parameters


DEFAULT_MNEMONIC = "test test test test test test test test test test test junk"
DEFAULT_ACCOUNT_PATH_PREFIX = "m/44'/60'/0'/0/"


def _add_repo_to_python_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    src_path = repo_root / "src" / "py"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return repo_root


def _parse_weights(weights_arg: str) -> np.ndarray:
    try:
        values = [float(x.strip()) for x in weights_arg.split(",") if x.strip()]
    except Exception as exc:
        raise ValueError(f"invalid weights list: {exc}") from exc

    if not values:
        raise ValueError("weights list is empty")
    return np.array(values, dtype=np.float32)


def _http_json(
    *,
    method: str,
    url: str,
    payload: dict | None = None,
    token: str = "",
) -> dict:
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-API-Token"] = token

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as err:
        raw = err.read().decode("utf-8")
        raise RuntimeError(f"HTTP {err.code}: {raw}") from err


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encrypt one update with wallet key and submit to Nitro relay."
    )
    parser.add_argument(
        "--relay-url",
        default="http://127.0.0.1:8080",
        help="Relay base URL. Use SSH tunnel for remote EC2 access.",
    )
    parser.add_argument(
        "--api-token",
        default="",
        help="Optional X-API-Token for relay auth.",
    )
    parser.add_argument(
        "--mnemonic",
        default=DEFAULT_MNEMONIC,
        help="Mnemonic to derive wallet private key.",
    )
    parser.add_argument(
        "--account-index",
        type=int,
        default=0,
        help="Account index in m/44'/60'/0'/0/{index}.",
    )
    parser.add_argument(
        "--round-id",
        type=int,
        default=1,
        help="Round identifier for server-side queueing.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="num_examples metadata used by FedAvg weighting.",
    )
    parser.add_argument(
        "--weights",
        default="2.0,4.0",
        help="Comma-separated float list for single tensor update.",
    )
    parser.add_argument(
        "--metadata",
        default="",
        help='Optional JSON object string (example: \'{"source":"laptop-a"}\').',
    )
    return parser.parse_args()


def main() -> int:
    _add_repo_to_python_path()

    from rizemind.tee.crypto import (
        aes_gcm_encrypt,
        derive_shared_secret,
        derive_symmetric_key,
        deserialize_public_key,
        ec_key_from_account,
        serialize_public_key,
    )
    from rizemind.tee.params import serialize_parameters

    args = _parse_args()
    Account.enable_unaudited_hdwallet_features()

    base = args.relay_url.rstrip("/")

    # 1) Fetch attestation/public key
    attestation = _http_json(
        method="GET",
        url=f"{base}/attestation",
        token=args.api_token,
    )
    if not attestation.get("ok"):
        print(f"[FAIL] Relay /attestation error: {attestation}", file=sys.stderr)
        return 1

    pubkey_hex = attestation.get("public_key_hex", "")
    if not isinstance(pubkey_hex, str) or not pubkey_hex:
        print("[FAIL] Relay did not return public_key_hex.", file=sys.stderr)
        return 1
    enclave_pubkey = deserialize_public_key(bytes.fromhex(pubkey_hex))

    # 2) Build wallet key
    account_path = f"{DEFAULT_ACCOUNT_PATH_PREFIX}{args.account_index}"
    account = Account.from_mnemonic(args.mnemonic, account_path=account_path)
    wallet_private = ec_key_from_account(account)
    wallet_pubkey_bytes = serialize_public_key(wallet_private.public_key())

    # 3) Encrypt update
    weights = _parse_weights(args.weights)
    params = ndarrays_to_parameters([weights])
    plaintext = serialize_parameters(params)

    shared_secret = derive_shared_secret(wallet_private, enclave_pubkey)
    symmetric_key = derive_symmetric_key(shared_secret)
    ciphertext, nonce = aes_gcm_encrypt(symmetric_key, plaintext)

    metadata: dict[str, object] = {}
    if args.metadata:
        try:
            loaded = json.loads(args.metadata)
            if isinstance(loaded, dict):
                metadata = loaded
            else:
                raise ValueError("metadata JSON must be an object")
        except Exception as exc:
            print(f"[FAIL] Invalid --metadata JSON: {exc}", file=sys.stderr)
            return 1

    metadata = {
        **metadata,
        "wallet_address": account.address,
        "account_path": account_path,
    }

    # 4) Submit
    submit_resp = _http_json(
        method="POST",
        url=f"{base}/submit",
        token=args.api_token,
        payload={
            "round_id": args.round_id,
            "num_examples": args.num_examples,
            "ciphertext_hex": ciphertext.hex(),
            "nonce_hex": nonce.hex(),
            "client_pubkey_hex": wallet_pubkey_bytes.hex(),
            "metadata": metadata,
        },
    )

    if not submit_resp.get("ok"):
        print(f"[FAIL] Relay /submit error: {submit_resp}", file=sys.stderr)
        return 1

    print("[PASS] Encrypted update submitted.")
    print(f"[INFO] Relay: {base}")
    print(f"[INFO] Round: {args.round_id}")
    print(f"[INFO] Wallet address: {account.address}")
    print(f"[INFO] Ciphertext length: {len(ciphertext)} bytes")
    print(f"[INFO] Pending updates: {submit_resp.get('pending_updates')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
