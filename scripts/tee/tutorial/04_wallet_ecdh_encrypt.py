#!/usr/bin/env python3
"""Wallet-key ECDH walkthrough (client-side cryptography only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ec import SECP256K1, derive_private_key
from eth_account import Account


DEFAULT_MNEMONIC = "test test test test test test test test test test test junk"
DEFAULT_ACCOUNT_PATH_PREFIX = "m/44'/60'/0'/0/"


def _add_repo_to_python_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    src_path = repo_root / "src" / "py"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return repo_root


def _parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate wallet-private-key ECDH and AES-GCM encryption."
    )
    parser.add_argument(
        "--mnemonic",
        default=DEFAULT_MNEMONIC,
        help="BIP-39 mnemonic to derive wallet account.",
    )
    parser.add_argument(
        "--account-index",
        type=int,
        default=0,
        help="Account index in m/44'/60'/0'/0/{index}.",
    )
    parser.add_argument(
        "--payload",
        default="hello from wallet client",
        help="Plaintext payload for encryption demo.",
    )
    parser.add_argument(
        "--enclave-pubkey-hex",
        default="",
        help="Optional enclave public key hex for target encryption output.",
    )
    parser.add_argument(
        "--attestation-json",
        default=str(repo_root / "results" / "nitro_attestation.json"),
        help="Optional attestation artifact path from Step 3.",
    )
    parser.add_argument(
        "--output",
        default=str(repo_root / "results" / "wallet_ecdh_demo.json"),
        help="Where to write demo artifact JSON.",
    )
    return parser.parse_args()


def _load_target_pubkey(args: argparse.Namespace) -> tuple[bytes | None, str]:
    if args.enclave_pubkey_hex:
        return bytes.fromhex(args.enclave_pubkey_hex), "arg:--enclave-pubkey-hex"

    attestation_path = Path(args.attestation_json).expanduser().resolve()
    if attestation_path.exists():
        data = json.loads(attestation_path.read_text(encoding="utf-8"))
        pub_hex = data.get("public_key_hex", "")
        if isinstance(pub_hex, str) and pub_hex:
            return bytes.fromhex(pub_hex), f"artifact:{attestation_path}"

    return None, "none"


def main() -> int:
    _add_repo_to_python_path()

    from rizemind.tee.crypto import (
        aes_gcm_decrypt,
        aes_gcm_encrypt,
        derive_shared_secret,
        derive_symmetric_key,
        deserialize_public_key,
        ec_key_from_account,
        serialize_public_key,
    )

    args = _parse_args(Path(__file__).resolve().parents[3])
    Account.enable_unaudited_hdwallet_features()

    account_path = f"{DEFAULT_ACCOUNT_PATH_PREFIX}{args.account_index}"
    account = Account.from_mnemonic(args.mnemonic, account_path=account_path)
    wallet_private = ec_key_from_account(account)
    wallet_public = serialize_public_key(wallet_private.public_key())
    payload = args.payload.encode("utf-8")

    print(f"[INFO] Wallet address: {account.address}")
    print(f"[INFO] Account path: {account_path}")
    print("[INFO] Running control round-trip with local enclave simulator key...")

    # Deterministic local enclave simulator key so this step is reproducible
    simulated_enclave_private = derive_private_key(0xA11CE, SECP256K1())
    simulated_enclave_public = simulated_enclave_private.public_key()

    shared_client = derive_shared_secret(wallet_private, simulated_enclave_public)
    key_client = derive_symmetric_key(shared_client)
    ciphertext, nonce = aes_gcm_encrypt(key_client, payload)

    shared_enclave = derive_shared_secret(simulated_enclave_private, wallet_private.public_key())
    key_enclave = derive_symmetric_key(shared_enclave)
    recovered = aes_gcm_decrypt(key_enclave, nonce, ciphertext)

    if recovered != payload:
        print("[FAIL] Control decrypt check failed.", file=sys.stderr)
        return 1

    print("[PASS] Control decrypt check succeeded.")
    print(f"[INFO] Control ciphertext length: {len(ciphertext)} bytes")
    print(f"[INFO] Control nonce length: {len(nonce)} bytes")

    target_pubkey_bytes, target_source = _load_target_pubkey(args)
    target_data: dict[str, str | int | bool] = {"available": False}

    if target_pubkey_bytes is not None:
        target_pubkey = deserialize_public_key(target_pubkey_bytes)
        target_shared = derive_shared_secret(wallet_private, target_pubkey)
        target_key = derive_symmetric_key(target_shared)
        target_ciphertext, target_nonce = aes_gcm_encrypt(target_key, payload)
        target_data = {
            "available": True,
            "source": target_source,
            "target_pubkey_hex": target_pubkey_bytes.hex(),
            "target_ciphertext_hex": target_ciphertext.hex(),
            "target_ciphertext_len": len(target_ciphertext),
            "target_nonce_hex": target_nonce.hex(),
            "target_nonce_len": len(target_nonce),
        }
        print(f"[PASS] Target encryption generated using {target_source}.")
    else:
        print("[INFO] No target enclave public key found. Control demo only.")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "wallet_address": account.address,
        "account_path": account_path,
        "wallet_public_key_hex": wallet_public.hex(),
        "control": {
            "ciphertext_hex": ciphertext.hex(),
            "ciphertext_len": len(ciphertext),
            "nonce_hex": nonce.hex(),
            "nonce_len": len(nonce),
            "recovered_plaintext": recovered.decode("utf-8"),
        },
        "target": target_data,
    }
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(f"[PASS] Wrote artifact: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
