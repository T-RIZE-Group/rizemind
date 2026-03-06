#!/usr/bin/env python3
"""End-to-end encrypted aggregation round-trip on Nitro enclave."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from eth_account import Account
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays


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
        description="Run encrypted two-client aggregation through Nitro enclave."
    )
    parser.add_argument(
        "--eif-path",
        default=str(repo_root / "enclave.eif"),
        help="Path to enclave EIF file.",
    )
    parser.add_argument(
        "--cpu-count",
        type=int,
        default=2,
        help="vCPU count for enclave.",
    )
    parser.add_argument(
        "--memory-mib",
        type=int,
        default=4096,
        help="Memory in MiB for enclave.",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Enable Nitro debug mode.",
    )
    parser.add_argument(
        "--mnemonic",
        default=DEFAULT_MNEMONIC,
        help="Mnemonic used to derive client wallet keys.",
    )
    parser.add_argument(
        "--output",
        default=str(repo_root / "results" / "nitro_round_trip.json"),
        help="Path to write round-trip artifact JSON.",
    )
    return parser.parse_args()


def _encrypt_update(
    *,
    weights: np.ndarray,
    num_examples: int,
    account_index: int,
    mnemonic: str,
    enclave_pubkey_bytes: bytes,
):
    from rizemind.tee.crypto import (
        aes_gcm_encrypt,
        derive_shared_secret,
        derive_symmetric_key,
        deserialize_public_key,
        ec_key_from_account,
        serialize_public_key,
    )
    from rizemind.tee.params import serialize_parameters

    account_path = f"{DEFAULT_ACCOUNT_PATH_PREFIX}{account_index}"
    account = Account.from_mnemonic(mnemonic, account_path=account_path)
    client_private = ec_key_from_account(account)
    client_pubkey = serialize_public_key(client_private.public_key())

    enclave_pubkey = deserialize_public_key(enclave_pubkey_bytes)
    shared_secret = derive_shared_secret(client_private, enclave_pubkey)
    symmetric_key = derive_symmetric_key(shared_secret)

    params = ndarrays_to_parameters([weights.astype(np.float32)])
    plaintext = serialize_parameters(params)
    ciphertext, nonce = aes_gcm_encrypt(symmetric_key, plaintext)

    return (ciphertext, nonce, client_pubkey), num_examples, account.address


def main() -> int:
    repo_root = _add_repo_to_python_path()
    args = _parse_args(repo_root)

    from rizemind.tee.nitro.nitro_enclave import NitroTEEEnclave
    from rizemind.tee.params import deserialize_parameters

    Account.enable_unaudited_hdwallet_features()

    eif_path = Path(args.eif_path).expanduser().resolve()
    if not eif_path.exists():
        print(f"[FAIL] EIF file not found: {eif_path}", file=sys.stderr)
        return 1

    enclave = NitroTEEEnclave(
        eif_path=str(eif_path),
        cpu_count=args.cpu_count,
        memory_mib=args.memory_mib,
        debug_mode=args.debug_mode,
    )

    initialized = False
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        enclave.initialize()
        initialized = True

        report = enclave.get_attestation_report()
        enclave_pubkey = enclave.get_public_key()
        enclave_id = getattr(enclave, "_enclave_id", None)
        enclave_cid = getattr(enclave, "_enclave_cid", None)

        print(f"[INFO] EnclaveID: {enclave_id}")
        print(f"[INFO] EnclaveCID: {enclave_cid}")
        print(f"[INFO] Public key length: {len(enclave_pubkey)} bytes")
        print(f"[INFO] Attestation length: {len(report.document)} bytes")

        # Two deterministic client updates
        update_a, n_a, address_a = _encrypt_update(
            weights=np.array([2.0, 4.0], dtype=np.float32),
            num_examples=100,
            account_index=0,
            mnemonic=args.mnemonic,
            enclave_pubkey_bytes=enclave_pubkey,
        )
        update_b, n_b, address_b = _encrypt_update(
            weights=np.array([6.0, 8.0], dtype=np.float32),
            num_examples=100,
            account_index=1,
            mnemonic=args.mnemonic,
            enclave_pubkey_bytes=enclave_pubkey,
        )

        result_bytes = enclave.aggregate(
            encrypted_updates=[update_a, update_b],
            num_examples=[n_a, n_b],
            server_round=1,
        )

        aggregated_params = deserialize_parameters(result_bytes)
        aggregated = parameters_to_ndarrays(aggregated_params)[0]
        expected = np.array([4.0, 6.0], dtype=np.float32)
        np.testing.assert_allclose(aggregated, expected, rtol=1e-6, atol=1e-6)

        print("[PASS] Encrypted aggregation round-trip succeeded.")
        print(f"[INFO] Aggregated result: {aggregated.tolist()}")

        artifact = {
            "enclave_id": enclave_id,
            "enclave_cid": enclave_cid,
            "attestation_len": len(report.document),
            "client_addresses": [address_a, address_b],
            "num_examples": [n_a, n_b],
            "aggregated": aggregated.tolist(),
            "expected": expected.tolist(),
            "server_round": 1,
        }
        output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        print(f"[PASS] Wrote artifact: {output_path}")
        return 0
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    finally:
        if initialized:
            enclave.destroy()
            print("[INFO] Enclave terminated.")


if __name__ == "__main__":
    raise SystemExit(main())
