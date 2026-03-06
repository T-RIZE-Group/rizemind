#!/usr/bin/env python3
"""Start Nitro enclave, fetch attestation/public key, and persist artifact."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _add_repo_to_python_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    src_path = repo_root / "src" / "py"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return repo_root


def _parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Nitro enclave and fetch attestation/public key."
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
        help="vCPU count for nitro-cli run-enclave.",
    )
    parser.add_argument(
        "--memory-mib",
        type=int,
        default=4096,
        help="Memory allocation in MiB for enclave.",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Enable enclave debug mode.",
    )
    parser.add_argument(
        "--output",
        default=str(repo_root / "results" / "nitro_attestation.json"),
        help="Path to write attestation artifact JSON.",
    )
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="Keep enclave running after data fetch.",
    )
    return parser.parse_args()


def main() -> int:
    repo_root = _add_repo_to_python_path()
    args = _parse_args(repo_root)

    from rizemind.tee.nitro.nitro_enclave import NitroTEEEnclave

    eif_path = Path(args.eif_path).expanduser().resolve()
    if not eif_path.exists():
        print(f"[FAIL] EIF file not found: {eif_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    enclave = NitroTEEEnclave(
        eif_path=str(eif_path),
        cpu_count=args.cpu_count,
        memory_mib=args.memory_mib,
        debug_mode=args.debug_mode,
    )

    initialized = False

    try:
        enclave.initialize()
        initialized = True

        report = enclave.get_attestation_report()
        public_key = enclave.get_public_key()

        if not public_key:
            print("[FAIL] Enclave public key is empty.", file=sys.stderr)
            return 1
        if not report.document:
            print("[FAIL] Attestation document is empty.", file=sys.stderr)
            return 1

        enclave_id = getattr(enclave, "_enclave_id", None)
        enclave_cid = getattr(enclave, "_enclave_cid", None)

        payload = {
            "timestamp": int(time.time()),
            "eif_path": str(eif_path),
            "enclave_id": enclave_id,
            "enclave_cid": enclave_cid,
            "platform": report.platform,
            "public_key_hex": public_key.hex(),
            "public_key_len": len(public_key),
            "attestation_hex": report.document.hex(),
            "attestation_len": len(report.document),
        }

        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print("[PASS] Enclave initialized and attestation fetched.")
        print(f"[INFO] EnclaveID: {enclave_id}")
        print(f"[INFO] EnclaveCID: {enclave_cid}")
        print(f"[INFO] Public key length: {len(public_key)} bytes")
        print(f"[INFO] Attestation length: {len(report.document)} bytes")
        print(f"[PASS] Wrote artifact: {output_path}")

        if args.keep_running:
            print("[INFO] Enclave left running (--keep-running enabled).")
        return 0
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    finally:
        if initialized and not args.keep_running:
            enclave.destroy()
            print("[INFO] Enclave terminated after fetch.")


if __name__ == "__main__":
    raise SystemExit(main())
