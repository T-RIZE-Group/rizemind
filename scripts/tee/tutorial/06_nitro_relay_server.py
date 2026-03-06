#!/usr/bin/env python3
"""Minimal HTTP relay: laptop -> EC2 parent -> Nitro enclave."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _add_repo_to_python_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    src_path = repo_root / "src" / "py"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return repo_root


def _parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Nitro relay server for remote encrypted update submissions."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host. Keep 127.0.0.1 and use SSH tunnel for safety.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP listen port.",
    )
    parser.add_argument(
        "--eif-path",
        default=str(repo_root / "enclave.eif"),
        help="Path to EIF used for Nitro enclave startup.",
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
        "--api-token",
        default="",
        help="Optional static API token. If set, clients must send X-API-Token header.",
    )
    return parser.parse_args()


def _json_response(
    handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]
) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(content_length) if content_length > 0 else b"{}"
    return json.loads(raw.decode("utf-8"))


def _require_token(handler: BaseHTTPRequestHandler, api_token: str) -> bool:
    if not api_token:
        return True
    provided = handler.headers.get("X-API-Token", "")
    if provided != api_token:
        _json_response(
            handler,
            HTTPStatus.UNAUTHORIZED,
            {"ok": False, "error": "invalid_or_missing_token"},
        )
        return False
    return True


@dataclass
class UpdateBundle:
    ciphertext: bytes
    nonce: bytes
    client_pubkey: bytes
    num_examples: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelayState:
    api_token: str
    enclave: Any
    lock: threading.Lock = field(default_factory=threading.Lock)
    pending_by_round: dict[int, list[UpdateBundle]] = field(default_factory=dict)

    def submit(self, round_id: int, update: UpdateBundle) -> int:
        with self.lock:
            self.pending_by_round.setdefault(round_id, []).append(update)
            return len(self.pending_by_round[round_id])

    def aggregate(self, round_id: int, min_updates: int) -> tuple[bytes, int]:
        with self.lock:
            updates = self.pending_by_round.get(round_id, [])
            if len(updates) < min_updates:
                raise ValueError(f"not_enough_updates:{len(updates)}:{min_updates}")

            encrypted_updates = [
                (u.ciphertext, u.nonce, u.client_pubkey) for u in updates
            ]
            num_examples = [u.num_examples for u in updates]

        result = self.enclave.aggregate(
            encrypted_updates=encrypted_updates,
            num_examples=num_examples,
            server_round=round_id,
        )

        with self.lock:
            self.pending_by_round[round_id] = []

        return result, len(updates)

    def pending_count(self, round_id: int) -> int:
        with self.lock:
            return len(self.pending_by_round.get(round_id, []))


def _make_handler(state: RelayState):
    class RelayHandler(BaseHTTPRequestHandler):
        def _json_ok(self, payload: dict[str, Any], status: int = 200) -> None:
            _json_response(self, status, {"ok": True, **payload})

        def _json_error(self, status: int, code: str, message: str) -> None:
            _json_response(
                self,
                status,
                {"ok": False, "error": code, "message": message},
            )

        def do_GET(self) -> None:  # noqa: N802
            if not _require_token(self, state.api_token):
                return

            if self.path == "/health":
                enclave_id = getattr(state.enclave, "_enclave_id", None)
                enclave_cid = getattr(state.enclave, "_enclave_cid", None)
                self._json_ok({"status": "ok", "enclave_id": enclave_id, "enclave_cid": enclave_cid})
                return

            if self.path == "/attestation":
                report = state.enclave.get_attestation_report()
                pubkey = state.enclave.get_public_key()
                self._json_ok(
                    {
                        "platform": report.platform,
                        "public_key_hex": pubkey.hex(),
                        "public_key_len": len(pubkey),
                        "attestation_hex": report.document.hex(),
                        "attestation_len": len(report.document),
                    }
                )
                return

            self._json_error(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint")

        def do_POST(self) -> None:  # noqa: N802
            if not _require_token(self, state.api_token):
                return

            try:
                body = _read_json_body(self)
            except Exception as exc:
                self._json_error(HTTPStatus.BAD_REQUEST, "bad_json", str(exc))
                return

            if self.path == "/submit":
                try:
                    round_id = int(body["round_id"])
                    num_examples = int(body["num_examples"])
                    ciphertext = bytes.fromhex(body["ciphertext_hex"])
                    nonce = bytes.fromhex(body["nonce_hex"])
                    client_pubkey = bytes.fromhex(body["client_pubkey_hex"])
                    metadata = body.get("metadata", {})
                except Exception as exc:
                    self._json_error(HTTPStatus.BAD_REQUEST, "invalid_submit_payload", str(exc))
                    return

                pending = state.submit(
                    round_id,
                    UpdateBundle(
                        ciphertext=ciphertext,
                        nonce=nonce,
                        client_pubkey=client_pubkey,
                        num_examples=num_examples,
                        metadata=metadata if isinstance(metadata, dict) else {},
                    ),
                )
                self._json_ok({"round_id": round_id, "pending_updates": pending})
                return

            if self.path == "/aggregate":
                try:
                    round_id = int(body["round_id"])
                    min_updates = int(body.get("min_updates", 1))
                except Exception as exc:
                    self._json_error(HTTPStatus.BAD_REQUEST, "invalid_aggregate_payload", str(exc))
                    return

                try:
                    result_bytes, used = state.aggregate(round_id, min_updates)
                except ValueError as exc:
                    # not_enough_updates:have:need
                    text = str(exc)
                    if text.startswith("not_enough_updates:"):
                        _, have, need = text.split(":")
                        self._json_error(
                            HTTPStatus.CONFLICT,
                            "not_enough_updates",
                            f"have={have} need={need}",
                        )
                        return
                    self._json_error(HTTPStatus.BAD_REQUEST, "aggregate_error", text)
                    return
                except Exception as exc:
                    self._json_error(HTTPStatus.INTERNAL_SERVER_ERROR, "aggregate_failed", str(exc))
                    return

                self._json_ok(
                    {
                        "round_id": round_id,
                        "used_updates": used,
                        "aggregated_params_hex": result_bytes.hex(),
                    }
                )
                return

            if self.path == "/pending":
                try:
                    round_id = int(body["round_id"])
                except Exception as exc:
                    self._json_error(HTTPStatus.BAD_REQUEST, "invalid_pending_payload", str(exc))
                    return
                self._json_ok({"round_id": round_id, "pending_updates": state.pending_count(round_id)})
                return

            self._json_error(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint")

        def log_message(self, fmt: str, *args: object) -> None:
            # compact server logs
            print(f"[HTTP] {self.address_string()} - {fmt % args}")

    return RelayHandler


def main() -> int:
    repo_root = _add_repo_to_python_path()
    args = _parse_args(repo_root)

    from rizemind.tee.nitro.nitro_enclave import NitroTEEEnclave

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

    try:
        enclave.initialize()
        report = enclave.get_attestation_report()
        pubkey = enclave.get_public_key()
    except Exception as exc:
        print(f"[FAIL] Could not initialize enclave: {exc}", file=sys.stderr)
        return 1

    state = RelayState(api_token=args.api_token, enclave=enclave)
    handler_cls = _make_handler(state)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)

    print("[INFO] Nitro relay server started.")
    print(f"[INFO] Listen: http://{args.host}:{args.port}")
    print(f"[INFO] EnclaveID: {getattr(enclave, '_enclave_id', None)}")
    print(f"[INFO] EnclaveCID: {getattr(enclave, '_enclave_cid', None)}")
    print(f"[INFO] Public key length: {len(pubkey)} bytes")
    print(f"[INFO] Attestation length: {len(report.document)} bytes")
    if args.api_token:
        print("[INFO] API token auth enabled (X-API-Token required).")
    else:
        print("[WARN] API token auth disabled.")

    stop_event = threading.Event()

    def _shutdown_handler(signum: int, _frame: Any) -> None:
        print(f"[INFO] Received signal {signum}, shutting down...")
        stop_event.set()
        server.shutdown()

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    try:
        server.serve_forever()
    finally:
        server.server_close()
        enclave.destroy()
        print("[INFO] Relay server stopped and enclave terminated.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
