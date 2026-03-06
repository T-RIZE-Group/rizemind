#!/usr/bin/env python3
"""Trigger aggregation on remote relay and decode returned parameters."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from flwr.common.parameter import parameters_to_ndarrays


def _add_repo_to_python_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    src_path = repo_root / "src" / "py"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return repo_root


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


def _parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate pending encrypted updates via remote Nitro relay."
    )
    parser.add_argument(
        "--relay-url",
        default="http://127.0.0.1:8080",
        help="Relay base URL.",
    )
    parser.add_argument(
        "--api-token",
        default="",
        help="Optional X-API-Token for relay auth.",
    )
    parser.add_argument(
        "--round-id",
        type=int,
        default=1,
        help="Round ID to aggregate.",
    )
    parser.add_argument(
        "--min-updates",
        type=int,
        default=2,
        help="Minimum required updates before aggregation.",
    )
    parser.add_argument(
        "--output",
        default=str(repo_root / "results" / "remote_aggregate_result.json"),
        help="Path to write aggregation artifact JSON.",
    )
    return parser.parse_args()


def main() -> int:
    _add_repo_to_python_path()
    from rizemind.tee.params import deserialize_parameters

    args = _parse_args(Path(__file__).resolve().parents[3])
    base = args.relay_url.rstrip("/")

    try:
        response = _http_json(
            method="POST",
            url=f"{base}/aggregate",
            token=args.api_token,
            payload={"round_id": args.round_id, "min_updates": args.min_updates},
        )
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    if not response.get("ok"):
        print(f"[FAIL] Relay /aggregate error: {response}", file=sys.stderr)
        return 1

    params_hex = response.get("aggregated_params_hex", "")
    if not isinstance(params_hex, str) or not params_hex:
        print("[FAIL] Relay returned empty aggregated_params_hex.", file=sys.stderr)
        return 1

    aggregated_bytes = bytes.fromhex(params_hex)
    params = deserialize_parameters(aggregated_bytes)
    ndarrays = parameters_to_ndarrays(params)
    vectors = [arr.tolist() for arr in ndarrays]

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "round_id": args.round_id,
        "used_updates": response.get("used_updates"),
        "aggregated_vectors": vectors,
    }
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("[PASS] Aggregation completed via Nitro relay.")
    print(f"[INFO] Round: {args.round_id}")
    print(f"[INFO] Used updates: {response.get('used_updates')}")
    print(f"[INFO] Aggregated vectors: {vectors}")
    print(f"[PASS] Wrote artifact: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
