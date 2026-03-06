from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
NITRO_DEMO_DIR = REPO_ROOT / "scripts" / "nitro_demo"
COMMON_PATH = NITRO_DEMO_DIR / "common.py"
INIT_KEYS_SCRIPT = NITRO_DEMO_DIR / "01_init_keys.py"
LOCAL_ROUNDTRIP_SCRIPT = NITRO_DEMO_DIR / "02_local_roundtrip.py"


def _load_common_module():
    spec = importlib.util.spec_from_file_location("nitro_demo_common", COMMON_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {COMMON_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _load_common_module()


def _run_python(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )


def test_key_init_reuses_existing_material(tmp_path: Path) -> None:
    keys_path = tmp_path / "keys.json"
    first = _run_python(INIT_KEYS_SCRIPT, "--keys-path", str(keys_path))
    second = _run_python(INIT_KEYS_SCRIPT, "--keys-path", str(keys_path))

    assert first.returncode == 0
    assert second.returncode == 0

    first_payload = json.loads(first.stdout)
    second_payload = json.loads(second.stdout)

    assert first_payload["status"] == "created"
    assert second_payload["status"] == "reused"
    assert first_payload["sender_public_key_hex"] == second_payload["sender_public_key_hex"]
    assert first_payload["enclave_public_key_hex"] == second_payload["enclave_public_key_hex"]


def test_local_roundtrip_increments_number(tmp_path: Path) -> None:
    keys_path = tmp_path / "keys.json"
    init_proc = _run_python(INIT_KEYS_SCRIPT, "--keys-path", str(keys_path))
    assert init_proc.returncode == 0

    proc = _run_python(
        LOCAL_ROUNDTRIP_SCRIPT,
        "--keys-path",
        str(keys_path),
        "--number",
        "41",
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload == {"ok": True, "input": 41, "result": 42}


def test_tampered_ciphertext_fails_authentication(tmp_path: Path) -> None:
    keys_path = tmp_path / "keys.json"
    init_proc = _run_python(INIT_KEYS_SCRIPT, "--keys-path", str(keys_path))
    assert init_proc.returncode == 0

    proc = _run_python(
        LOCAL_ROUNDTRIP_SCRIPT,
        "--keys-path",
        str(keys_path),
        "--number",
        "41",
        "--tamper",
    )
    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert "error" in payload


def test_non_integer_payload_is_rejected() -> None:
    sender_private = common.generate_private_key_hex()
    enclave_private = common.generate_private_key_hex()
    enclave_public = common.public_key_hex_from_private_key_hex(enclave_private)
    envelope = common.encrypt_bytes(
        json.dumps({"number": "not-an-int"}).encode("utf-8"),
        sender_private,
        enclave_public,
    )
    with pytest.raises(ValueError, match="Payload field 'number' must be an integer"):
        common.decrypt_number(envelope, enclave_private)
