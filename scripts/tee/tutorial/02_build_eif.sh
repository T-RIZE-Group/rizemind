#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[FAIL] $1" >&2
  exit 1
}

pass() {
  echo "[PASS] $1"
}

info() {
  echo "[INFO] $1"
}

check_cmd() {
  local cmd="$1"
  if command -v "$cmd" >/dev/null 2>&1; then
    pass "Command available: $cmd"
  else
    fail "Command not found: $cmd"
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TEE_DIR="${REPO_ROOT}/src/py/rizemind/tee"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"
DOCKER_URI="${DOCKER_URI:-rizemind-enclave}"
EIF_PATH="${EIF_PATH:-${REPO_ROOT}/enclave.eif}"

if [[ $# -gt 1 ]]; then
  fail "Usage: $0 [eif_path]"
fi

if [[ $# -eq 1 ]]; then
  EIF_PATH="$1"
fi

BUILD_LOG="${RESULTS_DIR}/nitro_build_output.txt"
PCR_FILE="${RESULTS_DIR}/nitro_pcrs.txt"

mkdir -p "$RESULTS_DIR"

check_cmd docker
check_cmd nitro-cli
check_cmd python3

if [[ ! -d "$TEE_DIR" ]]; then
  fail "TEE directory not found: $TEE_DIR"
fi

info "Building Docker image for enclave code..."
(
  cd "$TEE_DIR"
  docker build -t "$DOCKER_URI" -f Dockerfile.enclave .
)
pass "Docker image built: $DOCKER_URI"

info "Running enclave image import smoke test..."
docker run --rm --entrypoint python "$DOCKER_URI" -c \
  "import rizemind.tee.nitro.enclave_server; print('import_ok')"
pass "Enclave image import smoke test passed"

info "Building EIF with nitro-cli..."
BUILD_OUTPUT="$(nitro-cli build-enclave --docker-uri "$DOCKER_URI" --output-file "$EIF_PATH")"
printf "%s\n" "$BUILD_OUTPUT" | tee "$BUILD_LOG" >/dev/null

if [[ ! -f "$EIF_PATH" ]]; then
  fail "EIF file not found after build: $EIF_PATH"
fi

if ! python3 - "$BUILD_LOG" "$PCR_FILE" <<'PY'
import json
import re
import sys
from pathlib import Path

build_log = Path(sys.argv[1])
pcr_file = Path(sys.argv[2])
text = build_log.read_text(encoding="utf-8")

pcr_values: dict[str, str] = {}

try:
    data = json.loads(text)
    measurements = data.get("Measurements", {}) if isinstance(data, dict) else {}
    if isinstance(measurements, dict):
        for key in ("PCR0", "PCR1", "PCR2"):
            value = measurements.get(key)
            if isinstance(value, str) and value:
                pcr_values[key] = value
except json.JSONDecodeError:
    pass

if not pcr_values:
    for key in ("PCR0", "PCR1", "PCR2"):
        match = re.search(rf'{key}"?\s*[:=]\s*"?(0x)?([0-9a-fA-F]+)', text)
        if match:
            pcr_values[key] = match.group(2)

if not pcr_values:
    raise SystemExit("Could not extract PCR values from nitro-cli build output")

lines = [f"{key}={pcr_values[key]}" for key in ("PCR0", "PCR1", "PCR2") if key in pcr_values]
pcr_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
then
  fail "Could not parse PCR values from build output"
fi

pass "EIF build complete: $EIF_PATH"
pass "Build log written: $BUILD_LOG"
pass "PCR values written: $PCR_FILE"
