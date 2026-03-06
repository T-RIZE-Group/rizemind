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

check_service_active() {
  local svc="$1"
  if ! command -v systemctl >/dev/null 2>&1; then
    fail "systemctl not found. This tutorial expects a systemd-based Linux host."
  fi

  if systemctl is-active --quiet "$svc"; then
    pass "Service is active: $svc"
  else
    fail "Service is not active: $svc"
  fi
}

info "Running Nitro host readiness checks..."

check_cmd nitro-cli
check_cmd docker
check_cmd python3

if nitro-cli --version >/dev/null 2>&1; then
  pass "nitro-cli responds to --version"
else
  fail "nitro-cli is installed but does not respond correctly"
fi

check_service_active docker
check_service_active nitro-enclaves-allocator.service

CURRENT_GROUPS="$(id -nG)"
for required_group in ne docker; do
  if grep -qw "$required_group" <<<"$CURRENT_GROUPS"; then
    pass "User is in group: $required_group"
  else
    fail "User is not in required group: $required_group"
  fi
done

if ! DESCRIBE_JSON="$(nitro-cli describe-enclaves 2>&1)"; then
  echo "$DESCRIBE_JSON" >&2
  fail "nitro-cli describe-enclaves failed"
fi

if ! ENCLAVE_COUNT="$(
  DESCRIBE_JSON_ENV="$DESCRIBE_JSON" python3 - <<'PY'
import json
import os
import sys

raw = os.environ.get("DESCRIBE_JSON_ENV", "").strip()
if not raw:
    print(0)
    raise SystemExit(0)

try:
    data = json.loads(raw)
except Exception as exc:  # pragma: no cover - runtime diagnostic
    raise SystemExit(f"Could not parse describe-enclaves output: {exc}")

if isinstance(data, list):
    count = len(data)
elif isinstance(data, dict) and isinstance(data.get("Enclaves"), list):
    count = len(data["Enclaves"])
else:
    count = 1

print(count)
PY
)"; then
  fail "Could not determine running enclave count"
fi

if [[ "$ENCLAVE_COUNT" == "0" ]]; then
  pass "No running enclaves found"
else
  echo "[INFO] nitro-cli describe-enclaves output:"
  echo "$DESCRIBE_JSON"
  fail "Expected zero running enclaves, found $ENCLAVE_COUNT"
fi

pass "Nitro host readiness checks completed successfully."
