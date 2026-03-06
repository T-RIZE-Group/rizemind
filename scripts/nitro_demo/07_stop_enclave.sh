#!/usr/bin/env bash
set -euo pipefail

if ! command -v nitro-cli >/dev/null 2>&1; then
  echo "[FAIL] nitro-cli not found in PATH" >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <enclave-id>" >&2
  exit 1
fi

ENCLAVE_ID="$1"

echo "[INFO] Terminating enclave ${ENCLAVE_ID}..."
nitro-cli terminate-enclave --enclave-id "${ENCLAVE_ID}"
echo "[PASS] Enclave terminated"
