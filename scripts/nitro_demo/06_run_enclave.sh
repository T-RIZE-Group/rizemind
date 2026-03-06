#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EIF_PATH="${EIF_PATH:-${REPO_ROOT}/data/nitro_demo/enclave.eif}"
CPU_COUNT="${CPU_COUNT:-2}"
MEMORY_MIB="${MEMORY_MIB:-512}"
DEBUG_MODE="${DEBUG_MODE:-true}"

if ! command -v nitro-cli >/dev/null 2>&1; then
  echo "[FAIL] nitro-cli not found in PATH" >&2
  exit 1
fi
if [[ ! -f "${EIF_PATH}" ]]; then
  echo "[FAIL] EIF not found: ${EIF_PATH}" >&2
  echo "[INFO] Run: scripts/nitro_demo/05_build_eif.sh" >&2
  exit 1
fi

CMD=(nitro-cli run-enclave --eif-path "${EIF_PATH}" --cpu-count "${CPU_COUNT}" --memory "${MEMORY_MIB}")
if [[ "${DEBUG_MODE}" == "true" ]]; then
  CMD+=(--debug-mode)
fi

echo "[INFO] Running enclave..."
RUN_OUTPUT="$("${CMD[@]}")"
printf "%s\n" "${RUN_OUTPUT}"

ENCLAVE_ID="$(python -c 'import json,sys; print(json.load(sys.stdin).get("EnclaveID",""))' <<<"${RUN_OUTPUT}")"
ENCLAVE_CID="$(python -c 'import json,sys; print(json.load(sys.stdin).get("EnclaveCID",""))' <<<"${RUN_OUTPUT}")"

if [[ -z "${ENCLAVE_ID}" || -z "${ENCLAVE_CID}" ]]; then
  echo "[FAIL] Could not parse EnclaveID/EnclaveCID from run-enclave output" >&2
  exit 1
fi

echo "[PASS] Enclave started"
echo "[INFO] EnclaveID: ${ENCLAVE_ID}"
echo "[INFO] EnclaveCID: ${ENCLAVE_CID}"
echo "[INFO] Test command:"
echo "  uv run python scripts/nitro_demo/04_parent_client.py --cid ${ENCLAVE_CID} --number 99"
