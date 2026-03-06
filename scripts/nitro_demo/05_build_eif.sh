#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DOCKER_IMAGE="${DOCKER_IMAGE:-rizemind-nitro-demo-enclave}"
EIF_PATH="${EIF_PATH:-${REPO_ROOT}/data/nitro_demo/enclave.eif}"
KEYS_PATH="${KEYS_PATH:-${REPO_ROOT}/data/nitro_demo/keys.json}"
BUILD_LOG_PATH="${BUILD_LOG_PATH:-${REPO_ROOT}/data/nitro_demo/build-enclave-output.txt}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[FAIL] docker not found in PATH" >&2
  exit 1
fi
if ! command -v nitro-cli >/dev/null 2>&1; then
  echo "[FAIL] nitro-cli not found in PATH" >&2
  exit 1
fi
if [[ ! -f "${KEYS_PATH}" ]]; then
  echo "[FAIL] Missing key material: ${KEYS_PATH}" >&2
  echo "[INFO] Run: uv run python scripts/nitro_demo/01_init_keys.py" >&2
  exit 1
fi

mkdir -p "$(dirname "${EIF_PATH}")"
mkdir -p "$(dirname "${BUILD_LOG_PATH}")"

echo "[INFO] Building enclave Docker image: ${DOCKER_IMAGE}"
docker build -t "${DOCKER_IMAGE}" -f "${SCRIPT_DIR}/enclave/Dockerfile" "${REPO_ROOT}"

echo "[INFO] Building EIF: ${EIF_PATH}"
BUILD_OUTPUT="$(nitro-cli build-enclave --docker-uri "${DOCKER_IMAGE}" --output-file "${EIF_PATH}")"
printf "%s\n" "${BUILD_OUTPUT}" | tee "${BUILD_LOG_PATH}" >/dev/null

if [[ ! -f "${EIF_PATH}" ]]; then
  echo "[FAIL] EIF build did not produce expected file: ${EIF_PATH}" >&2
  exit 1
fi

echo "[PASS] EIF built successfully"
echo "[INFO] EIF path: ${EIF_PATH}"
echo "[INFO] Build log: ${BUILD_LOG_PATH}"
