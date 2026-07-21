#!/usr/bin/env bash

# Run model training locally.
# Example:
#   scripts/baseline.sh conf_file=assets/conf/baseline/example.yaml
# to set a different device:
# CUDA_DEVICE=4 scripts/baseline.sh ...

set -Eeuo pipefail

ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_PATH}/scripts/common.sh"

# ---- user parameters -----
LOG_DIR="${LOG_DIR:-${ROOT_PATH}/logs}"
PYTHON="${PYTHON:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/baseline_${TIMESTAMP}_$$.log"
ERR_FILE="${LOG_DIR}/baseline_${TIMESTAMP}_$$.err"

mkdir -p "${LOG_DIR}"
# Always create both run artifacts, including when the process emits no stderr.
: > "${LOG_FILE}"
: > "${ERR_FILE}"
exec > >(tee >(write_log_without_progress "${LOG_FILE}")) 2> >(tee >(write_log_without_progress "${ERR_FILE}") >&2)

cd "${ROOT_PATH}"
echo "======= LOCAL ENVIRONMENT ======="
echo "Process ID        : $$"
echo "Current working path: ${ROOT_PATH}"
printf 'Script arguments  :'
printf ' %q' "$@"
printf '\n'
echo "Python path: $(command -v "${PYTHON}" || echo unavailable)"

echo "======= SYSTEM SETUP ======="
echo "PATH: ${PATH}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-}"
echo "CPUs available    : $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo unknown)"
if ! [[ "${CUDA_DEVICE}" =~ ^[0-9]+$ ]]; then
  echo "CUDA_DEVICE must be a non-negative integer; got '${CUDA_DEVICE}'" >&2
  report_exit_status 2 "training"
  exit 2
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  if GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | awk 'NF { count++ } END { print count + 0 }')"; then
    :
  else
    GPU_COUNT=0
  fi
  echo "CUDA status      : nvidia-smi available"
  echo "GPUs available   : ${GPU_COUNT}"
else
  echo "CUDA status      : nvidia-smi unavailable"
  echo "GPUs available   : 0 (CUDA unavailable)"
fi
echo "Selected GPU     : ${CUDA_DEVICE}"
echo "Log file          : ${LOG_FILE}"
echo "Error file        : ${ERR_FILE}"
echo "======= MAIN TASK ======="
if CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" baseline_main.py "$@"; then
  EXIT_CODE=0
else
  EXIT_CODE=$?
fi

report_exit_status "${EXIT_CODE}" "training"
exit "${EXIT_CODE}"
