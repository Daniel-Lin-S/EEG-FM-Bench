#!/usr/bin/env bash

# Run model training locally.
# Example:
#   scripts/baseline.sh conf_file=assets/conf/baseline/example.yaml
# selects the least-busy GPU automatically.
# CUDA_DEVICE remains a single-GPU compatibility override.

set -Eeuo pipefail

ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_PATH}/scripts/common.sh"

# ---- user parameters -----
LOG_DIR="${LOG_DIR:-${ROOT_PATH}/logs}"
PYTHON="${PYTHON:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-}"
GPU_TELEMETRY_QUERY=(
  --query-gpu=index,memory.free,memory.total,utilization.gpu
  --format=csv,noheader,nounits
)

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

select_vacant_gpu() {
  local telemetry
  local ranked_telemetry
  local gpu_row
  local free_memory_mib
  local utilization_percent
  local gpu_index
  local total_memory_mib
  local -a ranked_rows

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for automatic GPU selection." >&2
    return 1
  fi
  if ! telemetry="$(nvidia-smi "${GPU_TELEMETRY_QUERY[@]}")"; then
    echo "Unable to query GPU memory and utilization with nvidia-smi." >&2
    return 1
  fi
  if [[ -z "${telemetry}" ]]; then
    echo "nvidia-smi reported no GPU devices." >&2
    return 1
  fi
  if ! ranked_telemetry="$(
    printf '%s\n' "${telemetry}" |
      awk -F ',[[:space:]]*' '
        NF != 4 {
          printf "Invalid nvidia-smi telemetry row: %s\n", $0 > "/dev/stderr"
          exit 2
        }
        $1 !~ /^[0-9]+$/ || $2 !~ /^[0-9]+$/ ||
            $3 !~ /^[0-9]+$/ || $4 !~ /^[0-9]+$/ {
          printf "Non-numeric nvidia-smi telemetry row: %s\n", $0 \
            > "/dev/stderr"
          exit 2
        }
        $3 == 0 || $2 > $3 || $4 > 100 {
          printf "Out-of-range nvidia-smi telemetry row: %s\n", $0 \
            > "/dev/stderr"
          exit 2
        }
        { printf "%d,%d,%d,%d\n", $2, $4, $1, $3 }
      ' |
      sort -t ',' -k1,1nr -k2,2n -k3,3n
  )"; then
    return 1
  fi
  if [[ -z "${ranked_telemetry}" ]]; then
    echo "No valid GPU telemetry was available for selection." >&2
    return 1
  fi
  mapfile -t ranked_rows <<< "${ranked_telemetry}"
  gpu_row="${ranked_rows[0]}"
  IFS=',' read -r free_memory_mib utilization_percent gpu_index \
    total_memory_mib <<< "${gpu_row}"
  SELECTED_GPU_DEVICE="${gpu_index}"
  echo "GPU ${gpu_index}: ${free_memory_mib}/${total_memory_mib} MiB free, " \
    "${utilization_percent}% utilized"
}

resolve_gpu_device() {
  if [[ -n "${CUDA_DEVICE}" ]]; then
    if ! [[ "${CUDA_DEVICE}" =~ ^[0-9]+$ ]]; then
      echo "CUDA_DEVICE must be a non-negative integer; got " \
        "'${CUDA_DEVICE}'." >&2
      return 1
    fi
    SELECTED_GPU_DEVICE="${CUDA_DEVICE}"
    return 0
  fi
  select_vacant_gpu
}

echo "======= SYSTEM SETUP ======="
echo "PATH: ${PATH}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-}"
CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo unknown)"
echo "CPUs available    : ${CPU_COUNT}"
if ! resolve_gpu_device; then
  report_exit_status 2 "training"
  exit 2
fi
echo "Selected GPU      : ${SELECTED_GPU_DEVICE}"
echo "Log file          : ${LOG_FILE}"
echo "Error file        : ${ERR_FILE}"
echo "======= MAIN TASK ======="
if EEGFM_ERROR_LOG_PATH="${ERR_FILE}" \
  CUDA_VISIBLE_DEVICES="${SELECTED_GPU_DEVICE}" "${PYTHON}" \
  baseline_main.py "$@"; then
  EXIT_CODE=0
else
  EXIT_CODE=$?
fi

report_exit_status "${EXIT_CODE}" "training"
exit "${EXIT_CODE}"
