#!/usr/bin/env bash

# Run preprocessing locally.
# Example:
#   scripts/preproc.sh conf_file=preproc/preproc_example.yaml

set -Eeuo pipefail

ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_PATH}/scripts/common.sh"

# ----- user parameters ------
LOG_DIR="${LOG_DIR:-${ROOT_PATH}/logs}"
PYTHON="${PYTHON:-python}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/preproc_${TIMESTAMP}_$$.log"
ERR_FILE="${LOG_DIR}/preproc_${TIMESTAMP}_$$.err"

mkdir -p "${LOG_DIR}"

# Keep output visible in the terminal while also writing separate stdout and
# stderr files, matching the Slurm wrapper's logging convention.
exec > >(tee -a "${LOG_FILE}") 2> >(tee -a "${ERR_FILE}" >&2)

cd "${ROOT_PATH}"

echo "======= LOCAL ENVIRONMENT ======="
echo "Process ID        : $$"
echo "Current working path: ${ROOT_PATH}"
printf 'Script arguments  :'
printf ' %q' "$@"
printf '\n'
echo "Python path: $(which python)"

echo "======= SYSTEM SETUP ======="
echo "PATH: ${PATH}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-}"
echo "Log file          : ${LOG_FILE}"
echo "Error file        : ${ERR_FILE}"
echo "CPUs available    : $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo unknown)"

echo "======= MAIN TASK ======="
if "${PYTHON}" preproc.py "$@"; then
  EXIT_CODE=0
else
  EXIT_CODE=$?
fi

report_exit_status "${EXIT_CODE}" "preprocessing"
exit "${EXIT_CODE}"
