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
# Always create both run artifacts, including when the process emits no stderr.
: > "${LOG_FILE}"
: > "${ERR_FILE}"

# Keep output visible in the terminal while also writing separate stdout and
# stderr files, matching the Slurm wrapper's logging convention.
exec > >(tee >(write_log_without_progress "${LOG_FILE}")) 2> >(tee >(write_log_without_progress "${ERR_FILE}") >&2)

cd "${ROOT_PATH}"

echo "======= LOCAL ENVIRONMENT ======="
echo "Process ID        : $$"
echo "Current working path: ${ROOT_PATH}"
printf 'Script arguments  :'
printf ' %q' "$@"
printf '\n'
echo "Python command    : ${PYTHON}"

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

# Normal runs end with the dataset-level summary emitted by preproc.py. Only
# append a shell fallback when Python could not report its own final status.
case "${EXIT_CODE}" in
  126|127|130|137|143) report_exit_status "${EXIT_CODE}" "preprocessing" ;;
esac

exit "${EXIT_CODE}"
