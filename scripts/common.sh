#!/usr/bin/env bash

# Print a consistent, human-readable result for every local wrapper.
report_exit_status() {
  local exit_code="$1"
  local task="$2"
  local exit_status

  if (( exit_code == 0 )); then
    exit_status="SUCCESS: ${task} completed successfully (exit code 0)"
  else
    case "${exit_code}" in
      2) exit_status="FAILED: invalid command-line usage (exit code 2)" ;;
      126) exit_status="FAILED: command could not be executed (exit code 126)" ;;
      127) exit_status="FAILED: Python command or ${task} entrypoint was not found (exit code 127)" ;;
      130) exit_status="INTERRUPTED: process received Ctrl-C (exit code 130)" ;;
      137) exit_status="TERMINATED: process was killed (exit code 137)" ;;
      143) exit_status="TERMINATED: process received SIGTERM (exit code 143)" ;;
      *) exit_status="FAILED: ${task} exited with an error (exit code ${exit_code})" ;;
    esac
  fi

  echo "======= EXIT STATUS: ${exit_status} ======="
}
