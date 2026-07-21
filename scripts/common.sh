#!/usr/bin/env bash

# Mirror a stream to the terminal via ``tee`` while omitting tqdm-style refreshes
# (records containing carriage returns) from its saved log file. Normal messages,
# warnings, and tracebacks do not contain carriage returns and remain intact.
write_log_without_progress() {
  local destination="$1"
  awk -v destination="${destination}" '
    index($0, "\r") == 0 {
      print >> destination
      fflush(destination)
      next
    }
    # A logging record can follow a tqdm redraw without an intervening newline.
    # Preserve that warning/error suffix while dropping only the redraw itself.
    match($0, /[[:digit:]]+:(WARNING|ERROR|CRITICAL)[[:space:]]/) {
      print substr($0, RSTART) >> destination
      fflush(destination)
    }
  '
}

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
