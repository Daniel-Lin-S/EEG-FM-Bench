#!/usr/bin/env bash

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RAW_ROOT_DEFAULT="${REPO_ROOT}/assets/data/raw"
RAW_ROOT="${EEGFM_DATABASE_RAW_ROOT:-${RAW_ROOT_DEFAULT}}"
TIMEOUT_SECONDS="${OPENNEURO_TIMEOUT_SECONDS:-0}"
FORCE_REDOWNLOAD=0
TARGET_DATASET="all"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASETS=(
  "adftd|ds004504|1.0.9|https://openneuro.org/datasets/ds004504/versions/1.0.9"
  "chisco|ds005170|1.1.2|https://openneuro.org/datasets/ds005170/versions/1.1.2"
)

usage() {
  cat <<'EOF'
Usage: scripts/download_datasets.sh [OPTIONS]

Download required OpenNeuro datasets into EEG-FM-Bench raw-data layout.

This script REQUIRES DataLad and git-annex. It uses DataLad for all
repository management to ensure metadata consistency.

Options:
  --dataset {all|adftd|chisco}  Choose which dataset to process (default: all)
  --raw-root PATH               Override raw data root path
  --timeout SECONDS             Per-command timeout (0 disables timeout, default: env OPENNEURO_TIMEOUT_SECONDS or 0)
  --force                       Re-download even if dataset is already installed
  -h, --help                    Show this help

Expected output locations:
  ADFTD  -> <raw-root>/ADFTD/data
  Chisco -> <raw-root>/Chisco/data

Examples:
  scripts/download_datasets.sh
  scripts/download_datasets.sh --dataset adftd --timeout 7200
  OPENNEURO_TIMEOUT_SECONDS=10800 scripts/download_datasets.sh --force
EOF
}

log_info() {
  echo "[INFO] $*"
}

log_warn() {
  echo "[WARN] $*" >&2
}

log_error() {
  echo "[ERROR] $*" >&2
}

run_with_timeout() {
  local timeout_sec="$1"
  shift

  if [[ "${timeout_sec}" -gt 0 ]] && command -v timeout >/dev/null 2>&1; then
    timeout --foreground "${timeout_sec}" "$@"
    return $?
  fi

  "$@"
}

ensure_required_tools() {
  local missing=0

  if ! command -v git >/dev/null 2>&1; then
    log_error "Missing required command: git"
    missing=1
  fi

  if ! command -v git-annex >/dev/null 2>&1; then
    log_error "Missing required command: git-annex"
    missing=1
  fi

  if ! command -v datalad >/dev/null 2>&1; then
    log_error "Missing required command: datalad"
    missing=1
  fi

  if [[ "${missing}" -ne 0 ]]; then
    cat >&2 <<'EOF'
Required tools are missing.

Install recommendations:
  Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y git git-annex datalad
  Conda: conda install -c conda-forge git git-annex datalad

Then re-run this script.
EOF
    return 1
  fi

  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    if command -v python >/dev/null 2>&1; then
      PYTHON_BIN="python"
    else
      log_error "Python interpreter is required to resolve dataset layout from config classes."
      log_error "Set EEGFM_PYTHON_BIN or ensure python3/python is in PATH."
      return 1
    fi
  fi

  if [[ "${TIMEOUT_SECONDS}" -gt 0 ]] && ! command -v timeout >/dev/null 2>&1; then
    log_warn "timeout command is not available; timeout protection is disabled."
    log_warn "Install coreutils (Linux) if you want timeout enforcement."
  fi

  return 0
}

resolve_dataset_layout() {
  local dataset_name="$1"

  "${PYTHON_BIN}" - "${REPO_ROOT}" "${dataset_name}" <<'PY'
import ast
import os
import sys


def _is_eeg_config_class(class_node: ast.ClassDef) -> bool:
  for base in class_node.bases:
    if isinstance(base, ast.Name) and base.id == "EEGConfig":
      return True
    if isinstance(base, ast.Attribute) and base.attr == "EEGConfig":
      return True
  return False


def _extract_str_literal(node: ast.AST):
  if isinstance(node, ast.Constant) and isinstance(node.value, str):
    return node.value
  return None


def _extract_class_fields(class_node: ast.ClassDef, wanted):
  values = {}
  for stmt in class_node.body:
    target_name = None
    value_node = None

    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
      target_name = stmt.target.id
      value_node = stmt.value
    elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
      target_name = stmt.targets[0].id
      value_node = stmt.value

    if target_name in wanted and value_node is not None:
      value = _extract_str_literal(value_node)
      if value is not None:
        values[target_name] = value
  return values


def main() -> int:
  repo_root = sys.argv[1]
  dataset_name = sys.argv[2]
  dataset_dir = os.path.join(repo_root, "data", "dataset")
  wanted = {"dataset_name", "suffix_path", "scan_sub_dir", "file_ext"}

  if not os.path.isdir(dataset_dir):
    print(f"Dataset directory not found: {dataset_dir}", file=sys.stderr)
    return 2

  for filename in sorted(os.listdir(dataset_dir)):
    if not filename.endswith(".py") or filename.startswith("__"):
      continue
    file_path = os.path.join(dataset_dir, filename)
    try:
      with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=file_path)
    except SyntaxError:
      continue

    for node in tree.body:
      if not isinstance(node, ast.ClassDef):
        continue
      if not _is_eeg_config_class(node):
        continue

      values = _extract_class_fields(node, wanted)
      if values.get("dataset_name") != dataset_name:
        continue

      required = ["suffix_path", "scan_sub_dir", "file_ext"]
      missing = [key for key in required if key not in values]
      if missing:
        print(
          f"Config class {node.name} for dataset '{dataset_name}' is missing fields: {missing}",
          file=sys.stderr,
        )
        return 3

      print("|".join(values[key] for key in required))
      return 0

  print(f"Could not find an EEGConfig subclass with dataset_name='{dataset_name}'.", file=sys.stderr)
  return 4


if __name__ == "__main__":
  raise SystemExit(main())
PY
}

is_dataset_installed() {
  local dataset_dir="$1"
  local file_ext="$2"

  if [[ ! -d "${dataset_dir}" ]]; then
    return 1
  fi

  if [[ ! -f "${dataset_dir}/participants.tsv" ]]; then
    return 1
  fi

  if ! find "${dataset_dir}" -type f -name "*.${file_ext}" -print -quit | grep -q .; then
    return 1
  fi

  if [[ -d "${dataset_dir}/.git" ]]; then
    local missing_objects
    missing_objects="$(git -C "${dataset_dir}" annex find --not --in=here 2>/dev/null || true)"
    if [[ -n "${missing_objects}" ]]; then
      return 1
    fi
  fi

  return 0
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset)
        if [[ $# -lt 2 ]]; then
          log_error "--dataset requires a value"
          return 1
        fi
        TARGET_DATASET="$2"
        shift 2
        ;;
      --raw-root)
        if [[ $# -lt 2 ]]; then
          log_error "--raw-root requires a value"
          return 1
        fi
        RAW_ROOT="$2"
        shift 2
        ;;
      --timeout)
        if [[ $# -lt 2 ]]; then
          log_error "--timeout requires a value"
          return 1
        fi
        TIMEOUT_SECONDS="$2"
        shift 2
        ;;
      --force)
        FORCE_REDOWNLOAD=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        log_error "Unknown argument: $1"
        usage
        return 1
        ;;
    esac
  done

  case "${TARGET_DATASET}" in
    all|adftd|chisco)
      ;;
    *)
      log_error "Invalid --dataset value: ${TARGET_DATASET}. Use one of: all, adftd, chisco."
      return 1
      ;;
  esac

  if ! [[ "${TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]]; then
    log_error "Timeout must be a non-negative integer, got: ${TIMEOUT_SECONDS}"
    return 1
  fi

  return 0
}

clone_checkout_repo() {
  local target_dir="$1"
  local repo_url="$2"
  local version="$3"

  if [[ -d "${target_dir}/.git" ]]; then
    log_info "Existing DataLad dataset found at ${target_dir}; checking for updates."
    run_with_timeout "${TIMEOUT_SECONDS}" git -C "${target_dir}" fetch --tags --force
  else
    if [[ -e "${target_dir}" ]] && [[ -n "$(ls -A "${target_dir}" 2>/dev/null || true)" ]]; then
      log_error "Target directory exists and is not an empty git repo: ${target_dir}"
      log_error "Move/rename this directory or rerun with --force."
      return 2
    fi

    mkdir -p "$(dirname "${target_dir}")"
    rm -rf "${target_dir}"

    log_info "Installing dataset from ${repo_url} via DataLad into ${target_dir}"
    run_with_timeout "${TIMEOUT_SECONDS}" datalad install -s "${repo_url}" "${target_dir}"
    local install_rc=$?
    if [[ ${install_rc} -ne 0 ]]; then
      return ${install_rc}
    fi
  fi

  log_info "Checking out version ${version}"
  run_with_timeout "${TIMEOUT_SECONDS}" git -C "${target_dir}" checkout -f "${version}"
  return $?
}

download_annex_objects() {
  local target_dir="$1"

  log_info "Enabling OpenNeuro public annex remote (s3-PUBLIC)."
  run_with_timeout "${TIMEOUT_SECONDS}" git -C "${target_dir}" annex enableremote s3-PUBLIC

  log_info "Downloading annexed files with DataLad."
  run_with_timeout "${TIMEOUT_SECONDS}" datalad -C "${target_dir}" get .
  local get_rc=$?
  if [[ ${get_rc} -ne 0 ]]; then
    return ${get_rc}
  fi

  local missing_count
  missing_count="$(git -C "${target_dir}" annex find --not --in=here | wc -l | tr -d ' ')"
  if [[ "${missing_count}" != "0" ]]; then
    log_error "${missing_count} annex objects are still missing in ${target_dir}."
    return 3
  fi

  return 0
}

handle_failure_hint() {
  local rc="$1"
  local dataset_url="$2"

  if [[ "${rc}" -eq 124 ]]; then
    cat >&2 <<EOF
Command timed out while downloading data.

What to do next:
  1) Re-run the same script; git-annex can continue from partial state.
  2) Increase timeout, e.g. --timeout 10800 or OPENNEURO_TIMEOUT_SECONDS=10800.
  3) Check network stability and available disk space.
  4) If this keeps failing, try manual download from:
     ${dataset_url}
EOF
    return
  fi

  cat >&2 <<EOF
Dataset download failed with exit code ${rc}.

What to do next:
  1) Ensure DataLad and git-annex are installed.
  2) Retry the script; DataLad can resume partial downloads.
  3) If failure persists, manually verify dataset accessibility at:
     ${dataset_url}
  4) You can also try manual retrieval inside the dataset folder:
     git annex enableremote s3-PUBLIC
     datalad get .
EOF
}

download_one_dataset() {
  local short_name="$1"
  local accession="$2"
  local version="$3"
  local dataset_url="$4"

  local layout
  layout="$(resolve_dataset_layout "${short_name}")"
  local layout_rc=$?
  if [[ ${layout_rc} -ne 0 ]]; then
    log_error "${short_name}: failed to resolve suffix_path/scan_sub_dir/file_ext from dataset config class."
    return ${layout_rc}
  fi

  local suffix_path
  local scan_sub_dir
  local file_ext
  IFS='|' read -r suffix_path scan_sub_dir file_ext <<<"${layout}"

  local target_dir="${RAW_ROOT}/${suffix_path}/${scan_sub_dir}"
  local repo_url="https://github.com/OpenNeuroDatasets/${accession}.git"

  if [[ "${FORCE_REDOWNLOAD}" -eq 1 ]] && [[ -d "${target_dir}" ]]; then
    log_warn "--force is set; removing existing directory: ${target_dir}"
    rm -rf "${target_dir}"
  fi

  if is_dataset_installed "${target_dir}" "${file_ext}"; then
    log_info "${short_name}: already installed at ${target_dir}. Skipping."
    return 0
  fi

  log_info "${short_name}: installing from ${dataset_url}"

  clone_checkout_repo "${target_dir}" "${repo_url}" "${version}"
  local clone_rc=$?
  if [[ ${clone_rc} -ne 0 ]]; then
    log_error "${short_name}: failed to prepare git repository."
    handle_failure_hint "${clone_rc}" "${dataset_url}"
    return ${clone_rc}
  fi

  download_annex_objects "${target_dir}"
  local annex_rc=$?
  if [[ ${annex_rc} -ne 0 ]]; then
    log_error "${short_name}: failed to fetch annexed content."
    handle_failure_hint "${annex_rc}" "${dataset_url}"
    return ${annex_rc}
  fi

  if is_dataset_installed "${target_dir}" "${file_ext}"; then
    log_info "${short_name}: installation completed at ${target_dir}."
    return 0
  fi

  log_error "${short_name}: post-install validation failed."
  log_error "Expected participants.tsv and at least one .${file_ext} file in ${target_dir}."
  return 4
}

main() {
  parse_args "$@" || exit 1
  ensure_required_tools || exit 1

  mkdir -p "${RAW_ROOT}"
  log_info "Raw dataset root: ${RAW_ROOT}"

  local failed=0
  local failed_names=()

  for line in "${DATASETS[@]}"; do
    IFS='|' read -r short_name accession version dataset_url <<<"${line}"

    if [[ "${TARGET_DATASET}" != "all" ]] && [[ "${TARGET_DATASET}" != "${short_name}" ]]; then
      continue
    fi

    download_one_dataset "${short_name}" "${accession}" "${version}" "${dataset_url}"
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
      failed=1
      failed_names+=("${short_name}")
    fi
  done

  if [[ ${failed} -ne 0 ]]; then
    log_error "Failed datasets: ${failed_names[*]}"
    exit 1
  fi

  log_info "All requested datasets are available in the expected locations."
}

main "$@"
