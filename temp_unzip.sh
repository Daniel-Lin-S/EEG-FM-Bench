#!/usr/bin/env bash

set -Eeuo pipefail

# ============================================================
# Recursive ZIP extractor with resume support
#
# Usage:
#   ./extract_all_zips_resume.sh /path/to/root_directory
#
# Example:
#   ./extract_all_zips_resume.sh SEED
#
# Features:
#   - Recursively finds all .zip files
#   - Extracts each zip into its current directory
#   - Creates marker files after successful extraction
#   - Safe to interrupt and resume
#
# ============================================================

ROOT="${1:-.}"

if [[ ! -d "$ROOT" ]]; then
    echo "Error: '$ROOT' is not a directory."
    exit 1
fi


extract_zip() {
    local zipfile="$1"
    local extract_dir
    local marker
    local tmpdir

    extract_dir="$(dirname "$zipfile")"
    marker="${zipfile}.extracted"

    echo "------------------------------------------"
    echo "Processing:"
    echo "$zipfile"

    # Skip completed extraction
    if [[ -f "$marker" ]]; then
        echo "Already extracted. Skipping."
        return
    fi


    # Ignore macOS metadata zips
    if [[ "$(basename "$zipfile")" == ._* ]]; then
        echo "Skipping macOS metadata file."
        return
    fi


    # Validate archive
    if ! unzip -t "$zipfile" >/dev/null 2>&1; then
        echo "ERROR: Invalid zip:"
        echo "$zipfile"
        return 1
    fi


    tmpdir=$(mktemp -d -p "$extract_dir")

    cleanup() {
        rm -rf "$tmpdir"
    }

    trap cleanup EXIT INT TERM

    echo "Temporary extraction:"
    echo "$tmpdir"


    unzip -q "$zipfile" -d "$tmpdir"


    # Count top-level entries
    mapfile -t entries < <(find "$tmpdir" -mindepth 1 -maxdepth 1)

    if [[ ${#entries[@]} -eq 1 && -d "${entries[0]}" ]]; then

        echo "Detected single top-level folder:"
        echo "  $(basename "${entries[0]}")"
        echo "Flattening..."

        mv "${entries[0]}"/* "$extract_dir"/

        # Move hidden files too
        shopt -s dotglob
        mv "${entries[0]}"/* "$extract_dir"/ 2>/dev/null || true
        shopt -u dotglob

    else

        echo "No extra folder detected."
        mv "$tmpdir"/* "$extract_dir"/

    fi


    rm -rf "$tmpdir"

    touch "$marker"

    echo "Finished:"
    echo "$zipfile"
}


find "$ROOT" -type f \
    -iname "*.zip" \
    ! -name "._*" \
    -print0 |
while IFS= read -r -d '' zipfile; do
    extract_zip "$zipfile"
done


echo "All done."