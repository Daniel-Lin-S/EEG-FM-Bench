"""Summary-output paths and legacy-layout migration."""

import os
import shutil


def build_summary_path(
        summary_root: str, dataset_name: str | None, suffix_path: str, config_name: str) -> str:
    """Return the dedicated summary directory for a dataset configuration."""
    return os.path.join(summary_root, dataset_name or suffix_path, config_name)


def migrate_legacy_summary_artifacts(
        database_proc_root: str, suffix_path: str, config_name: str, summary_path: str) -> bool:
    """Move a legacy processed-tree summary directory into ``summary_path``.

    Entries in the legacy directory replace same-named destination entries.
    """
    legacy_summary_path = os.path.join(
        database_proc_root,
        suffix_path,
        'summary',
        config_name,
    )
    if legacy_summary_path == summary_path or not os.path.isdir(legacy_summary_path):
        return False

    os.makedirs(summary_path, exist_ok=True)
    for entry in os.listdir(legacy_summary_path):
        source = os.path.join(legacy_summary_path, entry)
        destination = os.path.join(summary_path, entry)
        if os.path.lexists(destination):
            if os.path.isdir(destination) and not os.path.islink(destination):
                shutil.rmtree(destination)
            else:
                os.remove(destination)
        shutil.move(source, destination)

    for directory in (
            legacy_summary_path,
            os.path.dirname(legacy_summary_path),
            os.path.dirname(os.path.dirname(legacy_summary_path)),
    ):
        try:
            os.rmdir(directory)
        except OSError:
            break
    return True
