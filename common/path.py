"""Project paths and the shared, read-only raw-data registry.

``assets/conf/data/data_paths.local.yaml`` is the authoritative source for
project-specific dataset locations. The adjacent ``data_paths.yaml`` file is a
Git-tracked template only. Dataset-specific entries are exact scan roots; when an entry is
absent, the legacy ``assets/data/raw/<suffix_path>/<scan_sub_dir>`` layout is
used instead.
"""

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # Keep path resolution usable before dependencies are installed.
    yaml = None


PLATFORM = os.getenv('EEGFM_PLATFORM', 'local')

PROJECT_ROOT = os.path.abspath(os.getenv('EEGFM_PROJECT_ROOT', os.getcwd()))
ASSETS_ROOT = os.path.join(PROJECT_ROOT, 'assets')
CONF_ROOT = os.getenv('EEGFM_CONF_ROOT', os.path.join(ASSETS_ROOT, 'conf'))
DATA_PATHS_TEMPLATE_FILE = os.path.join(ASSETS_ROOT, 'conf', 'data', 'data_paths.yaml')
DATA_PATHS_FILE = os.path.join(ASSETS_ROOT, 'conf', 'data', 'data_paths.local.yaml')


def _resolve_project_path(path: str) -> str:
    """Resolve a YAML path relative to the project root when necessary."""
    expanded = os.path.expanduser(path)
    if not os.path.isabs(expanded):
        expanded = os.path.join(PROJECT_ROOT, expanded)
    return os.path.abspath(expanded)


def _load_minimal_yaml(path: str) -> dict[str, Any]:
    """Parse the small ``output_root``/``raw_roots`` schema without PyYAML."""
    config: dict[str, Any] = {'raw_roots': {}}
    in_raw_roots = False
    with open(path, 'r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.split('#', 1)[0].rstrip()
            if not line.strip():
                continue
            if line.startswith('output_root:'):
                config['output_root'] = line.split(':', 1)[1].strip().strip('"\'')
                in_raw_roots = False
            elif line.startswith('raw_roots:'):
                raw_roots_value = line.split(':', 1)[1].strip()
                if raw_roots_value not in ('', '{}'):
                    raise ValueError(f'"raw_roots" must be a mapping in {path}')
                in_raw_roots = not raw_roots_value
            elif in_raw_roots and raw_line[:1].isspace() and ':' in line:
                dataset, raw_root = line.strip().split(':', 1)
                config['raw_roots'][dataset.strip()] = raw_root.strip().strip('"\'')
            else:
                raise ValueError(
                    f'PyYAML is not installed and {path} uses unsupported YAML syntax: {raw_line.rstrip()}'
                )
    return config


def _load_data_paths(path: str = DATA_PATHS_FILE) -> dict[str, Any]:
    """Read and validate the fixed shared data-path configuration file."""
    try:
        if yaml is not None:
            with open(path, 'r', encoding='utf-8') as handle:
                config = yaml.safe_load(handle) or {}
        else:
            config = _load_minimal_yaml(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f'Project-specific data-path configuration is required at {path}. '
            f'Copy {DATA_PATHS_TEMPLATE_FILE} to {DATA_PATHS_FILE} and edit it.'
        ) from exc

    if not isinstance(config, dict):
        raise ValueError(f'Shared data-path configuration must be a mapping: {path}')

    output_root = config.get('output_root')
    if not isinstance(output_root, str) or not output_root.strip():
        raise ValueError(f'"output_root" must be a non-empty string in {path}')

    raw_roots = config.get('raw_roots', {})
    if not isinstance(raw_roots, dict) or not all(
            isinstance(dataset, str) and isinstance(raw_root, str) and raw_root.strip()
            for dataset, raw_root in raw_roots.items()):
        raise ValueError(f'"raw_roots" must map dataset IDs to non-empty paths in {path}')

    return {
        'output_root': _resolve_project_path(output_root),
        'raw_roots': {
            dataset: _resolve_project_path(raw_root)
            for dataset, raw_root in raw_roots.items()
        },
    }


DATA_PATHS = _load_data_paths()
OUTPUT_ROOT = DATA_PATHS['output_root']
DATASET_RAW_ROOTS: dict[str, str] = DATA_PATHS['raw_roots']

# Legacy fallback only. Normal data-path selection comes from the ignored local YAML.
DATABASE_RAW_ROOT = os.path.join(ASSETS_ROOT, 'data', 'raw')
DATABASE_PROC_ROOT = os.path.join(OUTPUT_ROOT, 'processed')
DATABASE_CACHE_ROOT = os.path.join(OUTPUT_ROOT, 'cache')
DATABASE_SUMMARY_ROOT = os.path.join(OUTPUT_ROOT, 'summary')
RUN_ROOT = os.path.join(OUTPUT_ROOT, 'run')
LOG_ROOT = os.path.join(OUTPUT_ROOT, 'logs')


def get_dataset_raw_path(dataset_name: str | None, suffix_path: str) -> tuple[str, bool]:
    """Return a dataset's raw path and whether it is a direct YAML override.

    A configured entry is the final directory scanned by a dataset builder, so
    both the legacy suffix and scan-subdirectory layers are intentionally
    skipped. Missing entries preserve the repository's historical layout.
    """
    if dataset_name and dataset_name in DATASET_RAW_ROOTS:
        return DATASET_RAW_ROOTS[dataset_name], True
    return os.path.join(DATABASE_RAW_ROOT, suffix_path), False


def validate_configured_raw_path(dataset_name: str | None, raw_path: str, configured: bool) -> None:
    """Fail before preprocessing/log setup when a configured raw root is invalid."""
    if configured and not Path(raw_path).is_dir():
        raise FileNotFoundError(
            f'Configured raw root for dataset {dataset_name!r} does not exist or is not a directory: {raw_path}'
        )


def get_conf_file_path(path):
    if os.path.isabs(path):
        return path
    elif os.path.exists(path):
        return path
    elif os.path.exists(os.path.normpath(path)):
        return os.path.normpath(path)
    else:
        return os.path.join(CONF_ROOT, os.path.normpath(path))


def create_parent_dir(path):
    par_dir = os.path.dirname(path)
    if not os.path.exists(par_dir):
        os.makedirs(par_dir, exist_ok=True)
