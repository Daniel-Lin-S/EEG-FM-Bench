"""Persist and load baseline training-run artifacts.

Each invocation stores the fully resolved trainer configuration in YAML. The
helpers here validate saved data directly and never merge source files.
"""
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import yaml

if TYPE_CHECKING:
    from baseline.abstract.config import AbstractConfig


CONFIG_HASH_LENGTH = 12


def get_config_hash(config: dict[str, Any], multitask: bool) -> str:
    """Return a stable experiment hash for a resolved configuration."""
    identity = json.loads(json.dumps(config, sort_keys=True))
    identity["logging"].pop("run_dir", None)
    if not multitask:
        identity["data"].pop("datasets", None)
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:CONFIG_HASH_LENGTH]


def save_resolved_config(config: dict[str, Any], path: Path) -> None:
    """Write one fully resolved configuration snapshot atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(config, file_obj, sort_keys=False)
    temporary_path.replace(path)


def load_saved_run_config(
    run_dir: str | Path,
    dataset_name: Optional[str] = None,
) -> 'AbstractConfig':
    """Load a saved configuration without merging defaults or source YAML."""
    artifact_dir = Path(run_dir).resolve()
    if dataset_name is not None:
        completion_path = artifact_dir / "datasets" / dataset_name
        completion_path = completion_path / "completion.json"
        if not completion_path.is_file():
            raise FileNotFoundError(
                f"Completion metadata for dataset '{dataset_name}' was not "
                f"found at {completion_path}."
            )
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        execution_id = completion.get("execution_id")
        if not isinstance(execution_id, str) or not execution_id:
            raise ValueError(
                f"Completion metadata at {completion_path} has no execution ID."
            )
        config_path = artifact_dir / "configs" / f"{execution_id}.yaml"
    else:
        config_paths = sorted((artifact_dir / "configs").glob("*.yaml"))
        if not config_paths:
            raise FileNotFoundError(
                f"No saved resolved configuration exists under {artifact_dir}."
            )
        config_path = config_paths[-1]

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Saved resolved configuration was not found at {config_path}."
        )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Saved configuration at {config_path} is not a mapping.")
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise ValueError(f"Saved configuration at {config_path} has no model_type.")
    from baseline.abstract.factory import ModelRegistry

    config_class = ModelRegistry.get_config_class(model_type)
    return config_class.model_validate(config)


def load_final_checkpoint(run_dir: str | Path, dataset_name: str) -> Path:
    """Return a dataset final checkpoint recorded by a completed run."""
    artifact_dir = Path(run_dir).resolve()
    completion_path = artifact_dir / "datasets" / dataset_name / "completion.json"
    if not completion_path.is_file():
        raise FileNotFoundError(
            f"Completion metadata for dataset '{dataset_name}' was not found "
            f"at {completion_path}."
        )
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("has_checkpoint") is False:
        model_type = completion.get("model_type", "this baseline")
        raise FileNotFoundError(
            f"{model_type} completed {dataset_name} without a checkpoint; "
            "this feature-extractor baseline does not support checkpoint "
            "loading."
        )
    checkpoint_path = completion.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError(
            f"Completion metadata at {completion_path} has no checkpoint path."
        )
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Final checkpoint recorded at {completion_path} does not exist: "
            f"{checkpoint}."
        )
    return checkpoint
