"""Locate reusable flat artifacts for deterministic feature extractors.

Existing feature-extractor runs predate the campaign directory layout. This
module resolves an old flat root from its saved configuration without changing
the neural or multi-seed campaign artifact logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

from baseline.feature_extractor.config import FeatureExtractorConfig
from baseline.utils.run_artifacts import get_config_hash


def require_single_seed(cfg: FeatureExtractorConfig) -> None:
    """Require the one deterministic evaluation seed supported here.

    Parameters
    ----------
    cfg : FeatureExtractorConfig
        Feature-extractor configuration to validate.

    Raises
    ------
    ValueError
        If the extractor has zero or multiple configured seeds.
    """
    if len(cfg.seeds) != 1:
        raise ValueError(
            f"{cfg.model_type} supports exactly one deterministic seed, but "
            f"got {cfg.seeds}."
        )


def find_matching_artifact_root(
    cfg: FeatureExtractorConfig,
) -> Optional[Path]:
    """Return the unique flat artifact root matching ``cfg``.

    Parameters
    ----------
    cfg : FeatureExtractorConfig
        Requested deterministic feature-extractor configuration.

    Returns
    -------
    pathlib.Path or None
        Existing matching root, or ``None`` when this is a new run.

    Raises
    ------
    RuntimeError
        If more than one existing feature-extractor root matches.
    """
    require_single_seed(cfg)
    requested_config = cfg.model_dump(mode="json")
    requested_hash = get_config_hash(requested_config, multitask=False)
    model_root = Path(
        cfg.logging.run_dir,
        "log",
        "baseline",
        cfg.model_type,
    )
    if not model_root.is_dir():
        return None
    matches = [
        candidate
        for candidate in sorted(model_root.iterdir())
        if candidate.is_dir()
        and _artifact_root_matches(candidate, cfg, requested_hash)
    ]
    if len(matches) > 1:
        paths = ", ".join(str(path.resolve()) for path in matches)
        raise RuntimeError(
            "Multiple feature-extractor artifact roots match the requested "
            f"configuration: {paths}."
        )
    return matches[0] if matches else None


def _artifact_root_matches(
    artifact_root: Path,
    cfg: FeatureExtractorConfig,
    requested_hash: str,
) -> bool:
    """Return whether one root contains a matching saved configuration."""
    config_dir = artifact_root / "configs"
    if not config_dir.is_dir():
        return False
    for config_path in sorted(config_dir.glob("*.yaml")):
        try:
            saved_config = _load_saved_feature_config(config_path, cfg)
        except (OSError, ValueError, yaml.YAMLError):
            continue
        if get_config_hash(saved_config, multitask=False) == requested_hash:
            return True
    return False


def _load_saved_feature_config(
    config_path: Path,
    cfg: FeatureExtractorConfig,
) -> dict[str, Any]:
    """Load one saved config and normalize its historical scalar seed."""
    saved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(saved_config, dict):
        raise ValueError(
            f"Expected config mapping at {config_path.resolve()}, but got "
            f"{type(saved_config).__name__}."
        )
    if "seed" in saved_config:
        if "seeds" in saved_config:
            raise ValueError(
                f"Config at {config_path.resolve()} contains both seed and "
                "seeds."
            )
        seed = saved_config.pop("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(
                f"Expected non-negative integer seed at "
                f"{config_path.resolve()}, but got {seed!r}."
            )
        saved_config["seeds"] = [seed]
    resolved = type(cfg).model_validate(saved_config)
    if resolved.model_type != cfg.model_type:
        raise ValueError(
            f"Expected model type {cfg.model_type!r} at "
            f"{config_path.resolve()}, but got {resolved.model_type!r}."
        )
    if resolved.seeds != cfg.seeds:
        raise ValueError(
            f"Expected saved seeds {cfg.seeds} at {config_path.resolve()}, "
            f"but got {resolved.seeds}."
        )
    return resolved.model_dump(mode="json")


def resolve_feature_extractor_log_path(
    cfg: FeatureExtractorConfig,
) -> tuple[Path, bool]:
    """Return the reusable or new flat log root for ``cfg``.

    Parameters
    ----------
    cfg : FeatureExtractorConfig
        Requested deterministic feature-extractor configuration.

    Returns
    -------
    tuple[pathlib.Path, bool]
        Artifact root and whether it already contained a matching config.
    """
    artifact_root = find_matching_artifact_root(cfg)
    if artifact_root is not None:
        return artifact_root, True
    config_hash = get_config_hash(cfg.model_dump(mode="json"), False)
    experiment_name = f"{cfg.logging.experiment_name}-{config_hash[:12]}"
    log_path = Path(
        cfg.logging.run_dir,
        "log",
        "baseline",
        cfg.model_type,
        experiment_name,
    )
    return log_path, False
