"""Build semantic identities for baseline campaigns and completed runs.

Inputs are resolved model configurations and optional HPO configurations.
Builders return JSON-compatible semantic mappings, while digest helpers
return full SHA-256 identifiers. Operational invocation fields are excluded.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, MutableMapping, Optional


IDENTITY_VERSION = 5
DISPLAY_ID_LENGTH = 12
HPO_SEARCH_MARKER_KEY = "__hpo_search_parameter__"
DETERMINISTIC_MODEL_TYPES = frozenset({"catch22", "minirocket", "naive"})
INVOCATION_CONFIG_FIELDS = frozenset({
    "conf_file",
    "logging",
    "master_port",
    "seeds",
})
RUNTIME_CONFIG_PATHS = frozenset({
    "data.feature_batch_size",
    "data.load_batch_size",
    "data.memory_limit_gib",
    "data.num_workers",
    "data.pin_memory",
    "data.scratch_dir",
    "model.extractor.n_jobs",
})
MODEL_RUNTIME_CONFIG_PATHS = {
    "catch22": frozenset({
        "data.batch_size",
    }),
    "minirocket": frozenset({
        "data.batch_size",
    }),
    "naive": frozenset({
        "data.batch_size",
    }),
}
# Keep every historical path alias permanently. Future semantic schema moves
# must add an old-to-current entry so immutable campaign manifests still match.
SEMANTIC_CONFIG_PATH_ALIASES = {
    "model.classifier.ridge_alphas": "model.classifier.alphas",
    "model.classifier.ridge_selection_metric": (
        "model.classifier.selection_metric"
    ),
    "model.minirocket_max_dilations_per_kernel": (
        "model.extractor.max_dilations_per_kernel"
    ),
    "model.minirocket_num_features": "model.extractor.num_features",
    "model.minirocket_source_path": "model.extractor.source_path",
    "model.ridge_alphas": "model.classifier.alphas",
    "model.ridge_selection_metric": "model.classifier.selection_metric",
}
INVOCATION_HPO_FIELDS = frozenset({
    "max_consecutive_failed_trials",
    "n_trials",
})


def _json_copy(value: Any) -> Any:
    """Return a detached JSON-compatible copy.

    Parameters
    ----------
    value : Any
        Value from a resolved OmegaConf or Pydantic configuration.

    Returns
    -------
    Any
        A type-preserving copy containing JSON-compatible values.
    """
    return json.loads(json.dumps(value, sort_keys=True))


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize a semantic mapping deterministically.

    Parameters
    ----------
    payload : Mapping[str, Any]
        JSON-compatible semantic identity mapping.

    Returns
    -------
    str
        Compact JSON with sorted keys and preserved scalar types.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def semantic_digest(payload: Mapping[str, Any]) -> str:
    """Return a full SHA-256 digest for a semantic mapping.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Canonical semantic parameters.

    Returns
    -------
    str
        Sixty-four-character lowercase SHA-256 digest.
    """
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def short_identity(identity: str) -> str:
    """Return a non-authoritative identity prefix for labels and paths.

    Parameters
    ----------
    identity : str
        Full SHA-256 semantic identity.

    Returns
    -------
    str
        First twelve hexadecimal characters.

    Raises
    ------
    ValueError
        If ``identity`` is not a lowercase SHA-256 digest.
    """
    if (
        len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
    ):
        raise ValueError(
            "Expected a 64-character lowercase SHA-256 identity, but got "
            f"{identity!r}."
        )
    return identity[:DISPLAY_ID_LENGTH]


def is_runtime_only_config_path(dotted_path: str) -> bool:
    """Return whether a path targets an identity-excluded loader setting.

    Parameters
    ----------
    dotted_path : str
        Dotted resolved-configuration path.

    Returns
    -------
    bool
        Whether the path names a runtime-only data-loader setting.
    """
    runtime_paths = set(RUNTIME_CONFIG_PATHS)
    for model_paths in MODEL_RUNTIME_CONFIG_PATHS.values():
        runtime_paths.update(model_paths)
    return _canonical_config_path(dotted_path) in runtime_paths


def _pop_dotted_path(
    config: MutableMapping[str, Any],
    dotted_path: str,
) -> tuple[bool, Any]:
    """Remove and return one dotted path when it exists."""
    parts = dotted_path.split(".")
    parent: MutableMapping[str, Any] = config
    for part in parts[:-1]:
        child = parent.get(part)
        if not isinstance(child, MutableMapping):
            return False, None
        parent = child
    leaf = parts[-1]
    if leaf not in parent:
        return False, None
    return True, parent.pop(leaf)


def _get_dotted_path(
    config: Mapping[str, Any],
    dotted_path: str,
) -> tuple[bool, Any]:
    """Return one dotted path and whether it exists."""
    parts = dotted_path.split(".")
    parent: Mapping[str, Any] = config
    for part in parts[:-1]:
        child = parent.get(part)
        if not isinstance(child, Mapping):
            return False, None
        parent = child
    leaf = parts[-1]
    if leaf not in parent:
        return False, None
    return True, parent[leaf]


def _set_dotted_path(
    config: MutableMapping[str, Any],
    dotted_path: str,
    value: Any,
) -> None:
    """Set one dotted path, creating missing mapping parents."""
    parts = dotted_path.split(".")
    parent: MutableMapping[str, Any] = config
    for part in parts[:-1]:
        child = parent.get(part)
        if child is None:
            child = {}
            parent[part] = child
        if not isinstance(child, MutableMapping):
            raise ValueError(
                f"Cannot create semantic path {dotted_path!r} because "
                f"{part!r} is not a mapping."
            )
        parent = child
    parent[parts[-1]] = value


def _normalize_semantic_config_paths(
    semantic: MutableMapping[str, Any],
) -> None:
    """Canonicalize all registered historical semantic parameter paths."""
    for legacy_path, canonical_path in SEMANTIC_CONFIG_PATH_ALIASES.items():
        legacy_exists, legacy_value = _get_dotted_path(
            semantic,
            legacy_path,
        )
        if not legacy_exists:
            continue
        canonical_exists, canonical_value = _get_dotted_path(
            semantic,
            canonical_path,
        )
        if canonical_exists and canonical_value != legacy_value:
            raise ValueError(
                f"Conflicting semantic values exist at {legacy_path!r} and "
                f"{canonical_path!r}."
            )
        _pop_dotted_path(semantic, legacy_path)
        if not canonical_exists:
            _set_dotted_path(semantic, canonical_path, legacy_value)


def _canonical_config_path(dotted_path: str) -> str:
    """Return the current canonical name for one configuration path."""
    return SEMANTIC_CONFIG_PATH_ALIASES.get(dotted_path, dotted_path)


def _normalize_search_space_paths(
    search_space: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonicalize historical HPO paths and reject conflicting aliases."""
    normalized: dict[str, Any] = {}
    for dotted_path, distribution in search_space.items():
        canonical_path = _canonical_config_path(dotted_path)
        if (
            canonical_path in normalized
            and normalized[canonical_path] != distribution
        ):
            raise ValueError(
                f"Conflicting HPO distributions target canonical path "
                f"{canonical_path!r}."
            )
        normalized[canonical_path] = distribution
    return normalized


def _remove_runtime_fields(
    semantic: MutableMapping[str, Any],
) -> None:
    """Remove universal result-invariant invocation settings.

    Parameters
    ----------
    semantic : MutableMapping[str, Any]
        Detached resolved configuration to normalize in place.

    Notes
    -----
    Add a field only when it cannot change samples, splits, windows, seeds,
    optimization semantics, or metrics.
    """
    for dotted_path in RUNTIME_CONFIG_PATHS:
        _remove_dotted_path(semantic, dotted_path)


def _remove_dotted_path(
    config: MutableMapping[str, Any],
    dotted_path: str,
) -> None:
    """Remove one optional dotted path from a detached configuration.

    Parameters
    ----------
    config : MutableMapping[str, Any]
        Detached resolved configuration normalized in place.
    dotted_path : str
        Runtime-only path declared by one model implementation.
    """
    _pop_dotted_path(config, dotted_path)


def _remove_model_runtime_fields(
    semantic: MutableMapping[str, Any],
) -> None:
    """Remove declared result-invariant fields for one deterministic model.

    Runtime exclusions are explicit. Model settings remain semantic unless
    their implementation declares them after proving they do not change
    produced metrics.
    """
    model_type = semantic.get("model_type")
    if not isinstance(model_type, str):
        return
    for path in MODEL_RUNTIME_CONFIG_PATHS.get(model_type, ()):
        _remove_dotted_path(semantic, path)


def _base_semantic_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove invocation-only fields from a resolved model configuration.

    Parameters
    ----------
    config : Mapping[str, Any]
        Fully resolved model configuration.

    Returns
    -------
    dict[str, Any]
        Detached semantic configuration without invocation fields.
    """
    semantic = _json_copy(config)
    if not isinstance(semantic, dict):
        raise TypeError("Expected the resolved configuration to be a mapping.")
    for field in INVOCATION_CONFIG_FIELDS:
        semantic.pop(field, None)
    _normalize_semantic_config_paths(semantic)
    _remove_runtime_fields(semantic)
    _remove_model_runtime_fields(semantic)
    return semantic


def _replace_search_value(
    config: MutableMapping[str, Any],
    dotted_path: str,
) -> None:
    """Replace one searched leaf with a canonical marker.

    Parameters
    ----------
    config : MutableMapping[str, Any]
        Detached semantic model configuration.
    dotted_path : str
        Search-space key such as ``training.max_lr``.

    Raises
    ------
    ValueError
        If the path is protected or absent from ``config``.
    """
    parts = dotted_path.split(".")
    if not dotted_path or any(not part for part in parts):
        raise ValueError(f"Invalid search path component: {dotted_path!r}.")
    if is_runtime_only_config_path(dotted_path):
        raise ValueError(
            f"HPO search path '{dotted_path}' targets a runtime-only "
            "data-loader field."
        )
    if parts[0] in INVOCATION_CONFIG_FIELDS or parts[0] == "hpo":
        raise ValueError(
            f"HPO search path '{dotted_path}' targets an operational field."
        )
    parent: MutableMapping[str, Any] = config
    for part in parts[:-1]:
        child = parent.get(part)
        if not isinstance(child, MutableMapping):
            raise ValueError(
                f"HPO search path '{dotted_path}' does not exist in the "
                "resolved model configuration."
            )
        parent = child
    leaf = parts[-1]
    if leaf not in parent:
        raise ValueError(
            f"HPO search path '{dotted_path}' does not exist in the resolved "
            "model configuration."
        )
    parent[leaf] = {HPO_SEARCH_MARKER_KEY: dotted_path}


def build_campaign_semantic_config(
    config: Mapping[str, Any],
    hpo: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the semantic parameters that define one campaign.

    Parameters
    ----------
    config : Mapping[str, Any]
        Fully resolved model configuration.
    hpo : Mapping[str, Any], optional
        Resolved HPO configuration. Invocation budgets are excluded.

    Returns
    -------
    dict[str, Any]
        Campaign parameters with searched leaves replaced by markers.
    """
    semantic = _base_semantic_config(config)
    hpo_semantic: dict[str, Any] = {}
    hpo_enabled = bool(hpo and hpo.get("enabled"))
    if semantic.get("model_type") in DETERMINISTIC_MODEL_TYPES:
        hpo_enabled = False
    if hpo_enabled:
        hpo_semantic = _json_copy(hpo)
        if not isinstance(hpo_semantic, dict):
            raise TypeError("Expected the HPO configuration to be a mapping.")
        for field in INVOCATION_HPO_FIELDS:
            hpo_semantic.pop(field, None)
        search_space = hpo_semantic.get("search_space")
        if not isinstance(search_space, dict) or not search_space:
            raise ValueError(
                "Enabled HPO requires a non-empty search_space mapping."
            )
        search_space = _normalize_search_space_paths(search_space)
        hpo_semantic["search_space"] = search_space
        for dotted_path in sorted(search_space):
            _replace_search_value(semantic, dotted_path)
    if not semantic.get("multitask"):
        data_config = semantic.get("data")
        if not isinstance(data_config, dict):
            raise ValueError(
                "A separate-task campaign requires a data configuration "
                "mapping."
            )
        data_config.pop("datasets", None)
    semantic["hpo"] = hpo_semantic
    return semantic


def build_run_semantic_config(
    config: Mapping[str, Any],
    multitask: bool,
) -> dict[str, Any]:
    """Return semantic parameters for one final seed-scoped execution.

    Parameters
    ----------
    config : Mapping[str, Any]
        Resolved selected configuration containing one effective seed.
    multitask : bool
        Whether the final trainer jointly evaluates multiple datasets.

    Returns
    -------
    dict[str, Any]
        Final-run parameters including effective seed and datasets.

    Raises
    ------
    ValueError
        If the configuration does not contain exactly one integer seed.
    """
    seeds = config.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 1
        or isinstance(seeds[0], bool)
        or not isinstance(seeds[0], int)
    ):
        raise ValueError(
            "A final run identity requires exactly one integer effective seed."
        )
    semantic = _base_semantic_config(config)
    if not multitask:
        data_config = semantic.get("data")
        if not isinstance(data_config, dict):
            raise ValueError(
                "A separate-task run requires a data configuration mapping."
            )
        data_config.pop("datasets", None)
    semantic["effective_seed"] = seeds[0]
    semantic["multitask"] = bool(multitask)
    return semantic


def get_campaign_identity(
    config: Mapping[str, Any],
    hpo: Optional[Mapping[str, Any]],
) -> str:
    """Return the full semantic identity for one campaign.

    Parameters
    ----------
    config : Mapping[str, Any]
        Fully resolved model configuration.
    hpo : Mapping[str, Any], optional
        Fully resolved HPO configuration.

    Returns
    -------
    str
        Full SHA-256 campaign identity.
    """
    return semantic_digest(build_campaign_semantic_config(config, hpo))


def get_run_identity(
    config: Mapping[str, Any],
    multitask: bool,
) -> str:
    """Return the full semantic identity for one final execution.

    Parameters
    ----------
    config : Mapping[str, Any]
        Seed-scoped selected configuration.
    multitask : bool
        Whether the trainer is in multitask mode.

    Returns
    -------
    str
        Full SHA-256 final-run identity.
    """
    return semantic_digest(build_run_semantic_config(config, multitask))


def get_legacy_run_hash(
    config: Mapping[str, Any],
    multitask: bool,
) -> str:
    """Return the historical twelve-character resolved-config hash.

    Parameters
    ----------
    config : Mapping[str, Any]
        Historical resolved trainer configuration.
    multitask : bool
        Historical trainer multitask flag.

    Returns
    -------
    str
        Hash generated by the pre-versioned trainer implementation.
    """
    identity = _json_copy(config)
    logging_config = identity.get("logging")
    if not isinstance(logging_config, dict):
        raise ValueError("Resolved configuration has no logging mapping.")
    logging_config.pop("run_dir", None)
    logging_config.pop("level", None)
    if not multitask:
        data_config = identity.get("data")
        if not isinstance(data_config, dict):
            raise ValueError("Resolved configuration has no data mapping.")
        data_config.pop("datasets", None)
    digest = hashlib.sha256(canonical_json(identity).encode("utf-8"))
    return digest.hexdigest()[:DISPLAY_ID_LENGTH]
