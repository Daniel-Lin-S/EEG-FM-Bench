"""Campaign paths, failure fingerprints, and multi-seed summaries.

Inputs are resolved campaign configurations and seed-level completion JSON
files. Outputs are atomic JSON/CSV summaries under the campaign log root.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import yaml

from baseline.utils.identity import (
    IDENTITY_VERSION,
    DETERMINISTIC_MODEL_TYPES,
    build_campaign_semantic_config,
    build_run_semantic_config,
    get_campaign_identity,
    get_legacy_run_hash,
    semantic_digest,
    short_identity,
)
from baseline.utils.run_artifacts import get_config_hash


logger = logging.getLogger("baseline")
DIAGNOSTIC_NAMESPACES = frozenset(
    {"data", "model", "performance", "training"}
)
SEED_DIRECTORY_PATTERN = re.compile(r"^seed_(\d+)$")
ABSOLUTE_PATH_PATTERN = re.compile(r"(?:/[\w.\-]+){2,}")
VOLATILE_TOKEN_PATTERN = re.compile(
    r"\b(?:seed|rank|pid|process|device|gpu|cuda)"
    r"(?:\s+device)?[\s_:=#-]*\d+\b",
    flags=re.IGNORECASE,
)
VOLATILE_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z])[-+]?"
    r"(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
)


@dataclass(frozen=True)
class CampaignPaths:
    """Resolved log and checkpoint locations for one campaign."""

    log_root: Path
    checkpoint_root: Path
    flat_results: bool = False

    def seed_log_root(self, seed: int) -> Path:
        """Return the ordinary artifact root for one seed.

        Parameters
        ----------
        seed : int
            Effective evaluation seed.

        Returns
        -------
        pathlib.Path
            Seed artifact root matching ``log_root``.
        """
        if self.flat_results:
            return self.log_root
        return self.log_root / "logs" / f"seed_{seed}"

    def seed_checkpoint_root(self, seed: int) -> Path:
        """Return the ordinary checkpoint root for one seed.

        Parameters
        ----------
        seed : int
            Effective evaluation seed.

        Returns
        -------
        pathlib.Path
            Checkpoint root mirroring the ordinary seed log root.
        """
        return self.checkpoint_root / f"seed_{seed}"

    @property
    def summary_root(self) -> Path:
        """Return the campaign summary directory."""
        return self.log_root / "summary"


@dataclass(frozen=True)
class CampaignResolution:
    """Resolved semantic campaign and its artifact aliases.

    Parameters
    ----------
    paths : CampaignPaths
        Selected log and checkpoint roots.
    campaign_identity : str
        Full SHA-256 semantic campaign identity.
    semantic_config : Mapping[str, Any]
        Canonical semantic parameters for ``campaign.yaml``.
    aliases : frozenset[str]
        Historical or display identifiers accepted for legacy artifacts.
    legacy : bool
        Whether an existing legacy campaign manifest was selected.
    """

    paths: CampaignPaths
    campaign_identity: str
    semantic_config: Mapping[str, Any]
    aliases: frozenset[str]
    legacy: bool


@dataclass(frozen=True)
class CampaignSummaryResult:
    """In-memory campaign summary and persisted status payload."""

    status: Mapping[str, Any]
    test_runs: Sequence[Mapping[str, Any]]
    test_summary: Sequence[Mapping[str, Any]]
    written: bool


@dataclass(frozen=True)
class CompletionCompatibility:
    """Compatibility result for one dataset completion.

    Parameters
    ----------
    compatible : bool
        Whether the completion may be reused.
    mode : str
        Compatibility rule that accepted or rejected the completion.
    reason : str
        Human-readable diagnostic for rejection or recovery.
    completion : Mapping[str, Any], optional, default=None
        Parsed completion metadata when valid JSON was available.
    terminal : bool, optional, default=False
        Whether the record is a structurally valid completed result even when
        its semantic configuration is incompatible.
    """
    compatible: bool
    mode: str
    reason: str
    completion: Optional[Mapping[str, Any]] = None
    terminal: bool = False


@dataclass(frozen=True)
class LocatedCompletion:
    """Completion selected from the ordinary seed artifact path.

    Parameters
    ----------
    path : pathlib.Path
        Selected completion path, or the expected legacy path when missing.
    compatibility : CompletionCompatibility
        Compatibility result for the selected dataset-seed pair.
    existing_paths : tuple[pathlib.Path, ...]
        The direct completion path when it exists, otherwise an empty tuple.
    """

    path: Path
    compatibility: CompletionCompatibility
    existing_paths: tuple[Path, ...]

    @property
    def run_root(self) -> Path:
        """Return the ordinary seed artifact root for ``path``."""
        return self.path.parents[2]


def get_campaign_hash(
    config: Mapping[str, Any],
    hpo: Optional[Mapping[str, Any]],
) -> str:
    """Return the full versioned semantic campaign identity."""
    return get_campaign_identity(config, hpo)


def build_campaign_paths(
    run_dir: str,
    model_type: str,
    experiment_name: str,
    campaign_hash: str,
) -> CampaignPaths:
    """Build absolute log and checkpoint campaign roots."""
    root = Path(run_dir).resolve()
    display_hash = campaign_hash
    if len(campaign_hash) == 64:
        display_hash = short_identity(campaign_hash)
    name = f"{experiment_name}-{display_hash}"
    log_root = root / "log" / "baseline" / model_type / name
    checkpoint_root = root / "ckpt" / "baseline" / model_type / name
    return CampaignPaths(
        log_root=log_root,
        checkpoint_root=checkpoint_root,
        flat_results=model_type in DETERMINISTIC_MODEL_TYPES,
    )


def _read_campaign_manifest(path: Path) -> dict[str, Any]:
    """Read one campaign YAML mapping without modifying it.

    Parameters
    ----------
    path : pathlib.Path
        Absolute or relative path to ``campaign.yaml``.

    Returns
    -------
    dict[str, Any]
        Parsed campaign mapping.

    Raises
    ------
    ValueError
        If the file is unreadable, malformed, or not a mapping.
    """
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Campaign manifest at {path.resolve()} is invalid: {exc}."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a mapping at {path.resolve()}, but got "
            f"{type(payload).__name__}."
        )
    return payload


def _manifest_semantic_config(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonicalize a current or legacy campaign manifest.

    Parameters
    ----------
    manifest : Mapping[str, Any]
        Parsed ``campaign.yaml`` payload.

    Returns
    -------
    dict[str, Any]
        Version-two semantic campaign parameters.
    """
    model_config = json.loads(json.dumps(manifest, sort_keys=True))
    hpo = model_config.pop("hpo", {})
    return build_campaign_semantic_config(model_config, hpo)


def _normalize_legacy_seed_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a saved scalar seed without changing its source artifact."""
    normalized = json.loads(json.dumps(config, sort_keys=True))
    if "seed" not in normalized:
        return normalized
    if "seeds" in normalized:
        raise ValueError("saved configuration contains both seed and seeds")
    seed = normalized.pop("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"saved configuration has invalid seed {seed!r}")
    normalized["seeds"] = [seed]
    return normalized

def _load_legacy_flat_config(
    completion_path: Path,
    completion: Mapping[str, Any],
) -> dict[str, Any]:
    """Load one flat deterministic completion resolved configuration."""
    execution_id = completion.get("execution_id")
    if (
        not isinstance(execution_id, str)
        or not execution_id
        or Path(execution_id).name != execution_id
    ):
        raise ValueError("execution_id is missing or invalid")
    config_path = completion_path.parents[2] / "configs"
    config_path = config_path / f"{execution_id}.yaml"
    if not config_path.is_file():
        raise ValueError(
            f"resolved configuration does not exist at {config_path.resolve()}"
        )
    try:
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"resolved configuration is invalid: {exc}") from exc
    if not isinstance(saved, dict):
        raise ValueError("resolved configuration is not a mapping")
    return _normalize_legacy_seed_config(saved)


def _has_terminal_legacy_flat_completion(candidate: Path) -> bool:
    """Return whether a flat root contains a completed deterministic result."""
    for completion_path in candidate.glob("datasets/*/completion.json"):
        try:
            completion = json.loads(
                completion_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(completion, dict):
            continue
        if completion.get("status") == "completed" and isinstance(
            completion.get("test_metrics"),
            dict,
        ):
            return True
    return False


def _legacy_flat_root_matches(
    candidate: Path,
    model_type: str,
    semantic_config: Mapping[str, Any],
) -> bool:
    """Return whether a completed flat root has matching semantics."""
    if model_type not in DETERMINISTIC_MODEL_TYPES:
        return False
    if not _has_terminal_legacy_flat_completion(candidate):
        return False
    for config_path in sorted((candidate / "configs").glob("*.yaml")):
        try:
            saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not isinstance(saved, dict):
                continue
            normalized = _normalize_legacy_seed_config(saved)
            candidate_semantic = build_campaign_semantic_config(
                normalized,
                {},
            )
        except (OSError, ValueError, yaml.YAMLError):
            continue
        if candidate_semantic == semantic_config:
            return True
    return False
def _campaign_aliases(
    log_root: Path,
    experiment_name: str,
    campaign_identity: str,
) -> frozenset[str]:
    """Return accepted current and legacy identifiers for one root.

    Parameters
    ----------
    log_root : pathlib.Path
        Selected campaign log root.
    experiment_name : str
        Requested human-readable experiment label.
    campaign_identity : str
        Full semantic campaign identity.

    Returns
    -------
    frozenset[str]
        Full identity, display prefix, detectable directory suffix, and IDs
        recovered from immutable campaign and completion metadata.
    """
    aliases = {campaign_identity, short_identity(campaign_identity)}
    prefix = f"{experiment_name}-"
    if log_root.name.startswith(prefix):
        suffix = log_root.name[len(prefix):]
        if suffix:
            aliases.add(suffix)
    else:
        final_component = log_root.name.rsplit("-", maxsplit=1)[-1]
        if final_component:
            aliases.add(final_component)
    identity_path = log_root / "identity.json"
    if identity_path.is_file():
        try:
            identity_payload = json.loads(
                identity_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            identity_payload = None
        if isinstance(identity_payload, dict):
            stored_identity = identity_payload.get("campaign_identity")
            if isinstance(stored_identity, str) and stored_identity:
                aliases.add(stored_identity)
    for completion_path in sorted(
        log_root.glob("logs/seed_*/datasets/*/completion.json")
    ):
        try:
            completion = json.loads(
                completion_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(completion, dict):
            continue
        stored_identity = completion.get("campaign_hash")
        if isinstance(stored_identity, str) and stored_identity:
            aliases.add(stored_identity)
    return frozenset(aliases)


def resolve_campaign(
    run_dir: str,
    model_type: str,
    experiment_name: str,
    config: Mapping[str, Any],
    hpo: Optional[Mapping[str, Any]],
) -> CampaignResolution:
    """Resolve a unique semantic campaign root, including legacy roots.

    Parameters
    ----------
    run_dir : str
        Configured artifact root.
    model_type : str
        Registered baseline model identifier.
    experiment_name : str
        Requested display label for a newly created campaign.
    config : Mapping[str, Any]
        Fully resolved model configuration.
    hpo : Mapping[str, Any], optional
        Fully resolved HPO configuration.

    Returns
    -------
    CampaignResolution
        Unique existing semantic match or a safe new campaign location.

    Raises
    ------
    RuntimeError
        If multiple semantic matches exist, an identity file conflicts, or
        the intended new path already contains unrelated artifacts.
    """
    semantic_config = build_campaign_semantic_config(config, hpo)
    campaign_identity = semantic_digest(semantic_config)
    requested_paths = build_campaign_paths(
        run_dir,
        model_type,
        experiment_name,
        campaign_identity,
    )
    requested_manifest_path = requested_paths.log_root / "campaign.yaml"
    requested_root_is_incomplete = (
        model_type in DETERMINISTIC_MODEL_TYPES
        and requested_paths.log_root.is_dir()
        and not requested_manifest_path.is_file()
        and not _has_terminal_legacy_flat_completion(requested_paths.log_root)
    )
    if requested_paths.log_root.exists() and not requested_root_is_incomplete:
        manifest_path = requested_manifest_path
        if not manifest_path.is_file():
            raise RuntimeError(
                "The current campaign path exists without a manifest: "
                f"{requested_paths.log_root.resolve()}."
            )
        manifest = _read_campaign_manifest(manifest_path)
        if _manifest_semantic_config(manifest) != semantic_config:
            raise RuntimeError(
                "The current campaign path has different semantic "
                f"parameters: {requested_paths.log_root.resolve()}."
            )
        identity_path = requested_paths.log_root / "identity.json"
        if identity_path.is_file():
            try:
                identity_payload = json.loads(
                    identity_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Identity metadata at {identity_path.resolve()} is "
                    f"invalid: {exc}."
                ) from exc
            if identity_payload.get("campaign_identity") != campaign_identity:
                raise RuntimeError(
                    "The current campaign semantic parameters match but its "
                    f"identity metadata conflicts at {identity_path.resolve()}."
                )
        paths = requested_paths
        legacy = False
    else:
        model_root = requested_paths.log_root.parent
        matches: list[tuple[Path, bool]] = []
        if model_root.is_dir():
            for candidate in sorted(model_root.iterdir()):
                if not candidate.is_dir():
                    continue
                manifest_path = candidate / "campaign.yaml"
                if not manifest_path.is_file():
                    if _legacy_flat_root_matches(
                        candidate,
                        model_type,
                        semantic_config,
                    ):
                        matches.append((candidate, True))
                    continue
                try:
                    manifest = _read_campaign_manifest(manifest_path)
                    candidate_semantic = _manifest_semantic_config(manifest)
                except ValueError as exc:
                    logger.warning(
                        "Ignoring invalid campaign candidate: %s",
                        exc,
                    )
                    continue
                if candidate_semantic == semantic_config:
                    # This branch is reached only after the requested current
                    # path was absent. Preserve every selected historical
                    # directory and its immutable metadata, even when its
                    # normalized manifest exactly equals current semantics.
                    matches.append((candidate, True))

        if len(matches) > 1:
            candidates = ", ".join(
                str(path.resolve()) for path, _ in matches
            )
            raise RuntimeError(
                "Multiple campaign roots, including legacy roots, have the "
                "same semantic "
                f"parameters: {candidates}. Resolve the ambiguity without "
                "deleting results."
            )
        if matches:
            log_root, legacy = matches[0]
            checkpoint_root = (
                Path(run_dir).resolve()
                / "ckpt"
                / "baseline"
                / model_type
                / log_root.name
            )
            paths = CampaignPaths(
                log_root,
                checkpoint_root,
                flat_results=model_type in DETERMINISTIC_MODEL_TYPES,
            )
        else:
            if requested_root_is_incomplete:
                raise RuntimeError(
                    "The current deterministic campaign path is incomplete "
                    f"and has no reusable fallback: "
                    f"{requested_paths.log_root.resolve()}."
                )
            paths = requested_paths
            legacy = False

    aliases = _campaign_aliases(
        paths.log_root,
        experiment_name,
        campaign_identity,
    )
    return CampaignResolution(
        paths=paths,
        campaign_identity=campaign_identity,
        semantic_config=semantic_config,
        aliases=aliases,
        legacy=legacy,
    )


def normalize_failure_message(message: str) -> str:
    """Remove volatile identifiers from an exception message."""
    normalized = ABSOLUTE_PATH_PATTERN.sub("<path>", message)
    normalized = VOLATILE_TOKEN_PATTERN.sub("<runtime>", normalized)
    normalized = VOLATILE_NUMBER_PATTERN.sub("<number>", normalized)
    return " ".join(normalized.split())


def failure_fingerprint(exc: BaseException) -> str:
    """Return an error type plus normalized-message fingerprint."""
    message = normalize_failure_message(str(exc))
    return f"{type(exc).__module__}.{type(exc).__name__}: {message}"


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace one UTF-8 text artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Atomically write one CSV table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _config_identity_hash(config: Mapping[str, Any]) -> str:
    """Return the resolved identity hash for one configuration."""
    return get_config_hash(config, bool(config.get("multitask")))


def _completion_artifact_validation(
    path: Path,
    completion: Mapping[str, Any],
    campaign_identifiers: frozenset[str],
    seed: int,
) -> tuple[Optional[str], bool]:
    """Return an artifact error and terminal-record classification."""
    if completion.get("status") != "completed":
        return "completion status is not 'completed'", False

    metrics = completion.get("test_metrics")
    if not isinstance(metrics, dict):
        return "test_metrics is not a mapping", False
    numeric_count = 0
    for metric_key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric_count += 1
        if not math.isfinite(float(value)):
            return (
                (
                    f"test metric '{metric_key}' is not finite at "
                    f"{path.resolve()}"
                ),
                False,
            )
    if numeric_count == 0:
        return "test_metrics contains no numeric values", False

    try:
        validate_completion_diagnostics(completion)
    except ValueError as exc:
        return str(exc), False

    if completion.get("campaign_hash") not in campaign_identifiers:
        return (
            "campaign identity is not an accepted semantic or legacy ID",
            True,
        )
    if completion.get("seed") != seed:
        return "seed does not match", True
    return None, True


def validate_completion_diagnostics(
    completion: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return a validated typed diagnostics envelope from a completion."""
    diagnostics = completion.get("diagnostics")
    if diagnostics is None:
        return {}
    if not isinstance(diagnostics, Mapping):
        raise ValueError("completion diagnostics is not a mapping")
    unknown = set(diagnostics) - DIAGNOSTIC_NAMESPACES
    if unknown:
        raise ValueError(
            "completion diagnostics has unknown namespaces: "
            f"{sorted(unknown)}"
        )
    validated: Dict[str, Any] = {}
    for namespace in sorted(diagnostics):
        payload = diagnostics[namespace]
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"completion diagnostics namespace '{namespace}' is not "
                "a mapping"
            )
        if payload:
            validated[namespace] = dict(payload)
    return validated


def summarize_completion_diagnostics(
    completions: Iterable[
        tuple[int, str, Mapping[str, Any]]
    ],
) -> Dict[str, Any]:
    """Aggregate opaque typed diagnostics from completed dataset runs."""
    runs: list[Dict[str, Any]] = []
    for seed, dataset_name, completion in completions:
        details = validate_completion_diagnostics(completion)
        if not details:
            continue
        runs.append({
            "seed": seed,
            "dataset": dataset_name,
            "details": details,
        })
    if not runs:
        return {}
    runs.sort(key=lambda item: (item["seed"], item["dataset"]))
    return {"runs": runs}


def _compatibility_failure(
    reason: str,
    completion: Optional[Mapping[str, Any]] = None,
    terminal: bool = False,
) -> CompletionCompatibility:
    """Return one standardized rejected compatibility result."""
    return CompletionCompatibility(
        compatible=False,
        mode="rejected",
        reason=reason,
        completion=completion,
        terminal=terminal,
    )


def _load_saved_completion_config(
    path: Path,
    completion: Mapping[str, Any],
) -> dict[str, Any]:
    """Load and validate the resolved configuration for one completion."""
    execution_id = completion.get("execution_id")
    if (
        not isinstance(execution_id, str)
        or not execution_id
        or Path(execution_id).name != execution_id
    ):
        raise ValueError("execution_id is missing or invalid")
    config_path = path.parents[2] / "configs" / f"{execution_id}.yaml"
    if not config_path.is_file():
        raise ValueError(
            f"resolved configuration does not exist at {config_path}"
        )
    try:
        saved_config = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(
            f"resolved configuration is invalid: {exc}"
        ) from exc
    if not isinstance(saved_config, dict):
        raise ValueError("resolved configuration is not a mapping")
    try:
        multitask = bool(saved_config.get("multitask"))
        saved_hashes = {
            _config_identity_hash(saved_config),
            get_legacy_run_hash(saved_config, multitask),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"resolved configuration cannot be hashed: {exc}"
        ) from exc
    if completion.get("config_hash") not in saved_hashes:
        raise ValueError(
            "stored config hash does not match the resolved configuration"
        )
    return saved_config


def _legacy_runtime_batch_compatibility(
    path: Path,
    completion: Mapping[str, Any],
    expected_config: Mapping[str, Any],
) -> CompletionCompatibility:
    """Validate a historical config whose batch was runtime-mutated."""
    try:
        saved_config = _load_saved_completion_config(path, completion)
        multitask = bool(saved_config.get("multitask"))
    except ValueError as exc:
        return _compatibility_failure(
            str(exc),
            completion,
        )

    saved_semantic = build_run_semantic_config(saved_config, multitask)
    expected_semantic = build_run_semantic_config(
        expected_config,
        bool(expected_config.get("multitask")),
    )
    if saved_semantic == expected_semantic:
        return CompletionCompatibility(
            compatible=True,
            mode="legacy_semantic_compatible",
            reason="accepted a semantically identical legacy completion",
            completion=completion,
        )

    adaptive = saved_config.get("training", {}).get(
        "adaptive_batching",
        {},
    )
    if not isinstance(adaptive, dict) or not adaptive.get("enabled"):
        return _compatibility_failure(
            "resolved configuration did not enable adaptive batching",
            completion,
        )
    saved_data = saved_config.get("data")
    expected_data = expected_config.get("data")
    if not isinstance(saved_data, dict) or not isinstance(
        expected_data,
        dict,
    ):
        return _compatibility_failure(
            "data configuration is missing or invalid",
            completion,
        )
    stored_batch = saved_data.get("batch_size")
    requested_batch = expected_data.get("batch_size")
    if (
        isinstance(stored_batch, bool)
        or not isinstance(stored_batch, int)
        or stored_batch <= 0
        or isinstance(requested_batch, bool)
        or not isinstance(requested_batch, int)
        or requested_batch <= 0
        or requested_batch % stored_batch != 0
    ):
        return _compatibility_failure(
            "stored batch is not a positive divisor of the requested batch",
            completion,
        )

    saved_semantic["data"]["batch_size"] = requested_batch
    if saved_semantic != expected_semantic:
        return _compatibility_failure(
            "resolved configuration differs beyond the runtime batch",
            completion,
        )
    return CompletionCompatibility(
        compatible=True,
        mode="legacy_runtime_batch_compatible",
        reason=(
            "accepted historical completion with runtime-mutated batch"
        ),
        completion=completion,
    )


def check_completion_compatibility(
    path: Path,
    campaign_hash: str,
    seed: int,
    expected_config: Mapping[str, Any] | str,
    campaign_aliases: Iterable[str] = (),
) -> CompletionCompatibility:
    """Validate one completion against its selected semantic config."""
    if not path.is_file():
        return CompletionCompatibility(
            compatible=False,
            mode="missing",
            reason="completion.json does not exist",
        )
    try:
        completion = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _compatibility_failure(
            f"completion metadata is invalid: {exc}"
        )
    if not isinstance(completion, dict):
        return _compatibility_failure(
            "completion metadata is not a mapping"
        )

    campaign_identifiers = frozenset({
        campaign_hash,
        *campaign_aliases,
    })
    artifact_error, terminal = _completion_artifact_validation(
        path,
        completion,
        campaign_identifiers,
        seed,
    )
    if artifact_error is not None:
        return _compatibility_failure(
            artifact_error,
            completion,
            terminal=terminal,
        )

    if isinstance(expected_config, str):
        expected_hash = expected_config
    else:
        try:
            expected_hash = _config_identity_hash(expected_config)
        except (KeyError, TypeError, ValueError) as exc:
            return _compatibility_failure(
                f"selected configuration cannot be hashed: {exc}",
                completion,
            )
        expected_datasets = expected_config.get("data", {}).get(
            "datasets",
            {},
        )
        expected_dataset_config = expected_datasets.get(path.parent.name)
        if completion.get("dataset_config") != expected_dataset_config:
            return _compatibility_failure(
                "dataset configuration does not match",
                completion,
                terminal=True,
            )

    canonical_campaign = completion.get("campaign_hash") == campaign_hash
    if (
        completion.get("config_hash") == expected_hash
        and canonical_campaign
    ):
        return CompletionCompatibility(
            compatible=True,
            mode="exact_canonical_hash",
            reason="canonical configuration hash matches",
            completion=completion,
            terminal=True,
        )
    if isinstance(expected_config, str):
        if not canonical_campaign:
            return _compatibility_failure(
                "a legacy campaign alias requires a selected configuration "
                "for full semantic validation",
                completion,
                terminal=True,
            )
        return _compatibility_failure(
            "configuration hash does not match",
            completion,
            terminal=True,
        )
    legacy = _legacy_runtime_batch_compatibility(
        path,
        completion,
        expected_config,
    )
    return CompletionCompatibility(
        compatible=legacy.compatible,
        mode=legacy.mode,
        reason=legacy.reason,
        completion=completion,
        terminal=True,
    )


def _legacy_flat_completion_compatibility(
    path: Path,
    seed: int,
    expected_config: Mapping[str, Any] | str,
) -> CompletionCompatibility:
    """Validate one historical flat deterministic completion for reuse."""
    if not path.is_file():
        return CompletionCompatibility(
            compatible=False,
            mode="missing",
            reason="completion.json does not exist",
        )
    if isinstance(expected_config, str):
        return _compatibility_failure(
            "legacy flat reuse requires the selected configuration"
        )
    try:
        completion = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _compatibility_failure(
            f"completion metadata is invalid: {exc}"
        )
    if not isinstance(completion, dict):
        return _compatibility_failure("completion metadata is not a mapping")
    if completion.get("status") != "completed":
        return _compatibility_failure(
            "completion status is not 'completed'",
            completion,
        )
    metrics = completion.get("test_metrics")
    if not isinstance(metrics, dict) or not metrics:
        return _compatibility_failure(
            "test_metrics is missing or empty",
            completion,
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in metrics.values()
    ):
        return _compatibility_failure(
            "test_metrics contains a non-finite or non-numeric value",
            completion,
        )
    try:
        saved_config = _load_legacy_flat_config(path, completion)
        saved_semantic = build_run_semantic_config(saved_config, False)
        expected_semantic = build_run_semantic_config(
            expected_config,
            bool(expected_config.get("multitask")),
        )
    except ValueError as exc:
        return _compatibility_failure(str(exc), completion)
    if saved_config.get("seeds") != [seed]:
        return _compatibility_failure(
            "resolved configuration seed does not match",
            completion,
            terminal=True,
        )
    expected_datasets = expected_config.get("data", {}).get("datasets", {})
    if completion.get("dataset_config") != expected_datasets.get(
        path.parent.name
    ):
        return _compatibility_failure(
            "dataset configuration does not match",
            completion,
            terminal=True,
        )
    if saved_config.get("model_type") != expected_config.get("model_type"):
        return _compatibility_failure(
            "model type does not match",
            completion,
            terminal=True,
        )
    if saved_semantic != expected_semantic:
        return _compatibility_failure(
            "resolved configuration differs semantically",
            completion,
            terminal=True,
        )
    return CompletionCompatibility(
        compatible=True,
        mode="legacy_flat_semantic_compatible",
        reason="accepted a semantically identical flat completion",
        completion=completion,
        terminal=True,
    )
def locate_completion(
    campaign_root: Path,
    campaign_hash: str,
    seed: int,
    dataset_name: str,
    expected_config: Mapping[str, Any] | str,
    campaign_aliases: Iterable[str] = (),
) -> LocatedCompletion:
    """Locate one completion at the ordinary seed artifact path.

    Parameters
    ----------
    campaign_root : pathlib.Path
        Campaign log root.
    campaign_hash : str
        Full semantic campaign identity.
    seed : int
        Effective evaluation seed.
    dataset_name : str
        Dataset key in the selected configuration.
    expected_config : Mapping[str, Any] or str
        Selected configuration or its exact final identity.
    campaign_aliases : Iterable[str], optional, default=()
        Historical campaign identifiers accepted for recovery.

    Returns
    -------
    LocatedCompletion
        Direct-path completion and its compatibility result.
    """
    path = (
        campaign_root
        / "logs"
        / f"seed_{seed}"
        / "datasets"
        / dataset_name
        / "completion.json"
    )
    result = check_completion_compatibility(
        path,
        campaign_hash,
        seed,
        expected_config,
        campaign_aliases=campaign_aliases,
    )
    if path.is_file():
        return LocatedCompletion(path, result, (path,))
    if (
        isinstance(expected_config, str)
        or expected_config.get("model_type") not in DETERMINISTIC_MODEL_TYPES
    ):
        return LocatedCompletion(path, result, ())
    flat_path = campaign_root / "datasets" / dataset_name / "completion.json"
    flat_result = _legacy_flat_completion_compatibility(
        flat_path,
        seed,
        expected_config,
    )
    existing = (flat_path,) if flat_path.is_file() else ()
    return LocatedCompletion(flat_path, flat_result, existing)


def _append_metric_rows(
    rows: list[Dict[str, Any]],
    completion: Mapping[str, Any],
    dataset_name: str,
    seed: int,
) -> None:
    """Append numeric metrics from one validated completion."""
    metrics = completion["test_metrics"]
    for metric_key, value in sorted(metrics.items()):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        rows.append({
            "dataset": dataset_name,
            "seed": seed,
            "metric": metric_key.rsplit("/", maxsplit=1)[-1],
            "value": float(value),
        })


def collect_test_rows_with_diagnostics(
    campaign_root: Path,
    campaign_hash: str,
    compatible_configs: Mapping[
        tuple[int, str],
        Mapping[str, Any] | str,
    ],
    campaign_aliases: Iterable[str] = (),
) -> tuple[list[Dict[str, Any]], Dict[str, list[Dict[str, Any]]]]:
    """Collect compatible dataset-seed pairs and their diagnostics."""
    rows: list[Dict[str, Any]] = []
    diagnostics: Dict[str, list[Dict[str, Any]]] = {
        "accepted": [],
        "missing": [],
        "rejected": [],
    }
    for (seed, dataset_name), expected in sorted(
        compatible_configs.items()
    ):
        located = locate_completion(
            campaign_root,
            campaign_hash,
            seed,
            dataset_name,
            expected,
            campaign_aliases=campaign_aliases,
        )
        result = located.compatibility
        diagnostic = {
            "seed": seed,
            "dataset": dataset_name,
            "path": str(located.path.resolve()),
            "reason": result.reason,
        }
        if not result.compatible:
            diagnostics[result.mode].append(diagnostic)
            continue
        diagnostic["mode"] = result.mode
        diagnostics["accepted"].append(diagnostic)
        if result.completion is None:
            raise RuntimeError(
                "Compatible completion result has no parsed metadata."
            )
        _append_metric_rows(
            rows,
            result.completion,
            dataset_name,
            seed,
        )
    return rows, diagnostics


def collect_test_rows(
    campaign_root: Path,
    campaign_hash: str,
    compatible_config_hashes: Optional[
        Mapping[tuple[int, str], Mapping[str, Any] | str]
    ] = None,
    campaign_aliases: Iterable[str] = (),
) -> list[Dict[str, Any]]:
    """Collect compatible numeric test metrics from completed seed runs."""
    if compatible_config_hashes is None:
        compatible_config_hashes = {}
        logs_root = campaign_root / "logs"
        for completion_path in sorted(
            logs_root.glob("seed_*/datasets/*/completion.json")
        ):
            seed_parts = [
                part for part in completion_path.parts
                if SEED_DIRECTORY_PATTERN.match(part)
            ]
            if len(seed_parts) != 1:
                continue
            try:
                completion = json.loads(
                    completion_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                continue
            config_hash = completion.get("config_hash")
            if not isinstance(config_hash, str):
                continue
            match = SEED_DIRECTORY_PATTERN.match(seed_parts[0])
            if match is None:
                continue
            key = (
                int(match.group(1)),
                completion_path.parent.name,
            )
            previous = compatible_config_hashes.get(key)
            if previous is not None and previous != config_hash:
                raise ValueError(
                    "Multiple final configuration identities exist for "
                    f"seed {key[0]} and dataset '{key[1]}'. Provide the "
                    "expected configuration explicitly."
                )
            compatible_config_hashes[key] = config_hash
    rows, _ = collect_test_rows_with_diagnostics(
        campaign_root,
        campaign_hash,
        compatible_config_hashes,
        campaign_aliases=campaign_aliases,
    )
    return rows


def _validate_artifact_completion_config(
    path: Path,
    completion: Mapping[str, Any],
    seed: int,
) -> None:
    """Require saved configuration metadata to match one completion.

    Parameters
    ----------
    path : pathlib.Path
        Direct dataset completion metadata path.
    completion : Mapping[str, Any]
        Parsed terminal completion metadata.
    seed : int
        Seed encoded by the completion's direct artifact directory.

    Raises
    ------
    ValueError
        If the resolved configuration cannot prove the completion provenance.
    """
    saved_config = _load_saved_completion_config(path, completion)
    saved_seeds = saved_config.get("seeds")
    if saved_seeds != [seed]:
        raise ValueError(
            "resolved configuration seed does not match the artifact path"
        )
    data_config = saved_config.get("data")
    if not isinstance(data_config, dict):
        raise ValueError("resolved configuration data is not a mapping")
    datasets_config = data_config.get("datasets")
    if not isinstance(datasets_config, dict):
        raise ValueError(
            "resolved configuration datasets are not a mapping"
        )
    dataset_name = path.parent.name
    if completion.get("dataset_config") != datasets_config.get(dataset_name):
        raise ValueError(
            "dataset configuration does not match the resolved configuration"
        )


def collect_artifact_test_rows_with_diagnostics(
    campaign_root: Path,
    campaign_hash: str,
    campaign_aliases: Iterable[str] = (),
    invocation_id: Optional[str] = None,
) -> tuple[list[Dict[str, Any]], Dict[str, list[Dict[str, Any]]]]:
    """Collect every valid direct terminal result in one campaign artifact.

    Parameters
    ----------
    campaign_root : pathlib.Path
        Selected campaign log root.
    campaign_hash : str
        Canonical campaign identity.
    campaign_aliases : Iterable[str], optional
        Accepted historical campaign identifiers, default=().
    invocation_id : str, optional
        When provided, include only results written by this invocation.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]
        Metric rows and provenance diagnostics. Namespaced and archived
        completions are not discovered.
    """
    rows: list[Dict[str, Any]] = []
    diagnostics: Dict[str, list[Dict[str, Any]]] = {
        "accepted": [],
        "rejected": [],
    }
    campaign_identifiers = frozenset({campaign_hash, *campaign_aliases})
    completion_paths = sorted(
        campaign_root.glob("logs/seed_*/datasets/*/completion.json")
    )
    for completion_path in completion_paths:
        seed_match = SEED_DIRECTORY_PATTERN.match(
            completion_path.parents[2].name
        )
        if seed_match is None:
            continue
        seed = int(seed_match.group(1))
        dataset_name = completion_path.parent.name
        try:
            completion = json.loads(
                completion_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            diagnostics["rejected"].append({
                "seed": seed,
                "dataset": dataset_name,
                "path": str(completion_path.resolve()),
                "reason": f"completion metadata is invalid: {exc}",
            })
            continue
        if not isinstance(completion, dict):
            diagnostics["rejected"].append({
                "seed": seed,
                "dataset": dataset_name,
                "path": str(completion_path.resolve()),
                "reason": "completion metadata is not a mapping",
            })
            continue
        if (
            invocation_id is not None
            and completion.get("invocation_id") != invocation_id
        ):
            continue
        artifact_error, _ = _completion_artifact_validation(
            completion_path,
            completion,
            campaign_identifiers,
            seed,
        )
        if artifact_error is None:
            try:
                _validate_artifact_completion_config(
                    completion_path,
                    completion,
                    seed,
                )
            except ValueError as exc:
                artifact_error = str(exc)
        diagnostic = {
            "seed": seed,
            "dataset": dataset_name,
            "path": str(completion_path.resolve()),
            "reason": artifact_error or "artifact completion is valid",
        }
        if artifact_error is not None:
            diagnostics["rejected"].append(diagnostic)
            continue
        diagnostic["mode"] = "artifact_self_consistent"
        diagnostics["accepted"].append(diagnostic)
        _append_metric_rows(rows, completion, dataset_name, seed)
    for completion_path in sorted(
        campaign_root.glob("datasets/*/completion.json")
    ):
        dataset_name = completion_path.parent.name
        try:
            completion = json.loads(
                completion_path.read_text(encoding="utf-8")
            )
            if not isinstance(completion, dict):
                raise ValueError("completion metadata is not a mapping")
            if (
                invocation_id is not None
                and completion.get("invocation_id") != invocation_id
            ):
                continue
            saved_config = _load_legacy_flat_config(
                completion_path,
                completion,
            )
            saved_seeds = saved_config.get("seeds")
            if not isinstance(saved_seeds, list) or len(saved_seeds) != 1:
                raise ValueError("resolved configuration has invalid seeds")
            seed = saved_seeds[0]
            result = _legacy_flat_completion_compatibility(
                completion_path,
                seed,
                saved_config,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            diagnostics["rejected"].append({
                "seed": None,
                "dataset": dataset_name,
                "path": str(completion_path.resolve()),
                "reason": str(exc),
            })
            continue
        diagnostic = {
            "seed": seed,
            "dataset": dataset_name,
            "path": str(completion_path.resolve()),
            "reason": result.reason,
        }
        if not result.compatible or result.completion is None:
            diagnostics["rejected"].append(diagnostic)
            continue
        diagnostic["mode"] = result.mode
        diagnostics["accepted"].append(diagnostic)
        _append_metric_rows(rows, result.completion, dataset_name, seed)
    return rows, diagnostics


def summarize_test_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Aggregate individual test rows across seeds."""
    grouped: Dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["metric"]))
        grouped.setdefault(key, []).append(float(row["value"]))

    summary_rows: list[Dict[str, Any]] = []
    for (dataset_name, metric), values in sorted(grouped.items()):
        row: Dict[str, Any] = {
            "dataset": dataset_name,
            "metric": metric,
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
        }
        if len(values) >= 2:
            row["std"] = statistics.stdev(values)
        summary_rows.append(row)
    return summary_rows


def _log_compatibility_problems(
    diagnostics: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    """Emit every missing or rejected pair to the stderr logger.

    Parameters
    ----------
    diagnostics : Mapping[str, Sequence[Mapping[str, Any]]]
        Pair-level compatibility diagnostics grouped by status.
    """
    for status in ("missing", "rejected"):
        for diagnostic in diagnostics.get(status, ()):
            logger.warning(
                "Campaign compatibility %s for seed %s dataset %s at %s: %s",
                status,
                diagnostic["seed"],
                diagnostic["dataset"],
                diagnostic["path"],
                diagnostic["reason"],
            )


def _collect_completion_diagnostics(
    diagnostics: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Collect opaque typed diagnostics from accepted completions."""
    completions: list[tuple[int, str, Mapping[str, Any]]] = []
    for accepted in diagnostics.get("accepted", ()):
        path_value = accepted.get("path")
        if not isinstance(path_value, str):
            continue
        try:
            completion = json.loads(
                Path(path_value).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        seed = accepted.get("seed")
        dataset_name = accepted.get("dataset")
        if isinstance(seed, bool) or not isinstance(seed, int):
            continue
        if not isinstance(dataset_name, str) or not dataset_name:
            continue
        completions.append((seed, dataset_name, completion))
    return summarize_completion_diagnostics(completions)


def _compact_invocation_status(
    invocation: Mapping[str, Any],
) -> dict[str, Any]:
    """Return only seed lifecycle fields for the compact summary.

    Parameters
    ----------
    invocation : Mapping[str, Any]
        Completed invocation lifecycle mapping.

    Returns
    -------
    dict[str, Any]
        Invocation identifier and requested or resulting seed lists.
    """
    fields = (
        "id",
        "requested",
        "attempted",
        "dataset_attempts",
        "dataset_outcomes",
        "hpo_outcomes",
        "hpo_failed",
        "succeeded",
        "failed",
        "skipped",
        "unattempted",
    )
    return {
        field: invocation[field]
        for field in fields
        if field in invocation
    }


def write_campaign_summary(
    paths: CampaignPaths,
    campaign_hash: str,
    invocation: Mapping[str, Any],
    campaign_aliases: Iterable[str] = (),
) -> CampaignSummaryResult:
    """Regenerate artifact-wide metric CSVs and a status-only JSON summary.

    Parameters
    ----------
    paths : CampaignPaths
        Selected campaign artifact roots.
    campaign_hash : str
        Full semantic campaign identity.
    invocation : Mapping[str, Any]
        Current invocation seed lifecycle status.
    campaign_aliases : Iterable[str], optional
        Validated historical campaign identifiers, default=().

    Returns
    -------
    CampaignSummaryResult
        Compact status plus in-memory rows for console and cloud reporting.
        ``written`` is false when no compatible metric row exists.
    """
    test_rows, diagnostics = collect_artifact_test_rows_with_diagnostics(
        paths.log_root,
        campaign_hash,
        campaign_aliases=campaign_aliases,
    )
    _log_compatibility_problems(diagnostics)
    pair_status = {
        "discovered": (
            len(diagnostics["accepted"])
            + len(diagnostics["rejected"])
        ),
        "completed": len(diagnostics["accepted"]),
        "rejected": len(diagnostics["rejected"]),
    }
    partial = pair_status["rejected"] > 0
    status_payload = {
        "campaign_identity_version": IDENTITY_VERSION,
        "campaign_identity": campaign_hash,
        "status": "partial" if partial else "complete",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "latest_invocation": _compact_invocation_status(invocation),
        "dataset_pairs": pair_status,
    }
    completion_diagnostics = _collect_completion_diagnostics(diagnostics)
    if completion_diagnostics:
        status_payload["diagnostics"] = completion_diagnostics
    if not test_rows:
        logger.warning(
            "No compatible completed seed results exist under %s; previous "
            "metric CSVs and summary.json were left unchanged.",
            paths.log_root.resolve(),
        )
        status_payload["status"] = "no_compatible_results"
        return CampaignSummaryResult(
            status=status_payload,
            test_runs=(),
            test_summary=(),
            written=False,
        )

    summary_rows = summarize_test_rows(test_rows)
    summary_fields = ["dataset", "metric", "count", "mean", "median"]
    if any("std" in row for row in summary_rows):
        summary_fields.append("std")
    _write_csv(
        paths.summary_root / "test_runs.csv",
        ["dataset", "seed", "metric", "value"],
        test_rows,
    )
    _write_csv(
        paths.summary_root / "test_summary.csv",
        summary_fields,
        summary_rows,
    )
    _atomic_write_text(
        paths.summary_root / "summary.json",
        json.dumps(status_payload, indent=2, sort_keys=True),
    )
    return CampaignSummaryResult(
        status=status_payload,
        test_runs=test_rows,
        test_summary=summary_rows,
        written=True,
    )


def build_invocation_summary(
    paths: CampaignPaths,
    campaign_hash: str,
    invocation: Mapping[str, Any],
    campaign_aliases: Iterable[str] = (),
) -> CampaignSummaryResult:
    """Build a non-persisted summary of final scopes attempted now.

    Parameters
    ----------
    paths : CampaignPaths
        Selected campaign artifact roots.
    campaign_hash : str
        Full semantic campaign identity.
    invocation : Mapping[str, Any]
        Current invocation lifecycle and final scope attempts.
    campaign_aliases : Iterable[str], optional
        Validated historical campaign identifiers, default=().

    Returns
    -------
    CampaignSummaryResult
        Current-invocation metric rows and attempted-pair status without
        writing any artifact-level summary files.
    """
    invocation_id = invocation.get("id")
    if not isinstance(invocation_id, str) or not invocation_id:
        raise ValueError("Invocation summary requires a non-empty ID.")
    attempts = invocation.get("dataset_attempts", [])
    if not isinstance(attempts, list):
        raise ValueError("Invocation dataset_attempts must be a list.")
    attempted_pairs: set[tuple[int, str]] = set()
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise ValueError("Invocation dataset_attempts contains a non-map.")
        seed = attempt.get("seed")
        dataset_name = attempt.get("dataset")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Invocation dataset attempt has an invalid seed.")
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError(
                "Invocation dataset attempt has an invalid dataset name."
            )
        attempted_pairs.add((seed, dataset_name))
    test_rows, diagnostics = collect_artifact_test_rows_with_diagnostics(
        paths.log_root,
        campaign_hash,
        campaign_aliases=campaign_aliases,
        invocation_id=invocation_id,
    )
    _log_compatibility_problems(diagnostics)
    completed_pairs = {
        (item["seed"], item["dataset"])
        for item in diagnostics["accepted"]
    }
    unexpected_pairs = completed_pairs - attempted_pairs
    if unexpected_pairs:
        raise RuntimeError(
            "Invocation completions were not recorded as final attempts: "
            f"{sorted(unexpected_pairs)}."
        )
    pair_status = {
        "attempted": len(attempted_pairs),
        "completed": len(completed_pairs),
        "incomplete": len(attempted_pairs - completed_pairs),
        "rejected": len(diagnostics["rejected"]),
    }
    summary_rows = summarize_test_rows(test_rows)
    status_payload = {
        "campaign_identity_version": IDENTITY_VERSION,
        "campaign_identity": campaign_hash,
        "status": (
            "complete"
            if (
                bool(invocation.get("complete", True))
                and pair_status["incomplete"] == 0
                and pair_status["rejected"] == 0
            )
            else "partial"
        ),
        "latest_invocation": _compact_invocation_status(invocation),
        "dataset_pairs": pair_status,
    }
    completion_diagnostics = _collect_completion_diagnostics(diagnostics)
    if completion_diagnostics:
        status_payload["diagnostics"] = completion_diagnostics
    return CampaignSummaryResult(
        status=status_payload,
        test_runs=test_rows,
        test_summary=summary_rows,
        written=bool(test_rows),
    )
