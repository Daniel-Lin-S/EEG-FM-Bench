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
    build_campaign_semantic_config,
    build_run_semantic_config,
    get_campaign_identity,
    get_legacy_run_hash,
    semantic_digest,
    short_identity,
)
from baseline.utils.run_artifacts import get_config_hash


logger = logging.getLogger("baseline")
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
    return CampaignPaths(log_root=log_root, checkpoint_root=checkpoint_root)


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
        recovered from contained completion metadata.
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
    model_root = requested_paths.log_root.parent
    matches: list[tuple[Path, bool]] = []
    if model_root.is_dir():
        for candidate in sorted(model_root.iterdir()):
            manifest_path = candidate / "campaign.yaml"
            if not candidate.is_dir() or not manifest_path.is_file():
                continue
            try:
                manifest = _read_campaign_manifest(manifest_path)
                candidate_semantic = _manifest_semantic_config(manifest)
            except ValueError as exc:
                logger.warning("Ignoring invalid campaign candidate: %s", exc)
                continue
            if candidate_semantic != semantic_config:
                continue
            identity_path = candidate / "identity.json"
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
                stored_identity = identity_payload.get("campaign_identity")
                if stored_identity != campaign_identity:
                    raise RuntimeError(
                        "Campaign semantic parameters match but identity "
                        f"metadata conflicts at {identity_path.resolve()}."
                    )
            matches.append((candidate, manifest != semantic_config))

    if len(matches) > 1:
        candidates = ", ".join(
            str(path.resolve()) for path, _ in matches
        )
        raise RuntimeError(
            "Multiple campaign roots have the same semantic parameters: "
            f"{candidates}. Resolve the ambiguity without deleting results."
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
        paths = CampaignPaths(log_root, checkpoint_root)
    else:
        paths = requested_paths
        legacy = False
        if paths.log_root.exists():
            raise RuntimeError(
                "The intended campaign path already exists without a matching "
                f"semantic manifest: {paths.log_root.resolve()}."
            )
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

    if completion.get("campaign_hash") not in campaign_identifiers:
        return (
            "campaign identity is not an accepted semantic or legacy ID",
            True,
        )
    if completion.get("seed") != seed:
        return "seed does not match", True
    return None, True


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


def _legacy_runtime_batch_compatibility(
    path: Path,
    completion: Mapping[str, Any],
    expected_config: Mapping[str, Any],
) -> CompletionCompatibility:
    """Validate a historical config whose batch was runtime-mutated."""
    execution_id = completion.get("execution_id")
    if (
        not isinstance(execution_id, str)
        or not execution_id
        or Path(execution_id).name != execution_id
    ):
        return _compatibility_failure(
            "execution_id is missing or invalid",
            completion,
        )

    config_path = (
        path.parents[2]
        / "configs"
        / f"{execution_id}.yaml"
    )
    if not config_path.is_file():
        return _compatibility_failure(
            f"resolved configuration does not exist at {config_path}",
            completion,
        )
    try:
        saved_config = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError) as exc:
        return _compatibility_failure(
            f"resolved configuration is invalid: {exc}",
            completion,
        )
    if not isinstance(saved_config, dict):
        return _compatibility_failure(
            "resolved configuration is not a mapping",
            completion,
        )

    try:
        multitask = bool(saved_config.get("multitask"))
        saved_hashes = {
            _config_identity_hash(saved_config),
            get_legacy_run_hash(saved_config, multitask),
        }
    except (KeyError, TypeError, ValueError) as exc:
        return _compatibility_failure(
            f"resolved configuration cannot be hashed: {exc}",
            completion,
        )
    if completion.get("config_hash") not in saved_hashes:
        return _compatibility_failure(
            "stored config hash does not match the resolved configuration",
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
    existing = (path,) if path.is_file() else ()
    return LocatedCompletion(path, result, existing)


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
    compatible_configs: Mapping[
        tuple[int, str],
        Mapping[str, Any] | str,
    ],
    campaign_aliases: Iterable[str] = (),
) -> CampaignSummaryResult:
    """Regenerate metric CSVs and a compact status-only JSON summary.

    Parameters
    ----------
    paths : CampaignPaths
        Selected campaign artifact roots.
    campaign_hash : str
        Full semantic campaign identity.
    invocation : Mapping[str, Any]
        Current invocation seed lifecycle status.
    compatible_configs : Mapping[tuple[int, str], Mapping[str, Any] | str]
        Expected configuration for every seed and dataset pair.
    campaign_aliases : Iterable[str], optional
        Validated historical campaign identifiers, default=().

    Returns
    -------
    CampaignSummaryResult
        Compact status plus in-memory rows for console and cloud reporting.
        ``written`` is false when no compatible metric row exists.
    """
    test_rows, diagnostics = collect_test_rows_with_diagnostics(
        paths.log_root,
        campaign_hash,
        compatible_configs,
        campaign_aliases=campaign_aliases,
    )
    _log_compatibility_problems(diagnostics)
    pair_status = {
        "expected": len(compatible_configs),
        "completed": len(diagnostics["accepted"]),
        "missing": len(diagnostics["missing"]),
        "rejected": len(diagnostics["rejected"]),
    }
    partial = (
        not bool(invocation.get("complete", True))
        or pair_status["missing"] > 0
        or pair_status["rejected"] > 0
    )
    status_payload = {
        "campaign_identity_version": IDENTITY_VERSION,
        "campaign_identity": campaign_hash,
        "status": "partial" if partial else "complete",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "latest_invocation": _compact_invocation_status(invocation),
        "dataset_pairs": pair_status,
    }
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
