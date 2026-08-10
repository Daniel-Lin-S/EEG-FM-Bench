"""Run HPO studies and independent multi-seed baseline campaigns.

Inputs are one validated model configuration and an optional HpoConfig.
Outputs are persistent Optuna studies, current-style seed artifact trees,
completion metadata, and campaign test summaries.
"""

from __future__ import annotations
import csv

import copy
import datetime
import fcntl
import gc
import json
import logging
import math
import os
import shutil
import sqlite3
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, Mapping, Optional, Type

import torch
import yaml
from pydantic import BaseModel

from baseline.adaptive_batching import (
    configure_cuda_allocator,
    derive_batch_candidates,
    is_cuda_oom,
    select_safe_micro_batch,
)
from baseline.abstract.factory import ModelRegistry
from baseline.hpo.artifacts import (
    CampaignPaths,
    CampaignResolution,
    CampaignSummaryResult,
    build_invocation_summary,
    failure_fingerprint,
    locate_completion,
    resolve_campaign,
    write_campaign_summary,
)
from baseline.hpo.config import HpoConfig
from baseline.hpo.search import (
    objective_values,
    reduce_objective,
    sample_config,
    set_dotted_value,
    validate_search_space,
)
from baseline.utils.identity import (
    DETERMINISTIC_MODEL_TYPES,
    IDENTITY_VERSION,
    canonical_json,
    semantic_digest,
    short_identity,
)
from baseline.utils.run_artifacts import get_config_hash
from common.distributed.env import (
    clean_torch_distributed,
    get_is_master,
    get_world_size,
)
from common.log import setup_log


logger = logging.getLogger("baseline")
DETERMINISTIC_MODELS = DETERMINISTIC_MODEL_TYPES
PRUNED_STATUS = "pruned"
COMPLETE_STATUS = "complete"
FAILED_STATUS = "failed"
OOM_MESSAGE_PARTS = (
    "out of memory",
    "cuda error: memory allocation",
)
UNBUDGETED_TRIAL_STATES = frozenset({
    "FAIL",
    "RUNNING",
    "WAITING",
})


@dataclass
class ExecutionLocks:
    """Advisory locks held for one seed's selected scopes.

    Parameters
    ----------
    files : tuple[BinaryIO, ...]
        Open files whose exclusive nonblocking locks are held by rank zero.
    paths : tuple[pathlib.Path, ...]
        Absolute lock paths used for conflict diagnostics.
    """

    files: tuple[BinaryIO, ...]
    paths: tuple[Path, ...]


def _execution_lock_path(
    paths: CampaignPaths,
    seed: int,
    scope: str,
) -> Path:
    """Return a stable execution-lock path for one seed and scope."""
    scope_identity = semantic_digest({"scope": scope})
    return paths.log_root / ".locks" / f"seed_{seed}" / scope_identity


@dataclass(frozen=True)
class ReplacementArchive:
    """Completed artifacts copied before an authorized replacement.

    Parameters
    ----------
    root : pathlib.Path
        Exclusive archive root owned by the current invocation.
    copies : tuple[tuple[pathlib.Path, pathlib.Path], ...]
        Original and archived artifact pairs.
    cleanup_paths : tuple[pathlib.Path, ...]
        Ordinary partial paths removed before restoration.
    """

    root: Path
    copies: tuple[tuple[Path, Path], ...]
    cleanup_paths: tuple[Path, ...]


@dataclass
class StudyRuntime:
    """Mutable handle for one exclusively locked Optuna study.

    Parameters
    ----------
    study : Any
        Loaded Optuna study.
    study_identity : str
        Full semantic study identity.
    storage_path : pathlib.Path
        SQLite storage selected for the study.
    artifact_root : pathlib.Path
        Exclusive namespaced trial and winner root.
    checkpoint_root : pathlib.Path
        Exclusive namespaced trial checkpoint root.
    legacy : bool
        Whether the selected study lives in legacy shared SQLite storage.
    duplicate_names : tuple[str, ...]
        Preserved non-selected study names found in legacy storage.
    lock_file : BinaryIO
        Open file whose advisory lock is held for this runtime.
    removed_trial_numbers : tuple[int, ...], optional
        Trial numbers removed before this runtime resumed, default=().
    """

    study: Any
    study_identity: str
    storage_path: Path
    artifact_root: Path
    checkpoint_root: Path
    legacy: bool
    duplicate_names: tuple[str, ...]
    lock_file: BinaryIO
    removed_trial_numbers: tuple[int, ...] = ()


class CampaignExecutionError(RuntimeError):
    """Raised after one or more seed executions fail."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write one JSON mapping."""
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if existing == dict(payload):
            return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _exclusive_text(path: Path, content: str) -> None:
    """Create one text artifact without replacing an existing path.

    Parameters
    ----------
    path : pathlib.Path
        Destination that must not exist.
    content : str
        Complete UTF-8 file contents.

    Raises
    ------
    FileExistsError
        If another artifact already owns ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as file_obj:
            file_obj.write(content)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create one immutable JSON mapping.

    Parameters
    ----------
    path : pathlib.Path
        Destination that must not exist.
    payload : Mapping[str, Any]
        JSON-compatible mapping to serialize.
    """
    _exclusive_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True),
    )


def _exclusive_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    """Create one immutable YAML mapping.

    Parameters
    ----------
    path : pathlib.Path
        Destination that must not exist.
    payload : Mapping[str, Any]
        YAML-compatible mapping to serialize.
    """
    _exclusive_text(
        path,
        yaml.safe_dump(dict(payload), sort_keys=False),
    )


def _validate_immutable_json(
    path: Path,
    expected: Mapping[str, Any],
) -> None:
    """Require an existing JSON artifact to equal an expected mapping.

    Parameters
    ----------
    path : pathlib.Path
        Existing JSON artifact.
    expected : Mapping[str, Any]
        Exact expected payload.

    Raises
    ------
    RuntimeError
        If the artifact is invalid or differs from ``expected``.
    """
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Immutable JSON artifact at {path.resolve()} is invalid: {exc}."
        ) from exc
    if existing != dict(expected):
        raise RuntimeError(
            f"Refusing to overwrite mismatching immutable artifact at "
            f"{path.resolve()}."
        )


def _broadcast_object(value: Any) -> Any:
    """Broadcast a picklable master value to all distributed ranks."""
    if not (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        return value
    objects = [value if get_is_master() else None]
    torch.distributed.broadcast_object_list(objects, src=0)
    return objects[0]


def _scoped_config(
    config: Mapping[str, Any],
    seed: int,
    datasets_config: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Return one independent seed- and dataset-scoped config."""
    scoped = copy.deepcopy(dict(config))
    scoped["seeds"] = [seed]
    if datasets_config is not None:
        scoped["data"]["datasets"] = dict(datasets_config)
    return scoped


def _force_local_trial_logging(config: Dict[str, Any]) -> None:
    """Disable cloud trial logging and ensure a CSV validation trace."""
    logging_config = config["logging"]
    logging_config["use_cloud"] = False
    outputs = list(logging_config.get("outputs", []))
    if logging_config.get("level", "info") != "debug":
        outputs = [output for output in outputs if output != "log"]
    if "csv" not in outputs:
        outputs.append("csv")
    logging_config["outputs"] = outputs


def _study_scope_configs(
    config: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Return dataset-specific or joint HPO base configs."""
    if config["multitask"]:
        return {"multitask": copy.deepcopy(dict(config))}

    scopes: Dict[str, Dict[str, Any]] = {}
    for dataset_name, dataset_config in config["data"]["datasets"].items():
        scopes[dataset_name] = _scoped_config(
            config,
            seed=0,
            datasets_config={dataset_name: dataset_config},
        )
        scopes[dataset_name]["seeds"] = list(config["seeds"])
    return scopes


def _config_hash(config: Mapping[str, Any]) -> str:
    """Return one scoped trainer configuration hash."""
    return get_config_hash(
        copy.deepcopy(dict(config)),
        bool(config["multitask"]),
    )


def _expected_configs(
    selected_configs: Mapping[str, Mapping[str, Any]],
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    """Return selected dataset configurations for one seed."""
    expected: Dict[str, Dict[str, Any]] = {}
    for config in selected_configs.values():
        scoped = _scoped_config(config, seed)
        config_hash = _config_hash(scoped)
        for dataset_name in scoped["data"]["datasets"]:
            previous = expected.get(dataset_name)
            if previous is not None and _config_hash(previous) != config_hash:
                raise ValueError(
                    f"Dataset '{dataset_name}' has conflicting selected "
                    "configurations."
                )
            expected[dataset_name] = copy.deepcopy(scoped)
    if not expected:
        raise ValueError("Selected campaign configuration has no datasets.")
    return expected


def _seed_scope_is_complete(
    paths: CampaignPaths,
    campaign_hash: str,
    seed: int,
    selected_configs: Mapping[str, Mapping[str, Any]],
    campaign_aliases: tuple[str, ...] = (),
) -> bool:
    """Return whether every selected dataset scope has matching completion."""
    expected = _expected_configs(selected_configs, seed)
    for dataset_name, config in expected.items():
        located = locate_completion(
            paths.log_root,
            campaign_hash,
            seed,
            dataset_name,
            config,
            campaign_aliases=campaign_aliases,
        )
        result = located.compatibility
        if not result.compatible:
            logger.debug(
                "Seed %d dataset %s is incomplete at %s: %s",
                seed,
                dataset_name,
                located.path.resolve(),
                result.reason,
            )
            return False
    return True


def _scope_artifact_roots(
    paths: CampaignPaths,
    campaign_hash: str,
    seed: int,
    scoped_config: Mapping[str, Any],
    campaign_aliases: tuple[str, ...] = (),
    allow_replacement: bool = False,
) -> tuple[Path, Path, bool]:
    """Return ordinary roots after validating existing completions.

    Parameters
    ----------
    paths : CampaignPaths
        Campaign log and checkpoint roots.
    campaign_hash : str
        Full semantic campaign identity.
    seed : int
        Effective evaluation seed.
    scoped_config : Mapping[str, Any]
        Selected final configuration containing one effective seed.
    campaign_aliases : tuple[str, ...], optional, default=()
        Validated historical campaign identifiers.
    allow_replacement : bool, optional, default=False
        Whether proven HPO budget growth authorizes terminal replacement.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, bool]
        Ordinary log root, checkpoint root, and completion state.
    """
    located = [
        locate_completion(
            paths.log_root,
            campaign_hash,
            seed,
            dataset_name,
            scoped_config,
            campaign_aliases=campaign_aliases,
        )
        for dataset_name in scoped_config["data"]["datasets"]
    ]
    compatible = [
        item for item in located if item.compatibility.compatible
    ]
    conflicts = [
        item for item in located
        if item.compatibility.terminal
        and not item.compatibility.compatible
    ]
    if conflicts and not allow_replacement:
        reasons = "; ".join(
            f"{item.path.resolve()}: {item.compatibility.reason}"
            for item in conflicts
        )
        raise RuntimeError(
            "A completed result conflicts with the selected semantic "
            "configuration and cannot be overwritten: " + reasons
        )
    log_root = paths.seed_log_root(seed)
    checkpoint_root = paths.seed_checkpoint_root(seed)
    if len(compatible) == len(located):
        return log_root, checkpoint_root, True
    if compatible:
        completed = ", ".join(
            item.path.parent.name for item in compatible
        )
        raise RuntimeError(
            "A multitask scope has compatible completed datasets and cannot "
            "be retrained. Recover its missing datasets instead: "
            + completed
        )
    return log_root, checkpoint_root, False


def _is_recoverable_trial_failure(exc: Exception) -> bool:
    """Return whether one HPO trial failure may safely be continued."""
    if isinstance(exc, (ValueError, torch.cuda.OutOfMemoryError)):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return any(part in message for part in OOM_MESSAGE_PARTS)


def _release_training_state() -> None:
    """Release cyclic trainer references and cached CUDA allocations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _remove_checkpoint_tree(path: Path, purpose: str) -> Optional[str]:
    """Remove one checkpoint tree and return a failed absolute path.

    Parameters
    ----------
    path : pathlib.Path
        Checkpoint directory owned by the current run or trial.
    purpose : str
        Human-readable cleanup context for diagnostics.

    Returns
    -------
    str or None
        Absolute path when cleanup fails, otherwise ``None``.
    """
    if not path.exists():
        return None
    try:
        shutil.rmtree(path)
    except OSError as exc:
        resolved = str(path.resolve())
        logger.warning(
            "%s checkpoint cleanup failed at %s: %s",
            purpose,
            resolved,
            exc,
        )
        return resolved

    return None


def _remove_hpo_trial_artifact(path: Path, purpose: str) -> None:
    """Remove one exact HPO trial artifact or raise a cleanup error."""
    if not path.exists():
        return
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError as exc:
        raise RuntimeError(
            f"{purpose} cleanup failed at {path.resolve()}: {exc}."
        ) from exc


def _quote_sqlite_identifier(identifier: str) -> str:
    """Quote one SQLite identifier for a schema inspection query."""
    return '"' + identifier.replace('"', '""') + '"'


def _cleanup_unbudgeted_hpo_trials(
    storage_path: Path,
    study_name: str,
    artifact_roots: tuple[Path, ...],
    checkpoint_roots: tuple[Path, ...],
) -> tuple[int, ...]:
    """Remove failed or stale HPO trials before study resumption.

    Optuna does not expose a public trial-removal API. The SQLite storage is
    therefore edited under the campaign study lock, deleting all rows that
    reference each selected trial before deleting its parent row. Completed
    and pruned trials are retained and continue to determine the next trial
    number.

    Parameters
    ----------
    storage_path : pathlib.Path
        SQLite storage for the selected study.
    study_name : str
        Exact Optuna study name in the storage.
    artifact_roots : tuple[pathlib.Path, ...]
        Exact roots containing ``trials/trial_<number>`` directories.
    checkpoint_roots : tuple[pathlib.Path, ...]
        Exact roots containing ``trial_<number>`` checkpoint directories.

    Returns
    -------
    tuple[int, ...]
        Removed Optuna trial numbers in ascending order.

    Raises
    ------
    RuntimeError
        If the SQLite storage or an associated artifact cannot be cleaned.
    """
    if not storage_path.is_file():
        return ()
    connection: Optional[sqlite3.Connection] = None
    try:
        connection = sqlite3.connect(storage_path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        study_row = connection.execute(
            "SELECT study_id FROM studies WHERE study_name = ?",
            (study_name,),
        ).fetchone()
        if study_row is None:
            return ()
        study_id = int(study_row[0])
        trial_rows = connection.execute(
            "SELECT trial_id, number, state FROM trials "
            "WHERE study_id = ? AND state IN (?, ?, ?) "
            "ORDER BY number",
            (study_id, *sorted(UNBUDGETED_TRIAL_STATES)),
        ).fetchall()
        if not trial_rows:
            return ()
        trial_ids = [int(row[0]) for row in trial_rows]
        trial_numbers = tuple(int(row[1]) for row in trial_rows)
        placeholders = ",".join("?" for _ in trial_ids)
        for trial_number in trial_numbers:
            trial_name = f"trial_{trial_number:05d}"
            for root in artifact_roots:
                _remove_hpo_trial_artifact(
                    root / "trials" / trial_name,
                    f"HPO trial {trial_number} artifact",
                )
            for root in checkpoint_roots:
                _remove_hpo_trial_artifact(
                    root / trial_name,
                    f"HPO trial {trial_number} checkpoint",
                )
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        for table_row in table_rows:
            table_name = str(table_row[0])
            if table_name == "trials":
                continue
            columns = connection.execute(
                "PRAGMA table_info("
                f"{_quote_sqlite_identifier(table_name)})"
            ).fetchall()
            if not any(str(column[1]) == "trial_id" for column in columns):
                continue
            quoted_name = _quote_sqlite_identifier(table_name)
            connection.execute(
                f"DELETE FROM {quoted_name} "
                f"WHERE trial_id IN ({placeholders})",
                trial_ids,
            )
        connection.execute(
            f"DELETE FROM trials WHERE trial_id IN ({placeholders})",
            trial_ids,
        )
        connection.commit()
        return trial_numbers
    except (OSError, sqlite3.Error, RuntimeError) as exc:
        if connection is not None:
            connection.rollback()
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(
            f"Cannot clean unbudgeted HPO trials from "
            f"{storage_path.resolve()}: {exc}."
        ) from exc
    finally:
        if connection is not None:
            connection.close()


def _acquire_execution_locks(
    paths: CampaignPaths,
    seed: int,
    scopes: Mapping[str, Mapping[str, Any]],
) -> ExecutionLocks:
    """Acquire every selected scope lock without waiting.

    Parameters
    ----------
    paths : CampaignPaths
        Campaign paths containing the lock namespace.
    seed : int
        Effective evaluation seed.
    scopes : Mapping[str, Mapping[str, Any]]
        Selected final configurations keyed by scope name.

    Returns
    -------
    ExecutionLocks
        Rank-zero lock handles and their absolute paths.

    Raises
    ------
    RuntimeError
        If another invocation owns any requested scope.
    """
    files: list[BinaryIO] = []
    lock_paths = tuple(
        _execution_lock_path(paths, seed, scope).resolve()
        for scope in sorted(scopes)
    )
    error: Optional[str] = None
    if get_is_master():
        try:
            for lock_path in lock_paths:
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                lock_file = lock_path.open("a+b")
                try:
                    fcntl.flock(
                        lock_file.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                except BlockingIOError:
                    lock_file.close()
                    raise RuntimeError(
                        "Another invocation owns the execution lock at "
                        f"{lock_path}."
                    )
                files.append(lock_file)
        except Exception as exc:
            error = str(exc)
            for lock_file in reversed(files):
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            files = []
    error = _broadcast_object(error)
    if error is not None:
        raise RuntimeError(error)
    return ExecutionLocks(tuple(files), lock_paths)


def _release_execution_locks(locks: ExecutionLocks) -> None:
    """Release rank-zero execution locks held by an invocation."""
    if not get_is_master():
        return
    for lock_file in reversed(locks.files):
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def _checkpoint_scope_path(
    paths: CampaignPaths,
    seed: int,
    dataset_name: Optional[str],
) -> Path:
    """Return the ordinary checkpoint directory for one final scope."""
    seed_root = paths.seed_checkpoint_root(seed)
    if dataset_name is None:
        return seed_root / "unified"
    return seed_root / "seperated" / dataset_name


def _maintain_completed_scope(
    paths: CampaignPaths,
    seed: int,
    scoped_config: Mapping[str, Any],
    located: list[Any],
) -> list[str]:
    """Clean disabled leftovers or warn about expected retained checkpoints.

    Parameters
    ----------
    paths : CampaignPaths
        Ordinary campaign paths.
    seed : int
        Effective evaluation seed.
    scoped_config : Mapping[str, Any]
        Selected final configuration.
    located : list[Any]
        Compatible direct completion locations for this scope.

    Returns
    -------
    list[str]
        Absolute checkpoint paths whose cleanup failed.
    """
    if not get_is_master():
        return []
    retain = bool(
        scoped_config.get("logging", {}).get("save_checkpoints", False)
    )
    if not retain:
        dataset_names: list[Optional[str]]
        if scoped_config.get("multitask"):
            dataset_names = [None]
        else:
            dataset_names = [
                item.path.parent.name for item in located
            ]
        failures = []
        for dataset_name in dataset_names:
            checkpoint_root = _checkpoint_scope_path(
                paths,
                seed,
                dataset_name,
            )
            failure = _remove_checkpoint_tree(
                checkpoint_root,
                f"Completed seed {seed}",
            )
            if failure is not None:
                failures.append(failure)
        return failures

    for item in located:
        completion = item.compatibility.completion or {}
        if (
            completion.get("checkpoint_retention_requested") is False
            or completion.get("has_checkpoint") is False
        ):
            continue
        checkpoint_value = completion.get("checkpoint_path")
        checkpoint_path = (
            Path(checkpoint_value)
            if isinstance(checkpoint_value, str) and checkpoint_value
            else None
        )
        if checkpoint_path is None or not checkpoint_path.is_file():
            displayed = (
                checkpoint_path.resolve()
                if checkpoint_path is not None
                else item.path.resolve()
            )
            logger.warning(
                "Completed seed %d dataset %s requested checkpoint "
                "retention, but its checkpoint is missing at %s. The "
                "completion remains authoritative.",
                seed,
                item.path.parent.name,
                displayed,
            )
    return []


def _ignored_configuration_namespaces(log_root: Path) -> list[Path]:
    """Return legacy configuration namespaces excluded from execution.

    Parameters
    ----------
    log_root : pathlib.Path
        Campaign log root to inspect without modification.

    Returns
    -------
    list[pathlib.Path]
        Absolute legacy namespace paths in deterministic order.
    """
    return [
        path.resolve()
        for path in sorted(
            log_root.glob("logs/seed_*/configurations")
        )
        if path.is_dir()
    ]


def _warn_ignored_configuration_namespaces(log_root: Path) -> None:
    """Warn for every preserved legacy configuration namespace."""
    for path in _ignored_configuration_namespaces(log_root):
        logger.warning("Ignoring legacy configuration namespace at %s.", path)


def _copy_artifact(source: Path, destination: Path) -> None:
    """Copy one file or directory without replacing an archive artifact."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def _remove_artifact(path: Path) -> None:
    """Remove one current-attempt artifact when it exists."""
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _replacement_sources(
    paths: CampaignPaths,
    seed: int,
    scoped_config: Mapping[str, Any],
    located: list[Any],
) -> list[tuple[Path, Path]]:
    """Return original artifacts and archive-relative destinations."""
    seed_log_root = paths.seed_log_root(seed)
    sources: list[tuple[Path, Path]] = []
    seen: set[Path] = set()

    def add(source: Path, prefix: str, base: Path) -> None:
        resolved = source.resolve()
        if resolved in seen or not source.exists():
            return
        seen.add(resolved)
        sources.append((source, Path(prefix) / source.relative_to(base)))

    for item in located:
        dataset_name = item.path.parent.name
        completion = item.compatibility.completion or {}
        add(item.path, "log", paths.log_root)
        add(
            seed_log_root / "csv" / f"{dataset_name}.csv",
            "log",
            paths.log_root,
        )
        add(
            seed_log_root / "tensorboard" / dataset_name,
            "log",
            paths.log_root,
        )
        execution_id = completion.get("execution_id")
        if (
            isinstance(execution_id, str)
            and execution_id
            and Path(execution_id).name == execution_id
        ):
            add(
                seed_log_root / "configs" / f"{execution_id}.yaml",
                "log",
                paths.log_root,
            )
            add(
                seed_log_root / "logs" / f"{execution_id}.log",
                "log",
                paths.log_root,
            )
    checkpoint_name = None if scoped_config.get("multitask") else (
        next(iter(scoped_config["data"]["datasets"]))
    )
    add(
        _checkpoint_scope_path(paths, seed, checkpoint_name),
        "checkpoint",
        paths.checkpoint_root,
    )
    return sources


def _replacement_cleanup_paths(
    paths: CampaignPaths,
    seed: int,
    scoped_config: Mapping[str, Any],
    located: list[Any],
) -> tuple[Path, ...]:
    """Return ordinary partial paths removed before restoration."""
    seed_root = paths.seed_log_root(seed)
    cleanup: list[Path] = []
    for item in located:
        dataset_name = item.path.parent.name
        cleanup.extend((
            item.path,
            seed_root / "csv" / f"{dataset_name}.csv",
            seed_root / "tensorboard" / dataset_name,
        ))
    checkpoint_name = None if scoped_config.get("multitask") else (
        next(iter(scoped_config["data"]["datasets"]))
    )
    cleanup.append(
        _checkpoint_scope_path(paths, seed, checkpoint_name)
    )
    return tuple(cleanup)


def _archive_completed_scope(
    paths: CampaignPaths,
    invocation_root: Path,
    seed: int,
    scope: str,
    scoped_config: Mapping[str, Any],
    located: list[Any],
) -> Optional[ReplacementArchive]:
    """Copy a valid completed scope and remove its terminal markers.

    Parameters
    ----------
    paths : CampaignPaths
        Ordinary campaign artifact roots.
    invocation_root : pathlib.Path
        Exclusive current invocation root.
    seed : int
        Effective evaluation seed.
    scope : str
        Dataset or multitask scope being replaced.
    scoped_config : Mapping[str, Any]
        Newly selected final configuration.
    located : list[Any]
        Valid terminal completions authorized for replacement.

    Returns
    -------
    ReplacementArchive or None
        Rank-zero archive metadata, or ``None`` on other ranks.
    """
    if not get_is_master():
        return None
    archive_root = (
        invocation_root
        / "superseded"
        / f"seed_{seed}"
        / semantic_digest({"scope": scope})
    )
    if archive_root.exists():
        raise RuntimeError(
            "Replacement archive already exists at "
            f"{archive_root.resolve()}."
        )
    sources = _replacement_sources(
        paths,
        seed,
        scoped_config,
        located,
    )
    completion_sources = {item.path.resolve() for item in located}
    if not completion_sources:
        raise RuntimeError("Replacement requires a completed result.")
    copies: list[tuple[Path, Path]] = []
    try:
        for source, relative in sources:
            destination = archive_root / relative
            _copy_artifact(source, destination)
            copies.append((source, destination))
        archived_completions = {
            source.resolve()
            for source, _ in copies
            if source.resolve() in completion_sources
        }
        if archived_completions != completion_sources:
            raise RuntimeError(
                "Replacement archive did not capture every completion."
            )
    except Exception:
        if archive_root.exists():
            shutil.rmtree(archive_root)
        raise
    for completion_path in completion_sources:
        completion_path.unlink()
    logger.warning(
        "Archived completed seed %d scope %s before HPO replacement at %s.",
        seed,
        scope,
        archive_root.resolve(),
    )
    cleanup_paths = _replacement_cleanup_paths(
        paths,
        seed,
        scoped_config,
        located,
    )
    return ReplacementArchive(archive_root, tuple(copies), cleanup_paths)


def _restore_completed_scope(archive: Optional[ReplacementArchive]) -> None:
    """Restore an archived completed scope after replacement failure."""
    if archive is None or not get_is_master():
        return
    for partial_path in archive.cleanup_paths:
        if partial_path.exists():
            _remove_artifact(partial_path)
    for original, archived in archive.copies:
        if original.exists() and original not in archive.cleanup_paths:
            _remove_artifact(original)
        _copy_artifact(archived, original)
    logger.warning(
        "Restored completed result from %s after replacement failure.",
        archive.root.resolve(),
    )


def _hpo_scope_root(root: Path, scope: str) -> Path:
    """Return the requested log or checkpoint root for one HPO scope."""
    if scope == "multitask":
        return root / "hpo" / "multitask"
    return root / "hpo" / "datasets" / scope


def _finite_csv_metrics(
    rows: list[Mapping[str, str]],
    dataset_name: str,
    split: str,
    epoch: Optional[int] = None,
) -> Dict[str, float]:
    """Return one unambiguous finite metric event from a shared CSV."""
    values: Dict[str, set[float]] = {}
    for row in rows:
        if row.get("dataset") != dataset_name or row.get("split") != split:
            continue
        if epoch is not None:
            try:
                row_epoch = int(float(row.get("epoch", "")))
            except (TypeError, ValueError):
                continue
            if row_epoch != epoch:
                continue
        metric = row.get("metric")
        try:
            value = float(row.get("value", ""))
        except (TypeError, ValueError):
            continue
        if not metric or not math.isfinite(value):
            continue
        values.setdefault(metric, set()).add(value)
    ambiguous = [
        metric for metric, metric_values in values.items()
        if len(metric_values) != 1
    ]
    if ambiguous:
        raise ValueError(
            f"Dataset '{dataset_name}' has ambiguous {split} metrics in "
            f"the shared CSV: {sorted(ambiguous)}."
        )
    metrics = {
        f"{dataset_name}/{split}/{metric}": next(iter(metric_values))
        for metric, metric_values in values.items()
    }
    if not metrics:
        raise ValueError(
            f"Dataset '{dataset_name}' has no finite {split} metrics in "
            "the shared CSV."
        )
    return metrics


def _shared_validation_epoch(located: list[Any]) -> int:
    """Return the unique validation-best epoch from compatible completions."""
    epochs: set[int] = set()
    for item in located:
        if not item.compatibility.compatible:
            continue
        completion = item.compatibility.completion or {}
        metrics = completion.get("validation_metrics")
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if not str(key).endswith("/eval/epoch"):
                continue
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                continue
            numeric = float(value)
            if math.isfinite(numeric) and numeric.is_integer():
                epochs.add(int(numeric))
    if len(epochs) != 1:
        raise ValueError(
            "Compatible multitask completions do not identify one shared "
            f"validation-best epoch: {sorted(epochs)}."
        )
    return next(iter(epochs))


def _recover_multitask_from_csv(
    paths: CampaignPaths,
    campaign_hash: str,
    seed: int,
    scoped_config: Mapping[str, Any],
    located: list[Any],
    provenance: Optional[Mapping[str, Any]],
    invocation_id: str,
) -> bool:
    """Recover every missing multitask completion from one shared CSV.

    Returns ``False`` without writing when no partial compatible set exists.
    Raises before writing if the shared metric event is ambiguous or missing.
    """
    compatible = [
        item for item in located if item.compatibility.compatible
    ]
    incomplete = [
        item for item in located
        if not item.compatibility.compatible
        and not item.compatibility.terminal
    ]
    conflicts = [
        item for item in located
        if item.compatibility.terminal
        and not item.compatibility.compatible
    ]
    if conflicts or not compatible or not incomplete:
        return False
    csv_path = paths.seed_log_root(seed) / "csv" / "training.csv"
    if not csv_path.is_file():
        raise ValueError(
            "Partial multitask recovery requires the shared CSV at "
            f"{csv_path.resolve()}."
        )
    with csv_path.open(newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"dataset", "split", "epoch", "metric", "value"}
        if reader.fieldnames is None or not required.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"Shared CSV at {csv_path.resolve()} lacks required columns "
                f"{sorted(required)}."
            )
        rows = list(reader)
    best_epoch = _shared_validation_epoch(compatible)
    templates = [
        item.compatibility.completion or {} for item in compatible
    ]
    execution_ids = {
        item.get("execution_id") for item in templates
        if isinstance(item.get("execution_id"), str)
    }
    if len(execution_ids) != 1:
        raise ValueError(
            "Compatible multitask completions do not share one execution ID."
        )
    template = templates[0]
    retain = bool(
        scoped_config.get("logging", {}).get("save_checkpoints", False)
    )
    checkpoint_values = {
        item.get("checkpoint_path") for item in templates
        if isinstance(item.get("checkpoint_path"), str)
        and Path(item["checkpoint_path"]).is_file()
    }
    checkpoint_value = (
        next(iter(checkpoint_values))
        if retain and len(checkpoint_values) == 1
        else None
    )
    payloads: list[tuple[Path, Dict[str, Any]]] = []
    for item in incomplete:
        dataset_name = item.path.parent.name
        validation_metrics = _finite_csv_metrics(
            rows,
            dataset_name,
            "eval",
            epoch=best_epoch,
        )
        test_metrics = _finite_csv_metrics(
            rows,
            dataset_name,
            "test",
        )
        payloads.append((item.path, {
            "status": "completed",
            "campaign_hash": campaign_hash,
            "campaign_identity_version": IDENTITY_VERSION,
            "config_hash": _config_hash(scoped_config),
            "config_hash_version": IDENTITY_VERSION,
            "seed": seed,
            "dataset_config": scoped_config["data"]["datasets"][
                dataset_name
            ],
            "execution_id": next(iter(execution_ids)),
            "invocation_id": invocation_id,
            "has_checkpoint": checkpoint_value is not None,
            "checkpoint_path": checkpoint_value,
            "checkpoint_retention_requested": retain,
            "selection_provenance": dict(provenance or {}),
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "completed_at": datetime.datetime.now().isoformat(),
            "batching": template.get("batching"),
            "recovered_from_csv": str(csv_path.resolve()),
        }))
    for completion_path, payload in payloads:
        _atomic_json(completion_path, payload)
        logger.warning(
            "Recovered missing multitask completion from %s at %s.",
            csv_path.resolve(),
            completion_path.resolve(),
        )
    return True


def _read_sqlite_studies(storage_path: Path) -> list[Dict[str, Any]]:
    """Read Optuna study status using a read-only SQLite connection.

    Parameters
    ----------
    storage_path : pathlib.Path
        Existing Optuna SQLite database.

    Returns
    -------
    list[dict[str, Any]]
        Study names, terminal-state counts, and best complete trial metadata.

    Raises
    ------
    RuntimeError
        If the database schema cannot be read.
    """
    if not storage_path.is_file():
        return []
    uri = f"file:{storage_path.resolve()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        studies = connection.execute(
            "SELECT study_id, study_name FROM studies ORDER BY study_id"
        ).fetchall()
        records: list[Dict[str, Any]] = []
        for study in studies:
            state_rows = connection.execute(
                "SELECT state, COUNT(*) AS count FROM trials "
                "WHERE study_id = ? GROUP BY state",
                (study["study_id"],),
            ).fetchall()
            state_counts = {
                row["state"]: int(row["count"])
                for row in state_rows
            }
            direction_row = connection.execute(
                "SELECT direction FROM study_directions "
                "WHERE study_id = ? AND objective = 0",
                (study["study_id"],),
            ).fetchone()
            distribution_rows = connection.execute(
                "SELECT trial_params.param_name AS name, "
                "trial_params.distribution_json AS distribution "
                "FROM trial_params JOIN trials ON "
                "trial_params.trial_id = trials.trial_id "
                "WHERE trials.study_id = ? ORDER BY trial_params.param_id",
                (study["study_id"],),
            ).fetchall()
            distributions: Dict[str, Any] = {}
            distributions_consistent = True
            for row in distribution_rows:
                parsed_distribution = json.loads(row["distribution"])
                previous = distributions.get(row["name"])
                if (
                    previous is not None
                    and previous != parsed_distribution
                ):
                    distributions_consistent = False
                distributions[row["name"]] = parsed_distribution
            direction = (
                direction_row["direction"]
                if direction_row is not None
                else None
            )
            order = "ASC" if direction == "MINIMIZE" else "DESC"
            best_row = connection.execute(
                "SELECT trials.number AS number, "
                "trial_values.value AS value "
                "FROM trials JOIN trial_values ON "
                "trials.trial_id = trial_values.trial_id "
                "WHERE trials.study_id = ? "
                "AND trials.state = 'COMPLETE' "
                "AND trial_values.objective = 0 "
                f"ORDER BY trial_values.value {order} LIMIT 1",
                (study["study_id"],),
            ).fetchone()
            records.append({
                "study_name": study["study_name"],
                "storage_path": str(storage_path.resolve()),
                "direction": direction,
                "distributions": distributions,
                "distributions_consistent": distributions_consistent,
                "complete": state_counts.get("COMPLETE", 0),
                "pruned": state_counts.get("PRUNED", 0),
                "failed": state_counts.get("FAIL", 0),
                "running": state_counts.get("RUNNING", 0),
                "best_trial": (
                    int(best_row["number"])
                    if best_row is not None
                    else None
                ),
                "best_value": (
                    float(best_row["value"])
                    if best_row is not None
                    else None
                ),
            })
        return records
    except (json.JSONDecodeError, sqlite3.Error) as exc:
        raise RuntimeError(
            f"Cannot audit Optuna storage at {storage_path.resolve()}: {exc}."
        ) from exc
    finally:
        if "connection" in locals():
            connection.close()


def _study_progress(trials: list[Any]) -> tuple[int, int]:
    """Count budgeted trials and the trailing terminal-failure streak.

    Parameters
    ----------
    trials : list[Any]
        Persisted Optuna trials in chronological order. Each element must
        expose a state whose ``name`` is an Optuna trial-state name.

    Returns
    -------
    tuple[int, int]
        Count of complete-or-pruned trials and consecutive failed trials at
        the end of the persisted history.

    Raises
    ------
    ValueError
        If a trial does not expose a recognized state name.
    """
    budgeted_names = frozenset({"COMPLETE", "PRUNED"})
    terminal_names = budgeted_names | {"FAIL"}
    state_names = [trial.state.name for trial in trials]
    unknown = set(state_names) - terminal_names - {"RUNNING", "WAITING"}
    if unknown:
        raise ValueError(
            "Expected recognized Optuna trial states, but got "
            f"{sorted(unknown)}."
        )
    budgeted = sum(name in budgeted_names for name in state_names)
    consecutive_failures = 0
    for state_name in reversed(state_names):
        if state_name != "FAIL":
            break
        consecutive_failures += 1
    return budgeted, consecutive_failures


def _expected_trial_distributions(
    hpo_config: HpoConfig,
) -> Dict[str, Any]:
    """Build exact Optuna distributions for persisted-trial validation.

    Parameters
    ----------
    hpo_config : HpoConfig
        Current semantic HPO configuration.

    Returns
    -------
    dict[str, Any]
        Optuna distributions keyed by dotted search path.
    """
    import optuna

    expected: Dict[str, Any] = {}
    for path, distribution in hpo_config.search_space.items():
        if distribution.distribution == "float":
            expected[path] = optuna.distributions.FloatDistribution(
                low=float(distribution.low),
                high=float(distribution.high),
                log=distribution.log,
                step=distribution.step,
            )
        elif distribution.distribution == "int":
            step = (
                1
                if distribution.step is None
                else int(distribution.step)
            )
            expected[path] = optuna.distributions.IntDistribution(
                low=int(distribution.low),
                high=int(distribution.high),
                log=distribution.log,
                step=step,
            )
        else:
            choices = tuple(
                f"choice_{index:04d}"
                for index in range(len(distribution.choices or []))
            )
            expected[path] = (
                optuna.distributions.CategoricalDistribution(choices)
            )
    return expected


def _serialized_trial_distributions(
    hpo_config: HpoConfig,
) -> Dict[str, Any]:
    """Return JSON-compatible expected Optuna distributions.

    Parameters
    ----------
    hpo_config : HpoConfig
        Current semantic HPO configuration.

    Returns
    -------
    dict[str, Any]
        Persisted Optuna distribution payloads by search path.
    """
    import optuna

    return {
        path: json.loads(optuna.distributions.distribution_to_json(value))
        for path, value in _expected_trial_distributions(hpo_config).items()
    }


def _decoded_trial_parameters_match(
    trial: Any,
    hpo_config: HpoConfig,
) -> bool:
    """Return whether persisted decoded values match Optuna parameters.

    Parameters
    ----------
    trial : Any
        Persisted Optuna frozen trial.
    hpo_config : HpoConfig
        Current semantic HPO configuration.

    Returns
    -------
    bool
        Whether every winner-eligible trial has exact decoded values.
        Nonterminal trials may omit decoded values.
    """
    decoded = trial.user_attrs.get("decoded_params")
    if decoded is None:
        return trial.state.name not in {"COMPLETE", "PRUNED"}
    if not isinstance(decoded, dict):
        return False
    if set(decoded) != set(hpo_config.search_space):
        return False
    for path, distribution in hpo_config.search_space.items():
        raw_value = trial.params.get(path)
        if distribution.distribution != "categorical":
            if decoded[path] != raw_value:
                return False
            continue
        prefix = "choice_"
        if not isinstance(raw_value, str) or not raw_value.startswith(prefix):
            return False
        try:
            choice_index = int(raw_value[len(prefix):])
            expected_value = (distribution.choices or [])[choice_index]
        except (IndexError, ValueError):
            return False
        if decoded[path] != expected_value:
            return False
    return True


def _trial_metadata_matches(
    trials: list[Any],
    hpo_config: HpoConfig,
) -> bool:
    """Return whether trials prove exact search-space compatibility.

    Parameters
    ----------
    trials : list[Any]
        Persisted Optuna frozen trials from one study.
    hpo_config : HpoConfig
        Current semantic HPO configuration.

    Returns
    -------
    bool
        Whether every parameterized trial uses the exact current metadata.
    """
    expected = _expected_trial_distributions(hpo_config)
    parameterized = [trial for trial in trials if trial.distributions]
    if not parameterized:
        return False
    return all(
        trial.distributions == expected
        and _decoded_trial_parameters_match(trial, hpo_config)
        for trial in parameterized
    )


def _synchronize_micro_batch(micro_batch: int) -> int:
    """Select the smallest safe micro-batch across distributed ranks.

    Parameters
    ----------
    micro_batch : int
        Largest candidate predicted safe on the current rank.

    Returns
    -------
    int
        Minimum candidate reported by any initialized distributed rank.
    """
    if micro_batch <= 0:
        raise ValueError(
            f"Expected a positive micro-batch, but got {micro_batch}."
        )
    if not (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        return micro_batch
    device = torch.device("cuda", torch.cuda.current_device())
    value = torch.tensor(micro_batch, device=device, dtype=torch.int64)
    torch.distributed.all_reduce(
        value,
        op=torch.distributed.ReduceOp.MIN,
    )
    return int(value.item())


def _new_invocation_id() -> str:
    """Return a collision-resistant local invocation identifier.

    Returns
    -------
    str
        UTC timestamp with microseconds followed by the process identifier.
    """
    timestamp = datetime.datetime.now(
        datetime.timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{os.getpid()}"


class CampaignRunner:
    """Coordinate HPO, seed runs, failure policy, and summaries."""

    def __init__(
        self,
        config: BaseModel,
        hpo_config: HpoConfig,
        config_class: Type[BaseModel],
    ):
        self.config = config
        self.config_class = config_class
        self.base_dict = config.model_dump(mode="json")
        self.hpo_config = hpo_config
        self.hpo_dict = hpo_config.model_dump(mode="json")
        self.resolution: CampaignResolution = resolve_campaign(
            run_dir=self.base_dict["logging"]["run_dir"],
            model_type=self.base_dict["model_type"],
            experiment_name=(
                self.base_dict["logging"]["experiment_name"]
            ),
            config=self.base_dict,
            hpo=self.hpo_dict,
        )
        self.campaign_hash = self.resolution.campaign_identity
        self.campaign_aliases = self.resolution.aliases
        self.campaign_semantic_config = dict(
            self.resolution.semantic_config
        )
        self.paths = self.resolution.paths
        self.invocation_id = _new_invocation_id()
        self.invocation_root = (
            self.paths.log_root / "invocations" / self.invocation_id
        )
        self.distributed_initialized = False
        self._active_study_runtime: Optional[StudyRuntime] = None
        self.selection_provenance: Dict[str, Dict[str, Any]] = {}
        self.checkpoint_cleanup_failures: list[str] = []
        self.final_dataset_attempts: list[Dict[str, Any]] = []
        self._ignored_namespace_warnings: set[Path] = set()
        self.adaptive_batch_cache: Dict[str, Dict[str, Any]] = {}
        self.last_adaptive_memory_profile: Dict[str, Any] = {}

    def _configure_campaign_logging(self) -> None:
        """Configure campaign console logging and Optuna verbosity."""
        logging_config = self.base_dict.get("logging", {})
        level = str(logging_config.get("level", "info")).upper()
        setup_log(
            start_time=None,
            name="baseline",
            level=level,
        )
        try:
            import optuna
        except ImportError:
            return
        optuna_level = (
            optuna.logging.DEBUG
            if level == "DEBUG"
            else optuna.logging.WARNING
        )
        optuna.logging.set_verbosity(optuna_level)

    def _adaptive_signature(self, config: BaseModel) -> str:
        """Return an invocation-local memory-profile identity."""
        payload = config.model_dump(mode="json")
        training = payload["training"]
        signature = {
            "model_type": payload["model_type"],
            "model": payload["model"],
            "data": payload["data"],
            "fs": payload["fs"],
            "use_amp": training["use_amp"],
            "freeze_encoder": training["freeze_encoder"],
            "lora": training["lora"],
            "world_size": get_world_size(),
        }
        if torch.cuda.is_available():
            signature["gpu"] = torch.cuda.get_device_name(
                torch.cuda.current_device()
            )
        return json.dumps(signature, sort_keys=True)

    def _measure_adaptive_memory(
        self,
        config: BaseModel,
        global_batch_size: int,
        world_size: int,
    ) -> tuple[str, Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """Return a current or newly calibrated CUDA memory model."""
        batching = config.training.adaptive_batching
        signature = self._adaptive_signature(config)
        cache = getattr(self, "adaptive_batch_cache", {})
        cached = cache.get(signature)
        if cached is not None:
            return signature, cache, dict(cached)

        calibration_retried = False
        while True:
            calibration_config = config.model_copy(deep=True)
            calibration = ModelRegistry.create_trainer(
                calibration_config
            )
            calibration.configure_runtime_batching(
                global_batch_size,
                1,
                world_size,
            )
            try:
                memory_model = calibration.calibrate_cuda_memory()
            except BaseException as exc:
                self.last_adaptive_memory_profile = dict(
                    calibration.adaptive_batch_profile
                )
                if not is_cuda_oom(exc):
                    raise
                if calibration_retried:
                    raise torch.cuda.OutOfMemoryError(
                        "The model's fixed CUDA state cannot fit while "
                        "preserving the configured memory reserve after "
                        "two micro-batch-one calibration attempts."
                    ) from exc
                calibration_retried = True
                wait_seconds = batching.contention_wait_seconds
                logger.warning(
                    "Micro-batch-one calibration cannot fit; releasing "
                    "CUDA state and retrying in %d seconds.",
                    wait_seconds,
                )
                calibration = None
                _release_training_state()
                if wait_seconds:
                    time.sleep(wait_seconds)
                continue
            finally:
                calibration = None
                _release_training_state()
            cache[signature] = memory_model
            self.adaptive_batch_cache = cache
            return signature, cache, memory_model

    def _predicted_candidates(
        self,
        candidates: list[int],
        memory_model: Mapping[str, Any],
    ) -> tuple[list[int], int]:
        """Return candidates starting at the DDP-wide predicted maximum."""
        eligible = candidates
        oom_cap = memory_model.get("oom_cap")
        if oom_cap is not None:
            eligible = [
                candidate
                for candidate in candidates
                if candidate <= int(oom_cap)
            ]
        selected, _ = select_safe_micro_batch(
            eligible,
            int(memory_model["estimated_fixed_bytes"]),
            int(memory_model["calibration_peak_reserved_bytes"]),
            int(memory_model["calibration_batch_size"]),
            int(memory_model["process_limit_bytes"]),
        )
        selected = _synchronize_micro_batch(selected)
        _, predicted_peak = select_safe_micro_batch(
            [selected],
            int(memory_model["estimated_fixed_bytes"]),
            int(memory_model["calibration_peak_reserved_bytes"]),
            int(memory_model["calibration_batch_size"]),
            int(memory_model["process_limit_bytes"]),
        )
        return candidates[candidates.index(selected):], predicted_peak

    def _run_adaptive_trainer(
        self,
        config: BaseModel,
        prepare: Callable[[Any], None],
    ) -> Dict[str, Any]:
        """Run one neural config with internal exact-divisor OOM backoff."""
        batching = config.training.adaptive_batching
        world_size = get_world_size()
        global_batch_size = config.data.batch_size
        candidates = derive_batch_candidates(
            global_batch_size,
            world_size,
            batching.enabled,
        )
        self.last_adaptive_memory_profile = {}
        signature = self._adaptive_signature(config)
        cache = getattr(self, "adaptive_batch_cache", {})
        memory_profile: Dict[str, Any] = {}
        if batching.enabled and torch.cuda.is_available():
            signature, cache, memory_profile = (
                self._measure_adaptive_memory(
                    config,
                    global_batch_size,
                    world_size,
                )
            )
            device = torch.device(
                "cuda",
                torch.cuda.current_device(),
            )
            current_limit = configure_cuda_allocator(
                device,
                batching.memory_reserve_fraction,
            )
            if current_limit is None:
                raise RuntimeError(
                    "CUDA allocator configuration returned no limit."
                )
            memory_profile = dict(memory_profile)
            memory_profile.update(current_limit.as_dict())
            candidates, predicted_peak = self._predicted_candidates(
                candidates,
                memory_profile,
            )
            memory_profile["estimated_selected_peak_bytes"] = (
                predicted_peak
            )
            self.last_adaptive_memory_profile = dict(
                memory_profile
            )
        batch_one_retried = False
        last_error: Optional[BaseException] = None
        for candidate_index, micro_batch in enumerate(candidates):
            while True:
                attempt_config = config.model_copy(deep=True)
                trainer = ModelRegistry.create_trainer(attempt_config)
                trainer.configure_runtime_batching(
                    global_batch_size,
                    micro_batch,
                    world_size,
                )
                trainer.adaptive_batch_profile["adaptive_retry"] = (
                    candidate_index
                )
                trainer.adaptive_batch_profile.update(memory_profile)
                try:
                    prepare(trainer)
                    result = trainer.run()
                except BaseException as exc:
                    last_error = exc
                    self.last_adaptive_memory_profile = dict(
                        trainer.adaptive_batch_profile
                    )
                    if not is_cuda_oom(exc):
                        raise
                    logger.warning(
                        "CUDA OOM at micro-batch %d; recalibrating within "
                        "the same run attempt.",
                        micro_batch,
                    )
                    if micro_batch != 1:
                        next_micro_batch = candidates[
                            candidate_index + 1
                        ]
                        if memory_profile:
                            memory_profile["oom_cap"] = next_micro_batch
                            cache[signature] = dict(memory_profile)
                            self.adaptive_batch_cache = cache
                        break
                    if batch_one_retried:
                        raise torch.cuda.OutOfMemoryError(
                            "The model's fixed CUDA state cannot fit while "
                            "preserving the configured memory reserve after "
                            "two micro-batch-one runtime attempts."
                        ) from exc
                    batch_one_retried = True
                    wait_seconds = batching.contention_wait_seconds
                    if wait_seconds:
                        logger.warning(
                            "Micro-batch one cannot fit; waiting %d seconds "
                            "before one clean retry.",
                            wait_seconds,
                        )
                        trainer = None
                        _release_training_state()
                        time.sleep(wait_seconds)
                    continue
                finally:
                    trainer = None
                    _release_training_state()
                return result
        if last_error is None:
            raise RuntimeError("Adaptive batching produced no candidates.")
        raise last_error

    def _initialize_distributed(self, seed: int) -> None:
        """Initialize one process group reused by all neural executions."""
        if self.base_dict["model_type"] in DETERMINISTIC_MODELS:
            return
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            self.distributed_initialized = True
            return
        config_dict = _scoped_config(self.base_dict, seed)
        bootstrap_config = self.config_class.model_validate(config_dict)
        trainer = ModelRegistry.create_trainer(bootstrap_config)
        trainer.setup_distributed()
        self.distributed_initialized = True
        del trainer

    def _identity_payload(self) -> Dict[str, Any]:
        """Return immutable campaign identity metadata.

        Returns
        -------
        dict[str, Any]
            Version, full identity, and non-authoritative display prefix.
        """
        return {
            "identity_version": IDENTITY_VERSION,
            "campaign_identity": self.campaign_hash,
            "display_id": short_identity(self.campaign_hash),
        }

    def _invocation_payload(self) -> Dict[str, Any]:
        """Return the complete resolved invocation configuration.

        Returns
        -------
        dict[str, Any]
            Operational and semantic parameters for this invocation.
        """
        return {
            "invocation_id": self.invocation_id,
            "campaign_identity": self.campaign_hash,
            "created_at": self.invocation_started_at,
            "campaign_log_root": str(self.paths.log_root.resolve()),
            "campaign_checkpoint_root": str(
                self.paths.checkpoint_root.resolve()
            ),
            "model_config": copy.deepcopy(self.base_dict),
            "hpo": copy.deepcopy(self.hpo_dict),
        }

    def _update_invocation_status(
        self,
        state: str,
        invocation: Optional[Mapping[str, Any]] = None,
        dataset_pairs: Optional[Mapping[str, int]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Atomically update this invocation's compact lifecycle status.

        Parameters
        ----------
        state : str
            Current lifecycle state such as ``running`` or ``complete``.
        invocation : Mapping[str, Any], optional
            Seed lifecycle fields, default=None.
        dataset_pairs : Mapping[str, int], optional
            Expected and compatible pair counts, default=None.
        error : str, optional
            Normalized terminal error fingerprint, default=None.
        """
        if (
            not get_is_master()
            or not hasattr(self, "invocation_root")
        ):
            return
        payload: Dict[str, Any] = {
            "invocation_id": self.invocation_id,
            "campaign_identity": self.campaign_hash,
            "state": state,
            "updated_at": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
        }
        if self.checkpoint_cleanup_failures:
            payload["checkpoint_cleanup_failures"] = list(
                dict.fromkeys(self.checkpoint_cleanup_failures)
            )
        if invocation is not None:
            payload["seeds"] = dict(invocation)
        if dataset_pairs is not None:
            payload["dataset_pairs"] = dict(dataset_pairs)
        if error is not None:
            payload["error_fingerprint"] = error
        _atomic_json(self.invocation_root / "status.json", payload)

    def _save_campaign_config(self) -> None:
        """Persist immutable semantic and invocation metadata."""
        if not get_is_master():
            return
        self.paths.log_root.mkdir(parents=True, exist_ok=True)
        if self.base_dict["model_type"] not in DETERMINISTIC_MODELS:
            self.paths.checkpoint_root.mkdir(parents=True, exist_ok=True)
        campaign_path = self.paths.log_root / "campaign.yaml"
        if not campaign_path.exists():
            try:
                _exclusive_yaml(
                    campaign_path,
                    self.campaign_semantic_config,
                )
            except FileExistsError:
                stored = yaml.safe_load(
                    campaign_path.read_text(encoding="utf-8")
                )
                if stored != self.campaign_semantic_config:
                    raise RuntimeError(
                        "A concurrent process created a mismatching campaign "
                        f"manifest at {campaign_path.resolve()}."
                    )
        elif not self.resolution.legacy:
            stored = yaml.safe_load(
                campaign_path.read_text(encoding="utf-8")
            )
            if stored != self.campaign_semantic_config:
                raise RuntimeError(
                    "Refusing to overwrite a mismatching campaign manifest "
                    f"at {campaign_path.resolve()}."
                )
        identity_path = self.paths.log_root / "identity.json"
        identity_payload = self._identity_payload()
        if identity_path.exists():
            if not self.resolution.legacy:
                _validate_immutable_json(identity_path, identity_payload)
        else:
            try:
                _exclusive_json(identity_path, identity_payload)
            except FileExistsError:
                _validate_immutable_json(
                    identity_path,
                    identity_payload,
                )

        self.invocation_started_at = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        self.invocation_root.mkdir(parents=True, exist_ok=False)
        _exclusive_yaml(
            self.invocation_root / "invocation.yaml",
            self._invocation_payload(),
        )
        self._update_invocation_status("running")
        if self.resolution.legacy:
            logger.info(
                "Reusing semantically matching legacy campaign at %s.",
                self.paths.log_root.resolve(),
            )
        logger.info("Campaign log root: %s", self.paths.log_root.resolve())
        if self.base_dict["model_type"] not in DETERMINISTIC_MODELS:
            logger.info(
                "Campaign checkpoint root: %s",
                self.paths.checkpoint_root.resolve(),
            )
        result_layout = (
            "datasets/, csv/, configs/, logs/, and summary/"
            if self.paths.flat_results
            else "hpo/, logs/seed_<seed>/, and summary/"
        )
        logger.info(
            "Campaign layout: campaign.yaml, identity.json, invocations/, "
            "%s under %s.",
            result_layout,
            self.paths.log_root.resolve(),
        )

    def _study_identity(self, scope: str) -> tuple[str, Dict[str, Any]]:
        """Return the semantic identity and payload for one HPO scope.

        Parameters
        ----------
        scope : str
            Dataset name or ``multitask``.

        Returns
        -------
        tuple[str, dict[str, Any]]
            Full SHA-256 study identity and its canonical payload.
        """
        payload = {
            "identity_version": IDENTITY_VERSION,
            "campaign_identity": self.campaign_hash,
            "scope": scope,
        }
        return semantic_digest(payload), payload

    def _study_sampler_and_pruner(self) -> tuple[Any, Any]:
        """Construct the configured Optuna sampler and pruner.

        Returns
        -------
        tuple[Any, Any]
            Fresh TPE sampler and median pruner instances.
        """
        import optuna

        sampler = optuna.samplers.TPESampler(
            seed=self.hpo_config.sampler.seed,
            n_startup_trials=(
                self.hpo_config.sampler.n_startup_trials
            ),
        )
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=(
                self.hpo_config.pruner.n_startup_trials
            ),
            n_warmup_steps=self.hpo_config.pruner.n_warmup_epochs,
            interval_steps=self.hpo_config.pruner.interval_epochs,
        )
        return sampler, pruner

    def _legacy_study(
        self,
        storage_path: Path,
        scope: str,
        storage: str,
    ) -> tuple[Optional[Any], tuple[str, ...]]:
        """Select a unique compatible legacy study without touching others.

        Parameters
        ----------
        storage_path : pathlib.Path
            Existing legacy SQLite path.
        scope : str
            Dataset name or ``multitask``.
        storage : str
            Optuna SQLite storage URL.

        Returns
        -------
        tuple[Any or None, tuple[str, ...]]
            Selected study and preserved non-selected study names.

        Raises
        ------
        RuntimeError
            If multiple equally eligible legacy studies are compatible.
        """
        import optuna

        if not storage_path.is_file():
            return None, ()
        summaries = optuna.study.get_all_study_summaries(storage=storage)
        direction = self.hpo_config.objective.direction.upper()
        compatible: list[tuple[Any, int, bool]] = []
        all_names: list[str] = []
        expected_names = {
            f"{alias}-{scope}"
            for alias in self.campaign_aliases
        }
        for summary in summaries:
            all_names.append(summary.study_name)
            if not summary.study_name.endswith(f"-{scope}"):
                continue
            if summary.direction.name != direction:
                continue
            sampler, pruner = self._study_sampler_and_pruner()
            candidate = optuna.load_study(
                study_name=summary.study_name,
                storage=storage,
                sampler=sampler,
                pruner=pruner,
            )
            trials = candidate.get_trials(deepcopy=False)
            if not _trial_metadata_matches(trials, self.hpo_config):
                continue
            budgeted, _ = _study_progress(trials)
            compatible.append((
                candidate,
                budgeted,
                summary.study_name in expected_names,
            ))

        complete = [
            item for item in compatible
            if item[1] >= self.hpo_config.n_trials
        ]
        matching_partial = [
            item for item in compatible if item[2]
        ]
        if complete:
            eligible = complete
        elif matching_partial:
            eligible = matching_partial
        else:
            eligible = compatible
        if len(eligible) > 1:
            names = ", ".join(
                study.study_name for study, _, _ in eligible
            )
            raise RuntimeError(
                f"Multiple compatible legacy HPO studies exist for scope "
                f"'{scope}' in {storage_path.resolve()}: {names}."
            )
        selected = eligible[0][0] if eligible else None
        selected_name = selected.study_name if selected is not None else None
        duplicates = tuple(
            name for name in all_names if name != selected_name
        )
        return selected, duplicates

    def _create_study(self, scope: str) -> StudyRuntime:
        """Create or resume one exclusively locked rank-zero Optuna study."""
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "HPO requires Optuna. Install project requirements or run "
                "'pip install optuna'."
            ) from exc

        study_identity, identity_payload = self._study_identity(scope)
        scope_root = _hpo_scope_root(self.paths.log_root, scope)
        artifact_root = scope_root / "studies" / study_identity
        checkpoint_root = (
            _hpo_scope_root(self.paths.checkpoint_root, scope)
            / "studies"
            / study_identity
        )
        artifact_root.mkdir(parents=True, exist_ok=True)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        lock_path = artifact_root / ".study.lock"
        lock_file = lock_path.open("a+b")
        try:
            fcntl.flock(
                lock_file.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            lock_file.close()
            raise RuntimeError(
                "Another process is already writing semantic HPO study "
                f"{study_identity} at {artifact_root.resolve()}."
            ) from exc

        legacy_path = (scope_root / "study.sqlite3").resolve()
        legacy_storage = f"sqlite:///{legacy_path}"
        try:
            study, duplicates = self._legacy_study(
                legacy_path,
                scope,
                legacy_storage,
            )
            legacy = study is not None
            if legacy:
                storage_path = legacy_path
                storage = legacy_storage
            else:
                storage_path = (artifact_root / "study.sqlite3").resolve()
                storage = f"sqlite:///{storage_path}"
                sampler, pruner = self._study_sampler_and_pruner()
                study = optuna.create_study(
                    study_name=study_identity,
                    storage=storage,
                    sampler=sampler,
                    pruner=pruner,
                    direction=self.hpo_config.objective.direction,
                    load_if_exists=True,
                )
                stored_payload = study.user_attrs.get("semantic_payload")
                expected_payload = canonical_json(identity_payload)
                if stored_payload is None:
                    study.set_user_attr(
                        "identity_version",
                        IDENTITY_VERSION,
                    )
                    study.set_user_attr(
                        "study_identity",
                        study_identity,
                    )
                    study.set_user_attr(
                        "semantic_payload",
                        expected_payload,
                    )
                elif stored_payload != expected_payload:
                    raise RuntimeError(
                        "Study identity collision or corrupt semantic metadata "
                        f"at {storage_path}."
                    )
            artifact_roots = (
                (artifact_root, scope_root)
                if legacy
                else (artifact_root,)
            )
            checkpoint_scope_root = _hpo_scope_root(
                self.paths.checkpoint_root,
                scope,
            )
            checkpoint_roots = (
                (checkpoint_root, checkpoint_scope_root)
                if legacy
                else (checkpoint_root,)
            )
            removed_trials = _cleanup_unbudgeted_hpo_trials(
                storage_path,
                study.study_name,
                artifact_roots,
                checkpoint_roots,
            )
            if removed_trials:
                logger.warning(
                    "Removed failed or stale HPO trials for scope %s before "
                    "resume: %s.",
                    scope,
                    ", ".join(str(number) for number in removed_trials),
                )
                sampler, pruner = self._study_sampler_and_pruner()
                study = optuna.load_study(
                    study_name=study.study_name,
                    storage=storage,
                    sampler=sampler,
                    pruner=pruner,
                )
            if duplicates:
                logger.warning(
                    "Preserving non-selected HPO studies for scope %s in %s: "
                    "%s.",
                    scope,
                    legacy_path,
                    ", ".join(duplicates),
                )
            return StudyRuntime(
                study=study,
                study_identity=study_identity,
                storage_path=storage_path,
                artifact_root=artifact_root,
                checkpoint_root=checkpoint_root,
                legacy=legacy,
                duplicate_names=duplicates,
                lock_file=lock_file,
                removed_trial_numbers=removed_trials,
            )
        except Exception:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
            raise

    def _release_study(self, runtime: StudyRuntime) -> None:
        """Release one study's process lock.

        Parameters
        ----------
        runtime : StudyRuntime
            Runtime whose lock is currently held.
        """
        fcntl.flock(runtime.lock_file.fileno(), fcntl.LOCK_UN)
        runtime.lock_file.close()

    def _write_trial_status(
        self,
        trial_root: Path,
        status: str,
        decoded_params: Mapping[str, Any],
        objective: Optional[float] = None,
        error: Optional[str] = None,
        objective_history: Optional[list[Mapping[str, Any]]] = None,
        memory_information: Optional[Mapping[str, Any]] = None,
        performance: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Persist one trial's decoded parameters and terminal state."""
        if not get_is_master():
            return
        payload: Dict[str, Any] = {
            "status": status,
            "parameters": dict(decoded_params),
        }
        if objective is not None:
            payload["objective"] = objective
        if error is not None:
            payload["error"] = error
        if objective_history is not None:
            payload["objective_history"] = list(objective_history)
        if memory_information is not None:
            payload["memory_information"] = dict(memory_information)
        if performance is not None:
            payload["performance"] = dict(performance)
        _exclusive_json(trial_root / "trial.json", payload)

    def _run_hpo_scope(
        self,
        scope: str,
        scope_config: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Run or resume one HPO scope and return its selected config."""
        import optuna
        self._configure_campaign_logging()

        runtime = self._create_study(scope) if get_is_master() else None
        self._active_study_runtime = runtime
        study = runtime.study if runtime is not None else None
        study_identity, _ = self._study_identity(scope)
        artifact_root = (
            _hpo_scope_root(self.paths.log_root, scope)
            / "studies"
            / study_identity
        )
        checkpoint_study_root = (
            _hpo_scope_root(self.paths.checkpoint_root, scope)
            / "studies"
            / study_identity
        )
        if get_is_master():
            persisted_trials = study.get_trials(deepcopy=False)
            budgeted, consecutive_failures = _study_progress(
                persisted_trials
            )
            study_status = (
                "resumed"
                if persisted_trials or runtime.removed_trial_numbers
                else "new"
            )
        else:
            budgeted = 0
            consecutive_failures = 0
            study_status = "new"
        progress = _broadcast_object({
            "budgeted": budgeted,
            "consecutive_failures": consecutive_failures,
            "study_status": study_status,
        })
        budgeted = int(progress["budgeted"])
        consecutive_failures = int(progress["consecutive_failures"])
        study_status = str(progress["study_status"])
        if get_is_master():
            study_path = runtime.storage_path
            logger.info(
                "HPO scope %s: %s study, budget %d/%d at %s.",
                scope,
                study_status,
                budgeted,
                self.hpo_config.n_trials,
                study_path,
            )
        failure_limit = self.hpo_config.max_consecutive_failed_trials
        if consecutive_failures >= failure_limit:
            raise RuntimeError(
                f"HPO scope '{scope}' already has {consecutive_failures} "
                "consecutive failed trials. Correct the configuration or "
                "search space before resuming."
            )

        while budgeted < self.hpo_config.n_trials:
            if get_is_master():
                trial = study.ask()
                sampled, decoded = sample_config(
                    scope_config,
                    self.hpo_config,
                    trial,
                )
                payload = {
                    "number": trial.number,
                    "config": sampled,
                    "decoded": decoded,
                }
            else:
                trial = None
                payload = None
            payload = _broadcast_object(payload)
            trial_number = int(payload["number"])
            sampled = dict(payload["config"])
            decoded = dict(payload["decoded"])
            sampled["seeds"] = [self.hpo_config.seed]
            _force_local_trial_logging(sampled)

            trial_root = (
                artifact_root
                / "trials"
                / f"trial_{trial_number:05d}"
            )
            if get_is_master():
                trial_root.mkdir(parents=True, exist_ok=False)
                _exclusive_yaml(
                    trial_root / "resolved_config.yaml",
                    sampled,
                )
            checkpoint_root = (
                checkpoint_study_root
                / f"trial_{trial_number:05d}"
            )
            best_value: Optional[float] = None
            objective_history: list[Dict[str, Any]] = []
            trial_budgeted = False
            self.last_adaptive_memory_profile = {}

            def validation_callback(
                epoch: int,
                metrics: Mapping[str, Mapping[str, float]],
                train_sizes: Mapping[str, int],
            ) -> bool:
                nonlocal best_value
                values = objective_values(
                    metrics,
                    self.hpo_config.objective.metric,
                )
                value = reduce_objective(
                    values,
                    self.hpo_config.objective.multitask_reduction,
                    train_sizes,
                )
                if not math.isfinite(value):
                    raise ValueError(
                        f"Trial objective is not finite: {value}."
                    )
                if best_value is None:
                    best_value = value
                elif self.hpo_config.objective.direction == "minimize":
                    best_value = min(best_value, value)
                else:
                    best_value = max(best_value, value)
                objective_history.append({
                    "epoch": epoch,
                    "value": value,
                })
                trial.report(value, step=epoch)
                return trial.should_prune()

            if self.base_dict["logging"].get("level") != "debug":
                logger.setLevel(logging.WARNING)

            try:
                trial_config = self.config_class.model_validate(sampled)
                if not trial_config.validate_config():
                    raise ValueError(
                        f"Invalid sampled configuration for scope {scope}."
                    )

                def prepare_trial(attempt_trainer: Any) -> None:
                    """Attach this Optuna trial's managed-run context."""
                    attempt_trainer.configure_managed_run(
                        log_dir=trial_root,
                        checkpoint_dir=checkpoint_root,
                        campaign_hash=self.campaign_hash,
                        run_mode="hpo",
                        campaign_aliases=self.campaign_aliases,
                        validation_callback=(
                            validation_callback
                            if get_is_master()
                            else None
                        ),
                        external_distributed=True,
                    )

                result = self._run_adaptive_trainer(
                    trial_config,
                    prepare_trial,
                )
                pruned = bool(result.get("pruned"))
                performance = result.get("performance")
                if not isinstance(performance, Mapping):
                    raise TypeError(
                        "HPO trainer result must include performance timing "
                        "diagnostics."
                    )
                if get_is_master():
                    if best_value is None:
                        raise ValueError(
                            "Trial completed without a validation objective."
                        )
                    trial.set_user_attr("decoded_params", decoded)
                    if pruned:
                        study.tell(
                            trial,
                            state=optuna.trial.TrialState.PRUNED,
                        )
                        self._write_trial_status(
                            trial_root,
                            PRUNED_STATUS,
                            decoded,
                            objective=best_value,
                            objective_history=objective_history,
                            performance=performance,
                        )
                    else:
                        study.tell(trial, best_value)
                        self._write_trial_status(
                            trial_root,
                            COMPLETE_STATUS,
                            decoded,
                            objective=best_value,
                            objective_history=objective_history,
                            performance=performance,
                        )
                trial_budgeted = True
            except Exception as exc:
                if not _is_recoverable_trial_failure(exc):
                    raise
                if get_is_master():
                    fingerprint = failure_fingerprint(exc)
                    memory_information = getattr(
                        self,
                        "last_adaptive_memory_profile",
                        {},
                    )
                    trial.set_user_attr("error_fingerprint", fingerprint)
                    trial.set_user_attr(
                        "memory_information",
                        memory_information,
                    )
                    study.tell(
                        trial,
                        state=optuna.trial.TrialState.FAIL,
                    )
                    self._write_trial_status(
                        trial_root,
                        FAILED_STATUS,
                        decoded,
                        error=fingerprint,
                        objective_history=objective_history,
                        memory_information=memory_information,
                    )
                logger.warning(
                    "HPO trial %d failed: %s",
                    trial_number,
                    exc,
                )
            finally:
                _release_training_state()
                if get_is_master():
                    cleanup_failure = _remove_checkpoint_tree(
                        checkpoint_root,
                        f"HPO trial {trial_number}",
                    )
                    if cleanup_failure is not None:
                        self.checkpoint_cleanup_failures.append(
                            cleanup_failure
                        )
            if trial_budgeted:
                budgeted += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= failure_limit:
                    raise RuntimeError(
                        f"HPO scope '{scope}' reached "
                        f"{consecutive_failures} consecutive failed trials. "
                        "Correct the configuration or search space."
                    )

        self._configure_campaign_logging()
        if get_is_master():
            try:
                best_trial = study.best_trial
            except ValueError as exc:
                raise RuntimeError(
                    f"HPO scope '{scope}' has no completed trial."
                ) from exc
            decoded = best_trial.user_attrs.get("decoded_params")
            if not isinstance(decoded, dict):
                distribution_by_path = self.hpo_config.search_space
                decoded = {}
                for path, encoded in best_trial.params.items():
                    distribution = distribution_by_path[path]
                    if distribution.distribution == "categorical":
                        index = int(str(encoded).split("_")[-1])
                        decoded[path] = copy.deepcopy(
                            distribution.choices[index]
                        )
                    else:
                        decoded[path] = encoded
            best_payload = {
                "study_identity": study_identity,
                "trial_number": best_trial.number,
                "objective": best_trial.value,
                "parameters": decoded,
            }
            selection_payload = {
                **best_payload,
                "source": "hpo",
                "scope": scope,
                "requested_budget": self.hpo_config.n_trials,
                "effective_budget": budgeted,
                "parameter_digest": semantic_digest({
                    "parameters": decoded,
                }),
            }
            best_path = (
                artifact_root
                / f"best_trial_{best_trial.number:05d}.json"
            )
            if best_path.exists():
                _validate_immutable_json(best_path, best_payload)
            else:
                _exclusive_json(best_path, best_payload)
            _exclusive_json(
                self.invocation_root / "hpo" / f"{scope}.json",
                selection_payload,
            )
        else:
            selection_payload = None
            best_payload = None
            best_path = None
        selection_payload = _broadcast_object(selection_payload)
        best_payload = selection_payload
        if not hasattr(self, "selection_provenance"):
            self.selection_provenance = {}
        self.selection_provenance[scope] = dict(
            selection_payload
        )
        if get_is_master():
            logger.info(
                "HPO scope %s selected trial %d with objective %.8g; "
                "best parameters: %s. Best artifact: %s.",
                scope,
                best_payload["trial_number"],
                best_payload["objective"],
                json.dumps(best_payload["parameters"], sort_keys=True),
                best_path.resolve(),
            )

        selected = copy.deepcopy(dict(scope_config))
        for path, value in best_payload["parameters"].items():
            set_dotted_value(selected, path, value)
        validated = self.config_class.model_validate(selected)
        if not validated.validate_config():
            raise RuntimeError(
                f"Selected HPO configuration for '{scope}' is invalid."
            )
        if runtime is not None:
            self._release_study(runtime)
            self._active_study_runtime = None
        return selected

    def _run_hpo(self) -> Dict[str, Dict[str, Any]]:
        """Run all configured studies and return selected scope configs."""
        validate_search_space(
            self.base_dict,
            self.hpo_config,
            self.config_class,
        )
        selected: Dict[str, Dict[str, Any]] = {}
        for scope, scope_config in _study_scope_configs(
            self.base_dict
        ).items():
            try:
                selected[scope] = self._run_hpo_scope(
                    scope,
                    scope_config,
                )
            finally:
                runtime = self._active_study_runtime
                if runtime is not None and get_is_master():
                    self._release_study(runtime)
                    self._active_study_runtime = None
        return selected

    def _fixed_scopes(self) -> Dict[str, Dict[str, Any]]:
        """Return independently lockable scopes when HPO is disabled."""
        if self.base_dict["multitask"]:
            scopes = {"multitask": copy.deepcopy(self.base_dict)}
        else:
            scopes = {}
            for dataset_name, dataset_config in self.base_dict[
                "data"
            ]["datasets"].items():
                scoped = _scoped_config(
                    self.base_dict,
                    seed=0,
                    datasets_config={dataset_name: dataset_config},
                )
                scoped["seeds"] = list(self.base_dict["seeds"])
                scopes[dataset_name] = scoped
        for scope, config in scopes.items():
            self.selection_provenance[scope] = {
                "source": "fixed",
                "scope": scope,
                "parameter_digest": semantic_digest({
                    "configuration": config,
                }),
            }
        return scopes

    def _legacy_completion_provenance(
        self,
        scope: str,
        seed: int,
        completion: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Recover selection provenance from immutable invocation records.

        Parameters
        ----------
        scope : str
            Dataset name or ``multitask``.
        seed : int
            Effective evaluation seed.
        completion : Mapping[str, Any]
            Valid terminal completion lacking new provenance metadata.

        Returns
        -------
        dict[str, Any] or None
            Unique highest-budget matching selection, when recoverable.
        """
        completion_hash = completion.get("config_hash")
        if not isinstance(completion_hash, str):
            return None
        candidates: list[Dict[str, Any]] = []
        invocations_root = self.paths.log_root / "invocations"
        for invocation_root in sorted(invocations_root.glob("*")):
            invocation_path = invocation_root / "invocation.yaml"
            winner_path = invocation_root / "hpo" / f"{scope}.json"
            if not invocation_path.is_file() or not winner_path.is_file():
                continue
            try:
                invocation = yaml.safe_load(
                    invocation_path.read_text(encoding="utf-8")
                )
                winner = json.loads(
                    winner_path.read_text(encoding="utf-8")
                )
            except (OSError, yaml.YAMLError, json.JSONDecodeError):
                continue
            if not isinstance(invocation, dict) or not isinstance(
                winner,
                dict,
            ):
                continue
            model_config = invocation.get("model_config")
            parameters = winner.get("parameters")
            hpo = invocation.get("hpo")
            if (
                not isinstance(model_config, dict)
                or not isinstance(parameters, dict)
                or not isinstance(hpo, dict)
            ):
                continue
            if scope == "multitask":
                selected = _scoped_config(model_config, seed)
            else:
                datasets_config = model_config.get("data", {}).get(
                    "datasets",
                    {},
                )
                dataset_config = datasets_config.get(scope)
                if not isinstance(dataset_config, str):
                    continue
                selected = _scoped_config(
                    model_config,
                    seed,
                    datasets_config={scope: dataset_config},
                )
            try:
                for path, value in parameters.items():
                    set_dotted_value(selected, path, value)
                selected_hash = _config_hash(selected)
            except (KeyError, TypeError, ValueError):
                continue
            if selected_hash != completion_hash:
                continue
            requested_budget = hpo.get("n_trials")
            effective_budget = winner.get(
                "effective_budget",
                requested_budget,
            )
            study_identity = winner.get("study_identity")
            if (
                isinstance(requested_budget, bool)
                or not isinstance(requested_budget, int)
                or requested_budget <= 0
                or isinstance(effective_budget, bool)
                or not isinstance(effective_budget, int)
                or effective_budget <= 0
                or not isinstance(study_identity, str)
                or not study_identity
            ):
                continue
            candidates.append({
                "source": "hpo",
                "scope": scope,
                "study_identity": study_identity,
                "requested_budget": requested_budget,
                "effective_budget": effective_budget,
                "parameter_digest": semantic_digest({
                    "parameters": parameters,
                }),
            })
        if not candidates:
            return None
        maximum = max(
            int(candidate["effective_budget"])
            for candidate in candidates
        )
        highest = [
            candidate for candidate in candidates
            if candidate["effective_budget"] == maximum
        ]
        identities = {
            canonical_json(candidate) for candidate in highest
        }
        if len(identities) != 1:
            logger.warning(
                "Legacy HPO provenance for seed %d scope %s is ambiguous.",
                seed,
                scope,
            )
            return None
        return highest[0]

    def _completion_provenance(
        self,
        scope: str,
        seed: int,
        completion: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Return new or safely recovered completion provenance."""
        provenance = completion.get("selection_provenance")
        if isinstance(provenance, dict):
            return dict(provenance)
        return self._legacy_completion_provenance(
            scope,
            seed,
            completion,
        )

    def _replacement_is_authorized(
        self,
        scope: str,
        seed: int,
        located: list[Any],
    ) -> bool:
        """Return whether larger-budget HPO provenance proves replacement."""
        current = self.selection_provenance.get(scope)
        if not isinstance(current, dict) or current.get("source") != "hpo":
            return False
        current_budget = current.get("effective_budget")
        current_study = current.get("study_identity")
        current_digest = current.get("parameter_digest")
        if (
            isinstance(current_budget, bool)
            or not isinstance(current_budget, int)
            or current_budget <= 0
            or not isinstance(current_study, str)
            or not isinstance(current_digest, str)
        ):
            return False
        conflicts = [
            item for item in located
            if item.compatibility.terminal
            and not item.compatibility.compatible
        ]
        if not conflicts:
            return False
        for item in conflicts:
            completion = item.compatibility.completion
            if not isinstance(completion, dict):
                return False
            previous = self._completion_provenance(
                scope,
                seed,
                completion,
            )
            if previous is None:
                return False
            if (
                previous.get("source") != "hpo"
                or previous.get("study_identity") != current_study
                or previous.get("parameter_digest") == current_digest
            ):
                return False
            previous_budget = previous.get("effective_budget")
            if (
                isinstance(previous_budget, bool)
                or not isinstance(previous_budget, int)
                or current_budget <= previous_budget
            ):
                return False
        return True

    def _start_seed_cloud(self, seed: int) -> Dict[str, Any]:
        """Start one cloud run shared by all dataset runs for a seed."""
        logging_config = self.config.logging
        context: Dict[str, Any] = {}
        if not logging_config.use_cloud or not get_is_master():
            return context

        backend = logging_config.cloud_backend.lower()
        run_id = f"{self.campaign_hash}-seed-{seed}"
        if backend in {"wandb", "both"}:
            import wandb

            wandb.init(
                project=(
                    logging_config.project
                    or self.base_dict["model_type"]
                ),
                entity=logging_config.entity,
                id=run_id,
                name=f"seed_{seed}",
                group=self.campaign_hash,
                resume="allow",
                mode=(
                    "offline"
                    if logging_config.offline
                    else "online"
                ),
                tags=list(logging_config.tags) + [f"seed_{seed}"],
                config=self.base_dict,
                dir=str(self.paths.log_root.resolve()),
            )
            context["wandb"] = True

        if backend in {"comet", "both"}:
            import comet_ml

            api_key = (
                logging_config.api_key
                or os.environ.get("COMET_API_KEY")
            )
            if not api_key:
                warnings.warn(
                    "Comet API key is missing; the per-seed Comet run "
                    "was not created.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                key_path = (
                    self.paths.seed_log_root(seed)
                    / "cloud"
                    / "comet_experiment_key.txt"
                )
                kwargs = {
                    "api_key": api_key,
                    "project_name": (
                        logging_config.project
                        or self.base_dict["model_type"]
                    ),
                }
                if logging_config.entity:
                    kwargs["workspace"] = logging_config.entity
                if key_path.is_file():
                    experiment = comet_ml.ExistingExperiment(
                        previous_experiment=(
                            key_path.read_text(encoding="utf-8").strip()
                        ),
                        **kwargs,
                    )
                else:
                    experiment = comet_ml.Experiment(**kwargs)
                    key_path.parent.mkdir(parents=True, exist_ok=True)
                    key_path.write_text(
                        experiment.get_key(),
                        encoding="utf-8",
                    )
                experiment.set_name(f"seed_{seed}")
                experiment.add_tags(
                    list(logging_config.tags) + [f"seed_{seed}"]
                )
                experiment.log_parameters(self.base_dict)
                context["comet"] = experiment
        return context


    def _finish_seed_cloud(self, context: Mapping[str, Any]) -> None:
        """Finish a campaign-owned per-seed cloud run."""
        if context.get("wandb"):
            import wandb

            wandb.finish()
        comet_experiment = context.get("comet")
        if comet_experiment is not None:
            comet_experiment.end()

    def _run_seed(
        self,
        seed: int,
        selected_configs: Mapping[str, Mapping[str, Any]],
    ) -> bool:
        """Execute incomplete scopes and return whether training was needed."""
        locks = _acquire_execution_locks(
            self.paths,
            seed,
            selected_configs,
        )
        cloud_context: Dict[str, Any] = {}
        try:
            pending: Dict[
                str,
                tuple[
                    Dict[str, Any],
                    list[Any],
                    bool,
                    Optional[Path],
                ],
            ] = {}
            for scope, selected in selected_configs.items():
                scoped = _scoped_config(selected, seed)
                located = [
                    locate_completion(
                        self.paths.log_root,
                        self.campaign_hash,
                        seed,
                        dataset_name,
                        scoped,
                        campaign_aliases=self.campaign_aliases,
                    )
                    for dataset_name in scoped["data"]["datasets"]
                ]
                recovery_checkpoint: Optional[Path] = None
                recovery = {"recovered": False, "error": None}
                if scoped.get("multitask") and get_is_master():
                    try:
                        recovery["recovered"] = (
                            _recover_multitask_from_csv(
                                self.paths,
                                self.campaign_hash,
                                seed,
                                scoped,
                                located,
                                self.selection_provenance.get(scope),
                                self.invocation_id,
                            )
                        )
                    except ValueError as exc:
                        recovery["error"] = str(exc)
                recovery = _broadcast_object(recovery)
                if recovery["error"] is not None:
                    retained = {
                        Path(value).resolve()
                        for item in located
                        if item.compatibility.compatible
                        and isinstance(
                            item.compatibility.completion,
                            dict,
                        )
                        for value in [
                            item.compatibility.completion.get(
                                "checkpoint_path"
                            )
                        ]
                        if isinstance(value, str)
                        and Path(value).is_file()
                    }
                    if len(retained) != 1:
                        raise RuntimeError(
                            "Multitask completion recovery failed: "
                            f"{recovery['error']} Expected one compatible "
                            "retained shared checkpoint, but found "
                            f"{sorted(str(path) for path in retained)}."
                        )
                    recovery_checkpoint = next(iter(retained))
                    logger.warning(
                        "CSV recovery was unavailable; evaluating only "
                        "missing multitask datasets from %s.",
                        recovery_checkpoint,
                    )
                if recovery["recovered"]:
                    located = [
                        locate_completion(
                            self.paths.log_root,
                            self.campaign_hash,
                            seed,
                            dataset_name,
                            scoped,
                            campaign_aliases=self.campaign_aliases,
                        )
                        for dataset_name in scoped["data"]["datasets"]
                    ]
                if recovery_checkpoint is not None:
                    pending[scope] = (
                        scoped,
                        located,
                        False,
                        recovery_checkpoint,
                    )
                    continue
                replace = self._replacement_is_authorized(
                    scope,
                    seed,
                    located,
                )
                _, _, complete = _scope_artifact_roots(
                    self.paths,
                    self.campaign_hash,
                    seed,
                    scoped,
                    campaign_aliases=self.campaign_aliases,
                    allow_replacement=replace,
                )
                if not complete:
                    pending[scope] = (
                        scoped,
                        located,
                        replace,
                        None,
                    )
                    continue
                failures = _maintain_completed_scope(
                    self.paths,
                    seed,
                    scoped,
                    located,
                )
                self.checkpoint_cleanup_failures.extend(failures)
            if not pending:
                return False
            cloud_context = self._start_seed_cloud(seed)
            for scope, pending_scope in pending.items():
                trainer = None
                archive: Optional[ReplacementArchive] = None
                scoped, located, replace, recovery_checkpoint = pending_scope
                try:
                    archive_error: Optional[str] = None
                    if replace:
                        try:
                            archive = _archive_completed_scope(
                                self.paths,
                                self.invocation_root,
                                seed,
                                scope,
                                scoped,
                                located,
                            )
                        except Exception as exc:
                            archive_error = str(exc)
                    archive_error = _broadcast_object(archive_error)
                    if archive_error is not None:
                        raise RuntimeError(
                            "Completed-result archival failed: "
                            f"{archive_error}"
                        )
                    seed_log_root = self.paths.seed_log_root(seed)
                    seed_checkpoint_root = (
                        self.paths.seed_checkpoint_root(seed)
                    )
                    logger.debug(
                        "Seed %d selected artifact root %s.",
                        seed,
                        seed_log_root.resolve(),
                    )
                    final_config = self.config_class.model_validate(scoped)
                    if not final_config.validate_config():
                        raise ValueError(
                            "Invalid final configuration for seed "
                            f"{seed}."
                        )
                    for item in located:
                        if not item.compatibility.compatible:
                            self.final_dataset_attempts.append({
                                "seed": seed,
                                "dataset": item.path.parent.name,
                            })
                    is_neural = (
                        self.base_dict["model_type"]
                        not in DETERMINISTIC_MODELS
                    )
                    if is_neural:
                        def prepare_seed(
                            attempt_trainer: Any,
                        ) -> None:
                            """Attach this seed's managed-run context."""
                            attempt_trainer.comet_experiment = (
                                cloud_context.get("comet")
                            )
                            attempt_trainer.configure_managed_run(
                                log_dir=seed_log_root,
                                checkpoint_dir=seed_checkpoint_root,
                                campaign_hash=self.campaign_hash,
                                run_mode=(
                                    "recovery" if recovery_checkpoint
                                    else "final"
                                ),
                                campaign_aliases=self.campaign_aliases,
                                selection_provenance=(
                                    self.selection_provenance.get(scope)
                                ),
                                invocation_id=self.invocation_id,
                                external_distributed=True,
                                external_cloud=bool(cloud_context),
                            )

                        if recovery_checkpoint is None:
                            result = self._run_adaptive_trainer(
                                final_config,
                                prepare_seed,
                            )
                        else:
                            batching_profiles = [
                                item.compatibility.completion.get(
                                    "batching"
                                )
                                for item in located
                                if item.compatibility.compatible
                                and isinstance(
                                    item.compatibility.completion,
                                    dict,
                                )
                            ]
                            serialized_profiles = {
                                canonical_json(profile)
                                for profile in batching_profiles
                                if isinstance(profile, dict)
                            }
                            if len(serialized_profiles) != 1:
                                raise RuntimeError(
                                    "Compatible multitask completions do "
                                    "not share one runtime batching profile."
                                )
                            profile = json.loads(
                                next(iter(serialized_profiles))
                            )
                            requested_batch = int(
                                scoped["data"]["batch_size"]
                            )
                            world_size = get_world_size()
                            profile_world_size = profile.get("world_size")
                            profile_global_batch = profile.get(
                                "requested_global_batch_size"
                            )
                            micro_batch = profile.get(
                                "micro_batch_size"
                            )
                            if (
                                profile_world_size != world_size
                                or profile_global_batch != requested_batch
                                or isinstance(micro_batch, bool)
                                or not isinstance(micro_batch, int)
                                or micro_batch <= 0
                                or requested_batch
                                % (world_size * micro_batch) != 0
                            ):
                                raise RuntimeError(
                                    "Retained shared checkpoint batching is "
                                    "incompatible with this invocation: "
                                    f"{profile}."
                                )
                            trainer = ModelRegistry.create_trainer(
                                final_config
                            )
                            prepare_seed(trainer)
                            trainer.configure_runtime_batching(
                                requested_batch,
                                micro_batch,
                                world_size,
                            )
                            missing = [
                                item.path.parent.name
                                for item in located
                                if not item.compatibility.compatible
                            ]
                            result = trainer.recover_multitask_datasets(
                                missing,
                                recovery_checkpoint,
                            )
                        cleanup_failures = result.get(
                            "checkpoint_cleanup_failures",
                            [],
                        )
                        self.checkpoint_cleanup_failures.extend(
                            cleanup_failures
                        )
                    else:
                        trainer = ModelRegistry.create_trainer(final_config)
                        trainer.comet_experiment = cloud_context.get("comet")
                        trainer.log_dir_override = seed_log_root.resolve()
                        trainer.ckpt_dir_override = (
                            seed_checkpoint_root.resolve()
                        )
                        trainer.selection_provenance = (
                            self.selection_provenance.get(scope)
                        )
                        trainer.campaign_invocation_id = self.invocation_id
                        trainer.campaign_hash = self.campaign_hash
                        trainer.campaign_aliases = self.campaign_aliases
                        trainer.external_cloud = bool(cloud_context)
                        trainer.run()
                except BaseException:
                    _restore_completed_scope(archive)
                    raise
                finally:
                    trainer = None
                    _release_training_state()
        finally:
            if get_is_master() and cloud_context:
                self._finish_seed_cloud(cloud_context)
            _release_execution_locks(locks)
        return True

    def _run_seeds(
        self,
        selected_configs: Mapping[str, Mapping[str, Any]],
    ) -> tuple[Dict[str, Any], bool]:
        """Run configured seeds with consecutive-error stopping."""
        self.final_dataset_attempts = []
        seeds = list(self.base_dict["seeds"])
        attempted: list[int] = []
        succeeded: list[int] = []
        failed: list[Dict[str, Any]] = []
        skipped: list[int] = []
        unattempted: list[int] = []
        previous_failure: Optional[str] = None
        first_attempt_succeeded: Optional[bool] = None

        for index, seed in enumerate(seeds):
            try:
                did_run = self._run_seed(seed, selected_configs)
            except Exception as exc:
                attempted.append(seed)
                fingerprint = failure_fingerprint(exc)
                failed.append({
                    "seed": seed,
                    "fingerprint": fingerprint,
                })
                if first_attempt_succeeded is None:
                    first_attempt_succeeded = False
                repeated = fingerprint == previous_failure
                previous_failure = fingerprint
                logger.exception(
                    "Seed %d failed with fingerprint %s",
                    seed,
                    fingerprint,
                )
                if repeated:
                    unattempted.extend(seeds[index + 1:])
                    break
            else:
                if did_run is False:
                    skipped.append(seed)
                    logger.info(
                        "Seed %d is already complete and compatible; "
                        "skipping.",
                        seed,
                    )
                    continue
                attempted.append(seed)
                succeeded.append(seed)
                logger.info(
                    "Seed %d completed; artifacts: %s.",
                    seed,
                    self.paths.seed_log_root(seed).resolve(),
                )
                if first_attempt_succeeded is None:
                    first_attempt_succeeded = True
                previous_failure = None

        invocation = {
            "attempted": attempted,
            "dataset_attempts": list(self.final_dataset_attempts),
            "id": getattr(self, "invocation_id", "unmanaged"),
            "requested": seeds,
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "unattempted": unattempted,
            "complete": not failed,
        }
        if not failed:
            summary_eligible = True
        else:
            summary_eligible = bool(
                first_attempt_succeeded and succeeded
            )
        return invocation, summary_eligible

    def _log_invocation_summary(
        self,
        summary: CampaignSummaryResult,
    ) -> None:
        """Log final metric summaries produced by this invocation only."""
        pair_status = summary.status["dataset_pairs"]
        logger.info(
            "Invocation final dataset pairs: attempted=%d, completed=%d, "
            "incomplete=%d, rejected=%d.",
            pair_status["attempted"],
            pair_status["completed"],
            pair_status["incomplete"],
            pair_status["rejected"],
        )
        for row in summary.test_summary:
            message = (
                "%s test %s: count=%d, mean=%.6g, median=%.6g"
            )
            arguments: list[Any] = [
                row["dataset"],
                row["metric"],
                row["count"],
                row["mean"],
                row["median"],
            ]
            if "std" in row:
                message += ", std=%.6g"
                arguments.append(row["std"])
            logger.info(message, *arguments)


    def _update_cloud_summary(
        self,
        summary: CampaignSummaryResult,
    ) -> None:
        """Update one stable cloud summary with current-invocation metrics."""
        logging_config = self.config.logging
        if not logging_config.use_cloud or not get_is_master():
            return

        metrics: Dict[str, float] = {}
        for row in summary.test_summary:
            prefix = f"{row['dataset']}/test/{row['metric']}"
            for statistic in ("mean", "median", "std"):
                if statistic in row:
                    metrics[f"{prefix}/{statistic}"] = row[statistic]
        metrics["summary/completed_seed_count"] = len({
            row["seed"]
            for row in summary.test_runs
        })

        backend = logging_config.cloud_backend.lower()
        run_id = f"{self.campaign_hash}-summary"
        if backend in {"wandb", "both"}:
            import wandb

            run = wandb.init(
                project=(
                    logging_config.project
                    or self.base_dict["model_type"]
                ),
                entity=logging_config.entity,
                id=run_id,
                name=run_id,
                group=self.campaign_hash,
                resume="allow",
                mode=(
                    "offline"
                    if logging_config.offline
                    else "online"
                ),
                tags=list(logging_config.tags) + ["summary"],
            )
            run.log(metrics)
            run.summary["invocation"] = summary.status[
                "latest_invocation"
            ]
            run.finish()

        if backend in {"comet", "both"}:
            import comet_ml

            api_key = (
                logging_config.api_key
                or os.environ.get("COMET_API_KEY")
            )
            if not api_key:
                warnings.warn(
                    "Comet API key is missing; the stable summary run "
                    "was not updated.",
                    UserWarning,
                    stacklevel=2,
                )
                return
            key_path = (
                self.paths.summary_root
                / "comet_experiment_key.txt"
            )
            kwargs = {
                "api_key": api_key,
                "project_name": (
                    logging_config.project
                    or self.base_dict["model_type"]
                ),
            }
            if logging_config.entity:
                kwargs["workspace"] = logging_config.entity
            if key_path.is_file():
                experiment = comet_ml.ExistingExperiment(
                    previous_experiment=(
                        key_path.read_text(encoding="utf-8").strip()
                    ),
                    **kwargs,
                )
            else:
                experiment = comet_ml.Experiment(**kwargs)
                key_path.parent.mkdir(parents=True, exist_ok=True)
                key_path.write_text(
                    experiment.get_key(),
                    encoding="utf-8",
                )
            experiment.set_name(run_id)
            experiment.add_tags(
                list(logging_config.tags) + ["summary"]
            )
            experiment.log_metrics(metrics)
            experiment.log_other(
                "invocation",
                json.dumps(
                    summary.status["latest_invocation"],
                    sort_keys=True,
                ),
            )
            experiment.end()

    def _audit_hpo_scope(
        self,
        scope: str,
        scope_config: Mapping[str, Any],
    ) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Audit one HPO scope without opening writable Optuna storage.

        Parameters
        ----------
        scope : str
            Dataset name or ``multitask``.
        scope_config : Mapping[str, Any]
            Base configuration before applying the persisted winner.

        Returns
        -------
        tuple[dict[str, Any], dict[str, Any] or None]
            Study-selection report and reconstructed selected configuration.
        """
        scope_root = _hpo_scope_root(self.paths.log_root, scope)
        study_identity, _ = self._study_identity(scope)
        storage_paths = [scope_root / "study.sqlite3"]
        studies_root = scope_root / "studies"
        if studies_root.is_dir():
            storage_paths.extend(
                sorted(studies_root.glob("*/study.sqlite3"))
            )
        records: list[Dict[str, Any]] = []
        for storage_path in storage_paths:
            records.extend(_read_sqlite_studies(storage_path))
        expected_names = {
            study_identity,
            *(f"{alias}-{scope}" for alias in self.campaign_aliases),
        }
        expected_distributions = _serialized_trial_distributions(
            self.hpo_config
        )
        compatible = [
            record for record in records
            if (
                record["study_name"] == study_identity
                or record["study_name"].endswith(f"-{scope}")
            )
            and str(record["direction"]).lower()
            == self.hpo_config.objective.direction
            and record["distributions_consistent"]
            and record["distributions"]
            == expected_distributions
        ]
        complete = [
            record for record in compatible
            if record["complete"] + record["pruned"]
            >= self.hpo_config.n_trials
        ]
        matching_partial = [
            record for record in compatible
            if record["study_name"] in expected_names
        ]
        if complete:
            eligible = complete
        elif matching_partial:
            eligible = matching_partial
        else:
            eligible = compatible
        report: Dict[str, Any] = {
            "scope": scope,
            "study_identity": study_identity,
            "studies": records,
            "duplicates": [],
        }
        if len(eligible) > 1:
            report["status"] = "ambiguous"
            report["duplicates"] = [
                record["study_name"] for record in eligible
            ]
            return report, None
        if not eligible:
            report["status"] = "missing"
            return report, None

        selected_record = eligible[0]
        report["status"] = "selected"
        report["selected"] = selected_record
        report["duplicates"] = [
            record["study_name"]
            for record in records
            if record is not selected_record
        ]
        storage_path = Path(selected_record["storage_path"])
        legacy_path = (scope_root / "study.sqlite3").resolve()
        if storage_path == legacy_path:
            best_path = scope_root / "best.json"
        else:
            trial_number = selected_record["best_trial"]
            best_path = (
                storage_path.parent
                / f"best_trial_{trial_number:05d}.json"
            )
        report["best_path"] = str(best_path.resolve())
        if not best_path.is_file():
            report["status"] = "winner_artifact_missing"
            return report, None
        try:
            best_payload = json.loads(
                best_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            report["status"] = "winner_artifact_invalid"
            report["error"] = str(exc)
            return report, None
        if best_payload.get("trial_number") != selected_record["best_trial"]:
            report["status"] = "winner_artifact_mismatch"
            return report, None
        parameters = best_payload.get("parameters")
        if not isinstance(parameters, dict):
            report["status"] = "winner_parameters_invalid"
            return report, None
        selected = copy.deepcopy(dict(scope_config))
        for path, value in parameters.items():
            set_dotted_value(selected, path, value)
        report["parameters"] = parameters
        return report, selected

    def audit(self) -> Dict[str, Any]:
        """Return a read-only campaign, study, and completion audit.

        Returns
        -------
        dict[str, Any]
            Absolute roots, semantic identity, study selection, duplicates,
            and pair-level completion compatibility.
        """
        hpo_enabled = (
            self.hpo_config.enabled
            and self.base_dict["model_type"] not in DETERMINISTIC_MODELS
        )
        studies: Dict[str, Any] = {}
        selected: Dict[str, Dict[str, Any]] = {}
        if hpo_enabled:
            validate_search_space(
                self.base_dict,
                self.hpo_config,
                self.config_class,
            )
            for scope, scope_config in _study_scope_configs(
                self.base_dict
            ).items():
                study_report, selected_config = self._audit_hpo_scope(
                    scope,
                    scope_config,
                )
                studies[scope] = study_report
                if selected_config is not None:
                    selected[scope] = selected_config
        else:
            selected = self._fixed_scopes()

        completions: Dict[str, list[Dict[str, Any]]] = {
            "accepted": [],
            "missing": [],
            "rejected": [],
        }
        discovered_seeds = set(self.base_dict["seeds"])
        logs_root = self.paths.log_root / "logs"
        if logs_root.is_dir():
            for seed_root in logs_root.glob("seed_*"):
                try:
                    discovered_seeds.add(
                        int(seed_root.name.removeprefix("seed_"))
                    )
                except ValueError:
                    continue
        for seed in sorted(discovered_seeds):
            if not selected:
                break
            for dataset_name, config in _expected_configs(
                selected,
                seed,
            ).items():
                located = locate_completion(
                    self.paths.log_root,
                    self.campaign_hash,
                    seed,
                    dataset_name,
                    config,
                    campaign_aliases=self.campaign_aliases,
                )
                result = located.compatibility
                key = "accepted" if result.compatible else result.mode
                completions[key].append({
                    "seed": seed,
                    "dataset": dataset_name,
                    "path": str(located.path.resolve()),
                    "mode": result.mode,
                    "reason": result.reason,
                })
        return {
            "identity_version": IDENTITY_VERSION,
            "campaign_identity": self.campaign_hash,
            "campaign_aliases": sorted(self.campaign_aliases),
            "legacy_root": self.resolution.legacy,
            "campaign_log_root": str(self.paths.log_root.resolve()),
            "campaign_checkpoint_root": str(
                self.paths.checkpoint_root.resolve()
            ),
            "studies": studies,
            "completions": completions,
            "ignored_legacy_namespaces": [
                str(path)
                for path in _ignored_configuration_namespaces(
                    self.paths.log_root
                )
            ],
        }

    def run(self) -> Dict[str, Any]:
        """Run HPO if enabled, then execute independent configured seeds."""
        self._configure_campaign_logging()
        self._save_campaign_config()
        paths = getattr(self, "paths", None)
        if paths is not None:
            _warn_ignored_configuration_namespaces(paths.log_root)
        hpo_enabled = self.hpo_config.enabled
        if (
            hpo_enabled
            and self.base_dict["model_type"] in DETERMINISTIC_MODELS
        ):
            warnings.warn(
                f"{self.base_dict['model_type']} is deterministic; ignoring "
                "the configured HPO section.",
                UserWarning,
                stacklevel=2,
            )
            hpo_enabled = False

        initialization_seed = (
            self.hpo_config.seed
            if hpo_enabled
            else self.base_dict["seeds"][0]
        )
        invocation: Optional[Dict[str, Any]] = None
        status_recorded = False
        try:
            self._initialize_distributed(initialization_seed)
            selected = (
                self._run_hpo()
                if hpo_enabled
                else self._fixed_scopes()
            )
            invocation, _ = self._run_seeds(selected)
            self._configure_campaign_logging()
            summary_result: Optional[CampaignSummaryResult] = None
            invocation_summary: Optional[CampaignSummaryResult] = None
            if get_is_master() and hasattr(self, "paths"):
                summary_result = write_campaign_summary(
                    self.paths,
                    self.campaign_hash,
                    invocation,
                    campaign_aliases=self.campaign_aliases,
                )
                invocation_summary = build_invocation_summary(
                    self.paths,
                    self.campaign_hash,
                    invocation,
                    campaign_aliases=self.campaign_aliases,
                )
                self._log_invocation_summary(invocation_summary)
                if invocation_summary.written:
                    self._update_cloud_summary(invocation_summary)

            if invocation["failed"]:
                state = (
                    "partial"
                    if invocation["succeeded"]
                    else "failed"
                )
            else:
                state = "complete"
            pair_status = (
                invocation_summary.status["dataset_pairs"]
                if invocation_summary is not None
                else None
            )
            self._update_invocation_status(
                state,
                invocation=invocation,
                dataset_pairs=pair_status,
            )
            status_recorded = True
            if invocation["failed"]:
                raise CampaignExecutionError(
                    "One or more seed executions failed. See campaign "
                    f"artifacts at {self.paths.log_root.resolve()}."
                )
            return {
                "campaign_hash": self.campaign_hash,
                "selected_configs": selected,
                "invocation": invocation,
                "summary": (
                    dict(summary_result.status)
                    if summary_result is not None
                    else None
                ),
            }
        except Exception as exc:
            if not status_recorded:
                self._update_invocation_status(
                    "failed",
                    invocation=invocation,
                    error=failure_fingerprint(exc),
                )
            raise
        finally:
            if self.distributed_initialized:
                clean_torch_distributed()
