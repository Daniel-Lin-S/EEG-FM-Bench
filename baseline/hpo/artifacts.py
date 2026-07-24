"""Campaign paths, failure fingerprints, and multi-seed summaries.

Inputs are resolved campaign configurations and seed-level completion JSON
files. Outputs are atomic JSON/CSV summaries under the campaign log root.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml

from baseline.utils.run_artifacts import get_config_hash


CONFIG_HASH_LENGTH = 12
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
        """Return the current-style artifact root for one seed."""
        return self.log_root / "logs" / f"seed_{seed}"

    def seed_checkpoint_root(self, seed: int) -> Path:
        """Return the checkpoint root for one seed."""
        return self.checkpoint_root / f"seed_{seed}"

    @property
    def summary_root(self) -> Path:
        """Return the campaign summary directory."""
        return self.log_root / "summary"


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
    """

    compatible: bool
    mode: str
    reason: str
    completion: Optional[Mapping[str, Any]] = None


def get_campaign_hash(
    config: Mapping[str, Any],
    hpo: Optional[Mapping[str, Any]],
) -> str:
    """Return a stable campaign identity excluding seeds and HPO budget."""
    identity = json.loads(json.dumps(config, sort_keys=True))
    identity.pop("seeds", None)
    logging_config = identity.get("logging")
    if isinstance(logging_config, dict):
        logging_config.pop("run_dir", None)
        logging_config.pop("level", None)

    hpo_identity: Dict[str, Any] = {}
    if hpo and hpo.get("enabled"):
        hpo_identity = json.loads(json.dumps(hpo, sort_keys=True))
        hpo_identity.pop("n_trials", None)
        hpo_identity.pop("max_consecutive_failed_trials", None)
    identity["hpo"] = hpo_identity
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[
        :CONFIG_HASH_LENGTH
    ]


def build_campaign_paths(
    run_dir: str,
    model_type: str,
    experiment_name: str,
    campaign_hash: str,
) -> CampaignPaths:
    """Build absolute log and checkpoint campaign roots."""
    root = Path(run_dir).resolve()
    name = f"{experiment_name}-{campaign_hash}"
    log_root = root / "log" / "baseline" / model_type / name
    checkpoint_root = root / "ckpt" / "baseline" / model_type / name
    return CampaignPaths(log_root=log_root, checkpoint_root=checkpoint_root)


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


def _normalized_identity(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe config without operational verbosity."""
    normalized = json.loads(json.dumps(config, sort_keys=True))
    logging_config = normalized.get("logging")
    if isinstance(logging_config, dict):
        logging_config.pop("level", None)
    return normalized


def _config_identity_hash(config: Mapping[str, Any]) -> str:
    """Return the resolved identity hash for one configuration."""
    return get_config_hash(config, bool(config.get("multitask")))


def _completion_artifact_error(
    path: Path,
    completion: Mapping[str, Any],
    campaign_hash: str,
    seed: int,
) -> Optional[str]:
    """Return a validation error for common completion artifacts."""
    if completion.get("status") != "completed":
        return "completion status is not 'completed'"
    if completion.get("campaign_hash") != campaign_hash:
        return "campaign hash does not match"
    if completion.get("seed") != seed:
        return "seed does not match"

    metrics = completion.get("test_metrics")
    if not isinstance(metrics, dict):
        return "test_metrics is not a mapping"
    numeric_count = 0
    for metric_key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric_count += 1
        if not math.isfinite(float(value)):
            return (
                f"test metric '{metric_key}' is not finite at "
                f"{path.resolve()}"
            )
    if numeric_count == 0:
        return "test_metrics contains no numeric values"

    checkpoint = completion.get("checkpoint_path")
    if completion.get("has_checkpoint") is not False:
        if not isinstance(checkpoint, str) or not Path(checkpoint).is_file():
            return "checkpoint_path is missing or does not exist"
    return None


def _compatibility_failure(
    reason: str,
    completion: Optional[Mapping[str, Any]] = None,
) -> CompletionCompatibility:
    """Return one standardized rejected compatibility result."""
    return CompletionCompatibility(
        compatible=False,
        mode="rejected",
        reason=reason,
        completion=completion,
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
        saved_hash = _config_identity_hash(saved_config)
    except (KeyError, TypeError, ValueError) as exc:
        return _compatibility_failure(
            f"resolved configuration cannot be hashed: {exc}",
            completion,
        )
    if completion.get("config_hash") != saved_hash:
        return _compatibility_failure(
            "stored config hash does not match the resolved configuration",
            completion,
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

    normalized_saved = _normalized_identity(saved_config)
    normalized_expected = _normalized_identity(expected_config)
    normalized_saved["data"]["batch_size"] = requested_batch
    if normalized_saved != normalized_expected:
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

    artifact_error = _completion_artifact_error(
        path,
        completion,
        campaign_hash,
        seed,
    )
    if artifact_error is not None:
        return _compatibility_failure(artifact_error, completion)

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
            )

    if completion.get("config_hash") == expected_hash:
        return CompletionCompatibility(
            compatible=True,
            mode="exact_canonical_hash",
            reason="canonical configuration hash matches",
            completion=completion,
        )
    if isinstance(expected_config, str):
        return _compatibility_failure(
            "configuration hash does not match",
            completion,
        )
    return _legacy_runtime_batch_compatibility(
        path,
        completion,
        expected_config,
    )


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
        completion_path = (
            campaign_root
            / "logs"
            / f"seed_{seed}"
            / "datasets"
            / dataset_name
            / "completion.json"
        )
        result = check_completion_compatibility(
            completion_path,
            campaign_hash,
            seed,
            expected,
        )
        diagnostic = {
            "seed": seed,
            "dataset": dataset_name,
            "path": str(completion_path.resolve()),
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
) -> list[Dict[str, Any]]:
    """Collect compatible numeric test metrics from completed seed runs."""
    if compatible_config_hashes is None:
        compatible_config_hashes = {}
        logs_root = campaign_root / "logs"
        for completion_path in sorted(
            logs_root.glob("seed_*/datasets/*/completion.json")
        ):
            match = SEED_DIRECTORY_PATTERN.match(
                completion_path.parents[2].name
            )
            if match is None:
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
            compatible_config_hashes[
                (int(match.group(1)), completion_path.parent.name)
            ] = config_hash
    rows, _ = collect_test_rows_with_diagnostics(
        campaign_root,
        campaign_hash,
        compatible_config_hashes,
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


def write_campaign_summary(
    paths: CampaignPaths,
    campaign_hash: str,
    invocation: Mapping[str, Any],
    compatible_configs: Mapping[
        tuple[int, str],
        Mapping[str, Any] | str,
    ],
) -> Optional[Dict[str, Any]]:
    """Regenerate summaries, preserving old summaries when none match."""
    test_rows, diagnostics = collect_test_rows_with_diagnostics(
        paths.log_root,
        campaign_hash,
        compatible_configs,
    )
    if not test_rows:
        report_path = (
            paths.summary_root / "compatibility_report.json"
        )
        report = {
            "campaign_hash": campaign_hash,
            "invocation": dict(invocation),
            "compatibility": diagnostics,
            "status": "no_compatible_results",
        }
        _atomic_write_text(
            report_path,
            json.dumps(report, indent=2, sort_keys=True),
        )
        warnings.warn(
            "No compatible completed seed results were found. "
            "Previous summaries were left unchanged; see "
            f"{report_path.resolve()}.",
            UserWarning,
            stacklevel=2,
        )
        return None

    summary_rows = summarize_test_rows(test_rows)
    include_std = any("std" in row for row in summary_rows)
    summary_fields = [
        "dataset",
        "metric",
        "count",
        "mean",
        "median",
    ]
    if include_std:
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
    partial = (
        not bool(invocation.get("complete", True))
        or bool(diagnostics["missing"] or diagnostics["rejected"])
    )
    payload = {
        "campaign_hash": campaign_hash,
        "invocation": dict(invocation),
        "status": "partial" if partial else "complete",
        "compatibility": diagnostics,
        "test_runs": test_rows,
        "test_summary": summary_rows,
    }
    _atomic_write_text(
        paths.summary_root / "summary.json",
        json.dumps(payload, indent=2, sort_keys=True),
    )
    return payload
