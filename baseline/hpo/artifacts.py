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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


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


def collect_test_rows(
    campaign_root: Path,
    campaign_hash: str,
    compatible_config_hashes: Optional[
        Mapping[tuple[int, str], str]
    ] = None,
) -> list[Dict[str, Any]]:
    """Collect compatible numeric test metrics from completed seed runs."""
    logs_root = campaign_root / "logs"
    rows: list[Dict[str, Any]] = []
    if not logs_root.is_dir():
        return rows

    for seed_root in sorted(logs_root.iterdir()):
        match = SEED_DIRECTORY_PATTERN.match(seed_root.name)
        if match is None or not seed_root.is_dir():
            continue
        seed = int(match.group(1))
        datasets_root = seed_root / "datasets"
        if not datasets_root.is_dir():
            continue

        if compatible_config_hashes is None:
            completion_paths = sorted(
                datasets_root.glob("*/completion.json")
            )
        else:
            expected = {
                dataset_name: config_hash
                for (expected_seed, dataset_name), config_hash
                in compatible_config_hashes.items()
                if expected_seed == seed
            }
            if not expected:
                continue
            completion_paths = [
                datasets_root / dataset_name / "completion.json"
                for dataset_name in sorted(expected)
            ]
            seed_is_complete = all(
                path.is_file()
                and _completion_matches(
                    path,
                    campaign_hash,
                    seed,
                    expected[path.parent.name],
                )
                for path in completion_paths
            )
            if not seed_is_complete:
                continue

        for completion_path in completion_paths:
            completion = json.loads(
                completion_path.read_text(encoding="utf-8")
            )
            if (
                completion.get("status") != "completed"
                or completion.get("campaign_hash") != campaign_hash
                or completion.get("seed") != seed
            ):
                continue
            dataset_name = completion_path.parent.name
            metrics = completion.get("test_metrics")
            if not isinstance(metrics, dict):
                raise ValueError(
                    f"Expected test_metrics mapping at {completion_path}."
                )
            for metric_key, value in sorted(metrics.items()):
                if isinstance(value, bool) or not isinstance(
                    value,
                    (int, float),
                ):
                    continue
                if not math.isfinite(float(value)):
                    raise ValueError(
                        f"Test metric '{metric_key}' is not finite at "
                        f"{completion_path.resolve()}: {value}."
                    )
                rows.append({
                    "dataset": dataset_name,
                    "seed": seed,
                    "metric": metric_key.rsplit("/", maxsplit=1)[-1],
                    "value": float(value),
                })
    return rows


def _completion_matches(
    path: Path,
    campaign_hash: str,
    seed: int,
    config_hash: str,
) -> bool:
    """Return whether one completion belongs to the selected seed config."""
    try:
        completion = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        completion.get("status") == "completed"
        and completion.get("campaign_hash") == campaign_hash
        and completion.get("seed") == seed
        and completion.get("config_hash") == config_hash
    )


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
    compatible_config_hashes: Optional[
        Mapping[tuple[int, str], str]
    ] = None,
) -> Dict[str, Any]:
    """Regenerate local test summaries from all compatible completed seeds."""
    test_rows = collect_test_rows(
        paths.log_root,
        campaign_hash,
        compatible_config_hashes,
    )
    if not test_rows:
        raise ValueError(
            f"No compatible completed seed results exist under "
            f"{paths.log_root.resolve()}."
        )
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
    payload = {
        "campaign_hash": campaign_hash,
        "invocation": dict(invocation),
        "test_runs": test_rows,
        "test_summary": summary_rows,
    }
    _atomic_write_text(
        paths.summary_root / "summary.json",
        json.dumps(payload, indent=2, sort_keys=True),
    )
    return payload
