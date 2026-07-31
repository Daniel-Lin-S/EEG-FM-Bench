"""Write one-seed summary artifacts for deterministic feature extractors.

Inputs are completed per-dataset ``completion.json`` files in a flat
feature-extractor artifact directory. Outputs are campaign-compatible summary
tables under ``summary/`` without neural training coordinates.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from baseline.hpo.artifacts import (
    summarize_completion_diagnostics,
    summarize_test_rows,
    validate_completion_diagnostics,
)
from baseline.utils.identity import IDENTITY_VERSION


FORBIDDEN_TEST_METRICS = frozenset({"epoch", "loss"})
TEST_RUN_FIELDS = ["dataset", "seed", "metric", "value"]
TEST_SUMMARY_FIELDS = ["dataset", "metric", "count", "mean", "median"]


def write_feature_extractor_summary(
    log_dir: Path,
    model_type: str,
    seed: int,
    datasets: Mapping[str, str],
    config_identity: str,
) -> None:
    """Write campaign-compatible test summaries from flat completions.

    Parameters
    ----------
    log_dir : pathlib.Path
        Feature-extractor artifact root containing ``datasets/<dataset>``.
    model_type : str
        Deterministic extractor identifier expected in every completion.
    seed : int
        Sole deterministic evaluation seed.
    datasets : Mapping[str, str]
        Dataset names mapped to their configured dataset variants.
    config_identity : str
        Semantic identity of the requested one-seed extractor configuration.

    Raises
    ------
    ValueError
        If a completion is missing, inconsistent, empty, non-numeric, or
        contains a non-finite or neural-only test metric.
    """
    rows, completions = _collect_test_rows(
        log_dir,
        model_type,
        seed,
        datasets,
    )
    summary_rows = summarize_test_rows(rows)
    if any("std" in row for row in summary_rows):
        raise ValueError(
            "Feature-extractor summaries require exactly one value per "
            "dataset and metric."
        )
    summary_dir = log_dir / "summary"
    _write_csv(summary_dir / "test_runs.csv", TEST_RUN_FIELDS, rows)
    _write_csv(
        summary_dir / "test_summary.csv",
        TEST_SUMMARY_FIELDS,
        summary_rows,
    )
    status = {
        "campaign_identity_version": IDENTITY_VERSION,
        "campaign_identity": config_identity,
        "status": "complete",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "latest_invocation": {
            "requested": [seed],
            "succeeded": [seed],
        },
        "dataset_pairs": {
            "expected": len(datasets),
            "completed": len(datasets),
            "missing": 0,
            "rejected": 0,
        },
    }
    diagnostics = summarize_completion_diagnostics(completions)
    if diagnostics:
        status["diagnostics"] = diagnostics
    _write_json(summary_dir / "summary.json", status)


def _collect_test_rows(
    log_dir: Path,
    model_type: str,
    seed: int,
    datasets: Mapping[str, str],
) -> tuple[
    list[dict[str, Any]],
    list[tuple[int, str, Mapping[str, Any]]],
]:
    """Collect validated test-metric rows from every dataset completion."""
    rows: list[dict[str, Any]] = []
    completions: list[tuple[int, str, Mapping[str, Any]]] = []
    for dataset_name, dataset_config in datasets.items():
        completion_path = (
            log_dir / "datasets" / dataset_name / "completion.json"
        )
        completion = _read_completion(completion_path)
        _validate_completion(
            completion,
            completion_path,
            dataset_name,
            dataset_config,
            model_type,
        )
        completions.append((seed, dataset_name, completion))
        metrics = completion["test_metrics"]
        rows.extend(
            _metric_rows(metrics, dataset_name, seed, completion_path)
        )
    return rows, completions


def _read_completion(completion_path: Path) -> Mapping[str, Any]:
    """Read one completion mapping from disk."""
    if not completion_path.is_file():
        raise ValueError(
            f"Feature-extractor completion does not exist: "
            f"{completion_path.resolve()}."
        )
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Feature-extractor completion is invalid JSON at "
            f"{completion_path.resolve()}: {exc}."
        ) from exc
    if not isinstance(completion, dict):
        raise ValueError(
            f"Expected completion mapping at {completion_path.resolve()}, "
            f"but got {type(completion).__name__}."
        )
    return completion


def _validate_completion(
    completion: Mapping[str, Any],
    completion_path: Path,
    dataset_name: str,
    dataset_config: str,
    model_type: str,
) -> None:
    """Require one completion to describe the expected extractor result."""
    if completion.get("status") != "completed":
        raise ValueError(
            f"Expected completed status for {dataset_name} at "
            f"{completion_path.resolve()}, but got "
            f"{completion.get('status')!r}."
        )
    if completion.get("dataset_config") != dataset_config:
        raise ValueError(
            f"Expected dataset config {dataset_config!r} for "
            f"{dataset_name} at {completion_path.resolve()}, but got "
            f"{completion.get('dataset_config')!r}."
        )
    if completion.get("model_type") != model_type:
        raise ValueError(
            f"Expected model type {model_type!r} for {dataset_name} at "
            f"{completion_path.resolve()}, but got "
            f"{completion.get('model_type')!r}."
        )
    if completion.get("has_checkpoint") is not False:
        raise ValueError(
            f"Expected no checkpoint for feature extractor {dataset_name} "
            f"at {completion_path.resolve()}."
        )
    if completion.get("checkpoint_path") is not None:
        raise ValueError(
            f"Expected null checkpoint_path for feature extractor "
            f"{dataset_name} at {completion_path.resolve()}."
        )
    validate_completion_diagnostics(completion)
    metrics = completion.get("test_metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError(
            f"Expected non-empty test_metrics mapping for {dataset_name} "
            f"at {completion_path.resolve()}."
        )


def _metric_rows(
    metrics: Mapping[str, Any],
    dataset_name: str,
    seed: int,
    completion_path: Path,
) -> list[dict[str, Any]]:
    """Convert validated extractor test metrics into summary table rows."""
    expected_prefix = f"{dataset_name}/test/"
    rows: list[dict[str, Any]] = []
    for metric_key, value in sorted(metrics.items()):
        if not isinstance(metric_key, str) or not metric_key.startswith(
            expected_prefix
        ):
            raise ValueError(
                f"Expected test metric key beginning with "
                f"{expected_prefix!r} at {completion_path.resolve()}, "
                f"but got {metric_key!r}."
            )
        metric = metric_key.removeprefix(expected_prefix)
        if not metric:
            raise ValueError(
                f"Test metric key is empty at {completion_path.resolve()}."
            )
        if metric in FORBIDDEN_TEST_METRICS:
            raise ValueError(
                f"Feature-extractor completion at {completion_path.resolve()} "
                f"must not contain neural-only test metric {metric!r}."
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Expected numeric test metric {metric_key!r} at "
                f"{completion_path.resolve()}, but got {value!r}."
            )
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(
                f"Test metric {metric_key!r} is not finite at "
                f"{completion_path.resolve()}."
            )
        rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
                "metric": metric,
                "value": numeric_value,
            }
        )
    return rows


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[Mapping[str, Any]],
) -> None:
    """Atomically write one summary CSV table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write one summary JSON mapping."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)
