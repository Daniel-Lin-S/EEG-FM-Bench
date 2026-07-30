"""Load, validate, aggregate, and persist cross-artifact test results.

Inputs are complete artifact ``summary/test_runs.csv`` files selected by a
local YAML configuration. Outputs are normalized data, aggregate metrics,
statistics, diagnostics, and display-ready comparison tables. Source
artifacts are only read; output paths are managed separately.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import mannwhitneyu

from plot.results.config import ResultComparisonConfig
from plot.results.table_visualizer import TableDisplay


TEST_RUN_FIELDS = ("dataset", "seed", "metric", "value")
RAW_FIELDS = ("artifact",) + TEST_RUN_FIELDS
SUMMARY_BASE_FIELDS = (
    "artifact",
    "dataset",
    "metric",
    "count",
    "mean",
    "median",
)
STATISTICS_FIELDS = (
    "metric",
    "dataset",
    "artifact_a",
    "artifact_b",
    "count_a",
    "count_b",
    "u_statistic",
    "cliffs_delta",
    "p_value",
    "q_value",
)
SUMMARY_DIRECTORY = "summary"
TEST_RUN_FILENAME = "test_runs.csv"
SUMMARY_STATUS_FILENAME = "summary.json"
MANIFEST_FILENAME = "comparison_manifest.json"
MIN_INFERENCE_SAMPLES = 2
DISPLAY_PRECISION = 4


@dataclass(frozen=True)
class ComparisonResult:
    """Validated and derived comparison data.

    Parameters
    ----------
    raw_rows : list[dict[str, Any]]
        Selected, normalized per-seed test metric rows.
    summary_rows : list[dict[str, Any]]
        Aggregate rows grouped by artifact, dataset, and metric.
    statistic_rows : list[dict[str, Any]]
        Eligible pairwise test rows with BH-adjusted q-values.
    diagnostics : list[dict[str, Any]]
        Non-fatal data availability and inference diagnostics.
    """

    raw_rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
    statistic_rows: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]


class ManagedOutput:
    """Safely refresh files owned by one comparison output directory."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._previous_files = self._load_previous_files()
        self._current_files: set[str] = set()

    def path(self, relative_path: str) -> Path:
        """Return and register one safe workflow-owned output path."""
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                "Managed output path must be relative and contained within "
                f"{self.output_dir}."
            )
        path = self.output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        self._current_files.add(relative.as_posix())
        return path

    def finalize(self, metadata: Mapping[str, Any]) -> None:
        """Write the manifest and remove stale files from the prior refresh."""
        manifest_path = self.path(MANIFEST_FILENAME)
        payload = dict(metadata)
        payload["managed_files"] = sorted(self._current_files)
        _atomic_write_json(manifest_path, payload)
        stale_files = self._previous_files - self._current_files
        for relative_path in stale_files:
            path = self._contained_path(relative_path)
            if path.is_file():
                path.unlink()
                _remove_empty_parents(path.parent, self.output_dir)

    def _load_previous_files(self) -> set[str]:
        """Read and validate the prior workflow manifest, when present."""
        manifest_path = self.output_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            return set()
        if not manifest_path.is_file():
            raise ValueError(
                "Expected comparison manifest file at "
                f"{manifest_path.resolve()}."
            )
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Comparison manifest is invalid JSON at "
                f"{manifest_path.resolve()}: {exc}."
            ) from exc
        managed_files = payload.get("managed_files")
        if not isinstance(managed_files, list):
            raise ValueError(
                "Comparison manifest must contain a managed_files list at "
                f"{manifest_path.resolve()}."
            )
        previous_files = set()
        for relative_path in managed_files:
            self._contained_path(str(relative_path))
            previous_files.add(Path(str(relative_path)).as_posix())
        return previous_files

    def _contained_path(self, relative_path: str) -> Path:
        """Validate one manifest-relative path and return its absolute path."""
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                "Comparison manifest contains unsafe managed path "
                f"{relative_path!r}."
            )
        path = self.output_dir / relative
        try:
            path.resolve().relative_to(self.output_dir)
        except ValueError as exc:
            raise ValueError(
                "Comparison manifest path escapes its output directory: "
                f"{relative_path!r}."
            ) from exc
        return path


def validate_comparison_paths(config: ResultComparisonConfig) -> None:
    """Require existing sources and an output root outside all sources.

    Parameters
    ----------
    config : ResultComparisonConfig
        Validated YAML settings.

    Raises
    ------
    ValueError
        If a source root is missing or the output root would alter a source
        artifact directory.
    """
    output_dir = config.output_dir.resolve()
    for artifact in config.artifacts:
        root = artifact.root.resolve()
        if not root.is_dir():
            raise ValueError(
                f"Artifact root for {artifact.label!r} does not exist or is "
                f"not a directory: {root}."
            )
        if output_dir == root or root in output_dir.parents:
            raise ValueError(
                "Comparison output_dir must not be an artifact root or a "
                f"descendant of one: {output_dir}."
            )


def collect_comparison_result(
    config: ResultComparisonConfig,
) -> ComparisonResult:
    """Load selected source rows and derive all comparison values.

    Parameters
    ----------
    config : ResultComparisonConfig
        Validated comparison configuration.

    Returns
    -------
    ComparisonResult
        Normalized, aggregate, pairwise, and diagnostic results.
    """
    validate_comparison_paths(config)
    raw_rows: list[dict[str, Any]] = []
    for artifact in config.artifacts:
        rows = _read_artifact_rows(
            artifact.label,
            artifact.root,
            config.selected_metric_names,
        )
        raw_rows.extend(rows)
    _require_configured_datasets(raw_rows, config)
    display_rows = _select_display_rows(raw_rows, config)
    if not display_rows:
        raise ValueError(
            "No configured dataset and task-metric pair has values from every "
            "provided artifact."
        )
    summary_rows = _summarize_rows(display_rows)
    statistic_rows, diagnostics = _build_statistics(display_rows)
    diagnostics.extend(_missing_dataset_metric_diagnostics(raw_rows, config))
    return ComparisonResult(
        raw_rows=display_rows,
        summary_rows=summary_rows,
        statistic_rows=statistic_rows,
        diagnostics=diagnostics,
    )


def create_table_display(
    result: ComparisonResult,
    config: ResultComparisonConfig,
) -> TableDisplay:
    """Create display-ready combined table values and best-result styles.

    Parameters
    ----------
    result : ComparisonResult
        Derived comparison values.
    config : ResultComparisonConfig
        Metric direction and near-best threshold settings.

    Returns
    -------
    TableDisplay
        Combined dataset/metric table prepared for Markdown and PNG outputs.
    """
    labels = [artifact.label for artifact in config.artifacts]
    directions = {metric.name: metric.direction for metric in config.metrics}
    summary_lookup = {
        (row["artifact"], row["dataset"], row["metric"]): row
        for row in result.summary_rows
    }
    pairs = _statistic_lookup(result.statistic_rows)
    dataset_metrics = _ordered_dataset_metrics(result.raw_rows, config)
    dataset_labels = {
        dataset.name: dataset.display_name or dataset.name
        for dataset in config.datasets
    }
    metric_labels = {
        metric.name: metric.display_name or metric.name
        for metric in config.metrics
    }
    headers = ["Dataset", "Metric", *labels]
    rows: list[list[str]] = []
    bold_cells: set[tuple[int, int]] = set()
    for row_index, (dataset, metric) in enumerate(dataset_metrics):
        values = {
            label: summary_lookup[(label, dataset, metric)]
            for label in labels
            if (label, dataset, metric) in summary_lookup
        }
        best_labels = _best_labels(values, directions[metric])
        display_row = [dataset_labels[dataset], metric_labels[metric]]
        for column_offset, label in enumerate(labels, start=2):
            summary = values.get(label)
            if summary is None:
                display_row.append("—")
                continue
            value = _format_summary_value(summary)
            if label in best_labels:
                bold_cells.add((row_index, column_offset))
            elif _is_near_best(
                label,
                best_labels,
                dataset,
                metric,
                pairs,
                config.statistics.near_best_q_threshold,
            ):
                value = f"{value}†"
            display_row.append(value)
        rows.append(display_row)
    return TableDisplay(
        headers=headers,
        rows=rows,
        bold_cells=frozenset(bold_cells),
    )


def write_result_data(
    result: ComparisonResult,
    output: ManagedOutput,
) -> None:
    """Save normalized result tables and diagnostics through ``output``."""
    _atomic_write_csv(
        output.path("comparison_runs.csv"),
        RAW_FIELDS,
        result.raw_rows,
    )
    summary_fields = list(SUMMARY_BASE_FIELDS)
    if any(row["std"] is not None for row in result.summary_rows):
        summary_fields.append("std")
    _atomic_write_csv(
        output.path("comparison_summary.csv"),
        summary_fields,
        result.summary_rows,
    )
    if result.statistic_rows:
        _atomic_write_csv(
            output.path("comparison_statistics.csv"),
            STATISTICS_FIELDS,
            result.statistic_rows,
        )
    _atomic_write_json(
        output.path("comparison_diagnostics.json"),
        {
            "diagnostic_count": len(result.diagnostics),
            "diagnostics": result.diagnostics,
        },
    )


def build_manifest_metadata(
    result: ComparisonResult,
    config: ResultComparisonConfig,
) -> dict[str, Any]:
    """Build source-path-free provenance metadata for one comparison."""
    dataset_metric_count = len(
        {(row["dataset"], row["metric"]) for row in result.raw_rows}
    )
    return {
        "artifacts": [artifact.label for artifact in config.artifacts],
        "metrics": [metric.model_dump() for metric in config.metrics],
        "statistics": {
            "method": "two-sided Mann-Whitney U",
            "effect_size": "Cliff's delta",
            "p_adjustment": "Benjamini-Hochberg per metric",
            "near_best_q_threshold": config.statistics.near_best_q_threshold,
        },
        "row_counts": {
            "test_runs": len(result.raw_rows),
            "dataset_metrics": dataset_metric_count,
            "eligible_pairwise_tests": len(result.statistic_rows),
        },
    }


def _read_artifact_rows(
    label: str,
    root: Path,
    selected_metrics: set[str],
) -> list[dict[str, Any]]:
    """Read selected canonical rows from one complete artifact summary."""
    summary_dir = root / SUMMARY_DIRECTORY
    _require_complete_summary(label, summary_dir)
    test_runs_path = summary_dir / TEST_RUN_FILENAME
    if not test_runs_path.is_file():
        raise ValueError(
            f"Test-run summary for {label!r} does not exist: "
            f"{test_runs_path.resolve()}."
        )
    with test_runs_path.open(newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        _validate_test_run_fields(reader.fieldnames, test_runs_path)
        rows = list(reader)
    if not rows:
        raise ValueError(
            f"Test-run summary for {label!r} is empty: "
            f"{test_runs_path.resolve()}."
        )
    selected_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for row_index, row in enumerate(rows, start=2):
        metric = row["metric"]
        if metric not in selected_metrics:
            continue
        normalized = _normalize_test_row(row, label, test_runs_path, row_index)
        key = (normalized["dataset"], normalized["seed"], normalized["metric"])
        if key in seen:
            raise ValueError(
                f"Duplicate selected test row for {label!r} at "
                f"{test_runs_path.resolve()}: {key}."
            )
        seen.add(key)
        selected_rows.append(normalized)
    return selected_rows


def _require_complete_summary(label: str, summary_dir: Path) -> None:
    """Require complete status metadata before using source results."""
    status_path = summary_dir / SUMMARY_STATUS_FILENAME
    if not status_path.is_file():
        raise ValueError(
            f"Summary status for {label!r} does not exist: "
            f"{status_path.resolve()}."
        )
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Summary status for {label!r} is invalid JSON at "
            f"{status_path.resolve()}: {exc}."
        ) from exc
    if payload.get("status") != "complete":
        raise ValueError(
            f"Expected complete summary status for {label!r} at "
            f"{status_path.resolve()}, but got {payload.get('status')!r}."
        )


def _validate_test_run_fields(
    fieldnames: list[str] | None,
    path: Path,
) -> None:
    """Require all canonical test-run CSV fields."""
    if fieldnames is None:
        raise ValueError(f"Test-run summary has no header: {path.resolve()}.")
    missing_fields = sorted(set(TEST_RUN_FIELDS) - set(fieldnames))
    if missing_fields:
        raise ValueError(
            f"Expected test-run fields {TEST_RUN_FIELDS} at {path.resolve()}, "
            f"but missing {missing_fields}."
        )


def _normalize_test_row(
    row: Mapping[str, str],
    label: str,
    path: Path,
    row_index: int,
) -> dict[str, Any]:
    """Validate and normalize one selected per-seed metric CSV row."""
    dataset = row["dataset"].strip()
    metric = row["metric"].strip()
    if not dataset or not metric:
        raise ValueError(
            f"Expected non-empty dataset and metric at {path.resolve()} row "
            f"{row_index}."
        )
    try:
        seed = int(row["seed"])
    except ValueError as exc:
        raise ValueError(
            f"Expected integer seed at {path.resolve()} row {row_index}, but "
            f"got {row['seed']!r}."
        ) from exc
    try:
        value = float(row["value"])
    except ValueError as exc:
        raise ValueError(
            f"Expected numeric value at {path.resolve()} row {row_index}, "
            f"but got {row['value']!r}."
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Expected finite value at {path.resolve()} row {row_index}, "
            f"but got {row['value']!r}."
        )
    return {
        "artifact": label,
        "dataset": dataset,
        "seed": seed,
        "metric": metric,
        "value": value,
    }


def _require_configured_datasets(
    rows: Iterable[Mapping[str, Any]],
    config: ResultComparisonConfig,
) -> None:
    """Require task metadata for every result dataset identifier."""
    configured = {dataset.name for dataset in config.datasets}
    observed = {str(row["dataset"]) for row in rows}
    unknown = sorted(observed - configured)
    if unknown:
        raise ValueError(
            "Result datasets require task metadata in datasets, but missing "
            f"{unknown}."
        )


def _select_display_rows(
    rows: Iterable[Mapping[str, Any]],
    config: ResultComparisonConfig,
) -> list[dict[str, Any]]:
    """Keep configured task metrics present for every artifact label."""
    rows_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    labels_by_pair: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        normalized = dict(row)
        key = (str(row["dataset"]), str(row["metric"]))
        rows_by_pair.setdefault(key, []).append(normalized)
        labels_by_pair.setdefault(key, set()).add(str(row["artifact"]))
    required_labels = {artifact.label for artifact in config.artifacts}
    task_metrics = {
        "binary": set(config.task_metrics.binary),
        "multiclass": set(config.task_metrics.multiclass),
    }
    selected_rows: list[dict[str, Any]] = []
    for dataset in config.datasets:
        for metric in task_metrics[dataset.task]:
            key = (dataset.name, metric)
            if labels_by_pair.get(key) == required_labels:
                selected_rows.extend(rows_by_pair[key])
    return selected_rows


def _summarize_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-seed rows by artifact, dataset, and metric."""
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["artifact"]), str(row["dataset"]), str(row["metric"]))
        grouped.setdefault(key, []).append(float(row["value"]))
    summaries = []
    for (artifact, dataset, metric), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=float)
        std = float(np.std(array, ddof=1)) if len(array) > 1 else None
        summaries.append(
            {
                "artifact": artifact,
                "dataset": dataset,
                "metric": metric,
                "count": len(array),
                "mean": float(np.mean(array)),
                "median": float(np.median(array)),
                "std": std,
            }
        )
    return summaries


def _build_statistics(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Calculate eligible pairwise tests and non-fatal availability messages."""
    grouped: dict[tuple[str, str, str], list[float]] = {}
    by_dataset_metric: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        artifact = str(row["artifact"])
        dataset = str(row["dataset"])
        metric = str(row["metric"])
        grouped.setdefault((artifact, dataset, metric), []).append(
            float(row["value"])
        )
        by_dataset_metric.setdefault((dataset, metric), set()).add(artifact)
    statistics: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for (dataset, metric), labels in sorted(by_dataset_metric.items()):
        for artifact_a, artifact_b in combinations(sorted(labels), 2):
            values_a = grouped[(artifact_a, dataset, metric)]
            values_b = grouped[(artifact_b, dataset, metric)]
            if min(len(values_a), len(values_b)) < MIN_INFERENCE_SAMPLES:
                diagnostics.append(
                    {
                        "kind": "insufficient_inference_samples",
                        "dataset": dataset,
                        "metric": metric,
                        "artifact_a": artifact_a,
                        "artifact_b": artifact_b,
                        "message": (
                            "Mann-Whitney U was unavailable because each "
                            "artifact requires at least two seed values."
                        ),
                    }
                )
                continue
            test = mannwhitneyu(values_a, values_b, alternative="two-sided")
            statistics.append(
                {
                    "metric": metric,
                    "dataset": dataset,
                    "artifact_a": artifact_a,
                    "artifact_b": artifact_b,
                    "count_a": len(values_a),
                    "count_b": len(values_b),
                    "u_statistic": float(test.statistic),
                    "cliffs_delta": _cliffs_delta(values_a, values_b),
                    "p_value": float(test.pvalue),
                }
            )
    _add_bh_q_values(statistics)
    return statistics, diagnostics


def _cliffs_delta(values_a: list[float], values_b: list[float]) -> float:
    """Return signed Cliff's delta for two independent groups."""
    array_a = np.asarray(values_a, dtype=float)
    array_b = np.asarray(values_b, dtype=float)
    differences = array_a[:, np.newaxis] - array_b[np.newaxis, :]
    difference_count = np.sum(differences > 0) - np.sum(
        differences < 0
    )
    return float(difference_count / differences.size)


def _add_bh_q_values(statistics: list[dict[str, Any]]) -> None:
    """Add Benjamini-Hochberg adjusted q-values independently per metric."""
    by_metric: dict[str, list[dict[str, Any]]] = {}
    for row in statistics:
        by_metric.setdefault(str(row["metric"]), []).append(row)
    for rows in by_metric.values():
        p_values = np.asarray([float(row["p_value"]) for row in rows])
        order = np.argsort(p_values)
        sorted_p = p_values[order]
        count = len(sorted_p)
        adjusted = np.empty(count, dtype=float)
        running_minimum = 1.0
        for index in range(count - 1, -1, -1):
            candidate = sorted_p[index] * count / (index + 1)
            running_minimum = min(running_minimum, candidate)
            adjusted[index] = min(running_minimum, 1.0)
        q_values = np.empty(count, dtype=float)
        q_values[order] = adjusted
        for row, q_value in zip(rows, q_values):
            row["q_value"] = float(q_value)


def _missing_dataset_metric_diagnostics(
    rows: Iterable[Mapping[str, Any]],
    config: ResultComparisonConfig,
) -> list[dict[str, Any]]:
    """Describe artifact labels without each available dataset/metric pair."""
    available = {
        (str(row["artifact"]), str(row["dataset"]), str(row["metric"]))
        for row in rows
    }
    labels = [artifact.label for artifact in config.artifacts]
    pairs = sorted({(dataset, metric) for _, dataset, metric in available})
    diagnostics = []
    for dataset, metric in pairs:
        missing = [
            label
            for label in labels
            if (label, dataset, metric) not in available
        ]
        if missing:
            diagnostics.append(
                {
                    "kind": "missing_dataset_metric",
                    "dataset": dataset,
                    "metric": metric,
                    "artifacts": missing,
                    "message": (
                        "Artifact labels have no result for this dataset "
                        "and metric."
                    ),
                }
            )
    return diagnostics


def _ordered_dataset_metrics(
    rows: Iterable[Mapping[str, Any]],
    config: ResultComparisonConfig,
) -> list[tuple[str, str]]:
    """Order table rows by configured dataset and task-metric order."""
    dataset_order = {
        dataset.name: index for index, dataset in enumerate(config.datasets)
    }
    metric_order = {
        ("binary", metric): index
        for index, metric in enumerate(config.task_metrics.binary)
    }
    metric_order.update(
        {
            ("multiclass", metric): index
            for index, metric in enumerate(config.task_metrics.multiclass)
        }
    )
    dataset_tasks = {
        dataset.name: dataset.task for dataset in config.datasets
    }
    pairs = {(str(row["dataset"]), str(row["metric"])) for row in rows}
    return sorted(
        pairs,
        key=lambda pair: (
            dataset_order[pair[0]],
            metric_order[(dataset_tasks[pair[0]], pair[1])],
        ),
    )



def _best_labels(
    values: Mapping[str, Mapping[str, Any]],
    direction: str,
) -> set[str]:
    """Return labels tied exactly for the directionally best mean."""
    means = {label: float(row["mean"]) for label, row in values.items()}
    best_mean = (
        max(means.values())
        if direction == "maximize"
        else min(means.values())
    )
    return {label for label, mean in means.items() if mean == best_mean}


def _statistic_lookup(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str, frozenset[str]], Mapping[str, Any]]:
    """Index statistics by dataset, metric, and unordered artifact labels."""
    return {
        (
            str(row["dataset"]),
            str(row["metric"]),
            frozenset({str(row["artifact_a"]), str(row["artifact_b"])}),
        ): row
        for row in rows
    }


def _is_near_best(
    label: str,
    best_labels: set[str],
    dataset: str,
    metric: str,
    pairs: Mapping[tuple[str, str, frozenset[str]], Mapping[str, Any]],
    threshold: float,
) -> bool:
    """Return whether a result is unseparated from every best tie."""
    comparisons = []
    for best_label in best_labels:
        key = (dataset, metric, frozenset({label, best_label}))
        statistic = pairs.get(key)
        if statistic is None:
            return False
        comparisons.append(float(statistic["q_value"]))
    return bool(comparisons) and all(
        q_value >= threshold for q_value in comparisons
    )


def _format_summary_value(summary: Mapping[str, Any]) -> str:
    """Format one group mean and optional standard deviation."""
    mean = float(summary["mean"])
    std = summary["std"]
    if std is None:
        return f"{mean:.{DISPLAY_PRECISION}f}"
    return f"{mean:.{DISPLAY_PRECISION}f} ± {float(std):.{DISPLAY_PRECISION}f}"


def _atomic_write_csv(
    path: Path,
    fieldnames: Iterable[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Atomically write one CSV file without introducing empty columns."""
    fields = list(fieldnames)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write one JSON mapping."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _remove_empty_parents(path: Path, root: Path) -> None:
    """Remove empty managed directories without traversing beyond ``root``."""
    current = path
    while current != root:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent
