"""Render per-metric cross-artifact performance comparison figures.

Inputs are validated normalized test rows and precomputed pairwise statistics.
Each metric produces one PNG with raw seed values, aggregate uncertainty, and
non-binary q-value plus Cliff's-delta annotations.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


MIN_FIGURE_WIDTH = 9.0
MIN_FIGURE_HEIGHT = 5.0
WIDTH_PER_DATASET = 1.45
HEIGHT_PER_ANNOTATION = 0.22
POINT_SIZE = 34.0
MEAN_MARKER_SIZE = 7.0
ERROR_LINE_WIDTH = 1.8
ANNOTATION_LINE_WIDTH = 1.4
ANNOTATION_FONT_SIZE = 7.0
DATASET_TICK_SIZE = 9.0
MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
MODEL_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
    "#999999",
)
JITTER_WIDTH = 0.07
GROUP_WIDTH = 0.72
ANNOTATION_GAP = 0.06


def save_metric_figures(
    raw_rows: Iterable[Mapping[str, Any]],
    statistic_rows: Iterable[Mapping[str, Any]],
    metric_names: Iterable[str],
    artifact_labels: list[str],
    output_dir: Path,
    dataset_display_names: Mapping[str, str],
    metric_display_names: Mapping[str, str],
    show_individual_points: bool,
    naive_artifact: str | None,
) -> list[Path]:
    """Save one annotated comparison figure for each requested metric.

    Parameters
    ----------
    raw_rows : Iterable[Mapping[str, Any]]
        Validated normalized rows with artifact, dataset, metric, and value.
    statistic_rows : Iterable[Mapping[str, Any]]
        Pairwise rows containing BH-adjusted q-values and Cliff's delta.
    metric_names : Iterable[str]
        Metrics to render in the requested configuration order.
    artifact_labels : list[str]
        Display order for artifact groups and legend entries.
    output_dir : pathlib.Path
        Existing directory for metric PNG outputs.

    Returns
    -------
    list[pathlib.Path]
        Every saved figure path in metric order.
    """
    raw_list = list(raw_rows)
    statistics_list = list(statistic_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for metric in metric_names:
        metric_rows = [row for row in raw_list if row["metric"] == metric]
        if not metric_rows:
            raise ValueError(
                f"Expected non-empty rows for selected metric {metric!r}."
            )
        metric_statistics = [
            row for row in statistics_list if row["metric"] == metric
        ]
        path = output_dir / f"{_safe_filename(metric)}.png"
        _save_metric_figure(
            metric,
            metric_rows,
            metric_statistics,
            artifact_labels,
            path,
            dataset_display_names,
            metric_display_names[metric],
            show_individual_points,
            naive_artifact,
        )
        saved_paths.append(path)
    return saved_paths


def _save_metric_figure(
    metric: str,
    rows: list[Mapping[str, Any]],
    statistics: list[Mapping[str, Any]],
    artifact_labels: list[str],
    path: Path,
    dataset_display_names: Mapping[str, str],
    metric_display_name: str,
    show_individual_points: bool,
    naive_artifact: str | None,
) -> None:
    """Render one metric figure with raw values and pairwise annotations."""
    sns.set_theme(style="whitegrid", context="notebook")
    datasets = sorted({str(row["dataset"]) for row in rows})
    values = [float(row["value"]) for row in rows]
    minimum = min(values)
    maximum = max(values)
    value_range = max(maximum - minimum, 1e-6)
    annotation_counts = _annotation_counts(statistics)
    max_annotations = max(annotation_counts.values(), default=0)
    width = max(MIN_FIGURE_WIDTH, WIDTH_PER_DATASET * len(datasets) + 3.0)
    height = max(
        MIN_FIGURE_HEIGHT,
        MIN_FIGURE_HEIGHT + HEIGHT_PER_ANNOTATION * max_annotations,
    )
    figure, axis = plt.subplots(figsize=(width, height))
    offsets = _artifact_offsets(len(artifact_labels))
    position_lookup = {
        (dataset, label): dataset_index + offsets[label_index]
        for dataset_index, dataset in enumerate(datasets)
        for label_index, label in enumerate(artifact_labels)
    }
    legend_labels: set[str] = set()
    for label_index, label in enumerate(artifact_labels):
        if label == naive_artifact:
            _draw_naive_reference(
                axis,
                rows,
                datasets,
                label,
                legend_labels,
            )
            continue
        label_rows = [row for row in rows if row["artifact"] == label]
        if not label_rows:
            continue
        color = MODEL_COLORS[label_index % len(MODEL_COLORS)]
        marker = MARKERS[label_index % len(MARKERS)]
        for dataset in datasets:
            group_values = [
                float(row["value"])
                for row in label_rows
                if row["dataset"] == dataset
            ]
            if not group_values:
                continue
            position = position_lookup[(dataset, label)]
            if show_individual_points:
                jitter = _deterministic_jitter(len(group_values))
                axis.scatter(
                    np.full(len(group_values), position) + jitter,
                    group_values,
                    color=color,
                    marker=marker,
                    s=POINT_SIZE,
                    alpha=0.75,
                    linewidths=0.6,
                    edgecolors="white",
                    zorder=3,
                )
            mean = float(np.mean(group_values))
            std = (
                float(np.std(group_values, ddof=1))
                if len(group_values) > 1
                else None
            )
            error = None if std is None else std
            axis.errorbar(
                position,
                mean,
                yerr=error,
                color=color,
                marker=marker,
                markersize=MEAN_MARKER_SIZE,
                markeredgecolor="white",
                markeredgewidth=0.8,
                linewidth=ERROR_LINE_WIDTH,
                capsize=3.0,
                label=label if label not in legend_labels else None,
                
                zorder=4,
            )
            legend_labels.add(label)
    upper_limit = _add_statistic_annotations(
        axis,
        statistics,
        position_lookup,
        rows,
        values,
        value_range,
    )
    lower_padding = value_range * 0.08
    axis.set_ylim(minimum - lower_padding, upper_limit + lower_padding)
    axis.set_xticks(
        range(len(datasets)),
        [dataset_display_names[dataset] for dataset in datasets],
        fontsize=DATASET_TICK_SIZE,
    )
    axis.set_xlabel("Dataset")
    axis.set_ylabel(metric_display_name)
    title = textwrap.fill(
        f"Cross-artifact comparison for test metric: {metric_display_name}",
        width=70,
    )
    axis.set_title(title, pad=18.0)
    axis.legend(
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=False,
        title="Artifact",
    )
    sns.despine(ax=axis)
    figure.tight_layout()
    _atomic_save_figure(figure, path)
    plt.close(figure)


def _annotation_counts(
    statistics: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    """Count pairwise annotation rows by dataset."""
    counts: dict[str, int] = {}
    for row in statistics:
        dataset = str(row["dataset"])
        counts[dataset] = counts.get(dataset, 0) + 1
    return counts


def _artifact_offsets(count: int) -> np.ndarray:
    """Return centered categorical offsets for artifact groups."""
    if count == 1:
        return np.array([0.0])
    return np.linspace(-GROUP_WIDTH / 2.0, GROUP_WIDTH / 2.0, count)


def _deterministic_jitter(count: int) -> np.ndarray:
    """Return repeatable point offsets for one artifact-dataset group."""
    if count == 1:
        return np.array([0.0])
    return np.linspace(-JITTER_WIDTH, JITTER_WIDTH, count)


def _draw_naive_reference(
    axis: plt.Axes,
    rows: list[Mapping[str, Any]],
    datasets: list[str],
    naive_artifact: str,
    legend_labels: set[str],
) -> None:
    """Draw the mean naive result as one dotted line per dataset group."""
    for dataset_index, dataset in enumerate(datasets):
        values = [
            float(row["value"])
            for row in rows
            if row["artifact"] == naive_artifact
            and row["dataset"] == dataset
        ]
        if not values:
            continue
        axis.hlines(
            float(np.mean(values)),
            dataset_index - GROUP_WIDTH / 2.0,
            dataset_index + GROUP_WIDTH / 2.0,
            color="#4d4d4d",
            linestyle=":",
            linewidth=ERROR_LINE_WIDTH,
            label=(
                naive_artifact
                if naive_artifact not in legend_labels
                else None
            ),
            zorder=2,
        )
        legend_labels.add(naive_artifact)


def _add_statistic_annotations(
    axis: plt.Axes,
    statistics: list[Mapping[str, Any]],
    positions: Mapping[tuple[str, str], float],
    rows: list[Mapping[str, Any]],
    values: list[float],
    value_range: float,
) -> float:
    """Draw q-value and Cliff's-delta brackets above each tested group."""
    global_upper = max(values)
    grouped = _group_statistics_by_dataset(statistics)
    for dataset, dataset_statistics in grouped.items():
        dataset_values = [
            float(row["value"])
            for row in rows
            if row["dataset"] == dataset
        ]
        base = max(dataset_values)
        for index, row in enumerate(dataset_statistics, start=1):
            artifact_a = str(row["artifact_a"])
            artifact_b = str(row["artifact_b"])
            x_a = positions[(dataset, artifact_a)]
            x_b = positions[(dataset, artifact_b)]
            height = base + value_range * (ANNOTATION_GAP * (index + 1))
            arm = value_range * ANNOTATION_GAP * 0.35
            axis.plot(
                [x_a, x_a, x_b, x_b],
                [height - arm, height, height, height - arm],
                color="#1a1a1a",
                linewidth=ANNOTATION_LINE_WIDTH,
                zorder=5,
            )
            text = (
                f"q={float(row['q_value']):.3g}; "
                f"δ={float(row['cliffs_delta']):+.2f}"
            )
            axis.text(
                (x_a + x_b) / 2.0,
                height + arm,
                text,
                ha="center",
                va="bottom",
                fontsize=ANNOTATION_FONT_SIZE,
            )
            global_upper = max(global_upper, height + arm * 3.0)
    return global_upper


def _group_statistics_by_dataset(
    statistics: Iterable[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    """Group and deterministically order annotation rows by dataset."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in statistics:
        grouped.setdefault(str(row["dataset"]), []).append(row)
    for rows in grouped.values():
        rows.sort(
            key=lambda row: (
                str(row["artifact_a"]),
                str(row["artifact_b"]),
            )
        )
    return grouped


def _safe_filename(metric: str) -> str:
    """Convert a metric name into a non-empty portable PNG stem."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", metric).strip("._")
    if not normalized:
        raise ValueError(f"Metric name cannot form a file name: {metric!r}.")
    return normalized


def _atomic_save_figure(figure: plt.Figure, path: Path) -> None:
    """Atomically save one PNG figure."""
    temporary_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary_path, dpi=200, bbox_inches="tight")
    temporary_path.replace(path)
