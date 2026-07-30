"""Validate YAML input for cross-artifact result comparisons.

The comparison YAML provides completed artifact roots and optionally XLSX
paper-result sources. It is intentionally local because source paths must not
appear in public configuration files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, field_validator, model_validator


FORBIDDEN_METRIC = "epoch"
MIN_ARTIFACT_COUNT = 2
MIN_Q_THRESHOLD = 0.0
MAX_Q_THRESHOLD = 1.0

TaskType = Literal["binary", "multiclass"]


class SpreadsheetMetricConfig(BaseModel):
    """Map one source metric label to benchmark metrics by task type."""

    binary: str
    multiclass: str


class SpreadsheetSourceConfig(BaseModel):
    """One configurable wide XLSX paper-result source.

    Parameters
    ----------
    name : str
        Identifier referenced by spreadsheet artifacts.
    path : pathlib.Path
        Absolute XLSX path.
    sheet : str or int, optional, default=0
        Worksheet name or zero-based worksheet index.
    header_row : int, optional, default=1
        One-based row containing the source column headers.
    dataset_column : str
        Header containing a source dataset name.
    metric_column : str
        Header containing a source metric label.
    dataset_map : dict[str, str]
        Source dataset names mapped to configured benchmark identifiers.
    metric_map : dict[str, SpreadsheetMetricConfig]
        Source metric labels mapped separately for binary and multiclass tasks.
    value_scale : float, optional, default=1.0
        Multiplier applied to reported means and standard deviations.
    """

    name: str
    path: Path
    sheet: str | int = 0
    header_row: int = Field(default=1, ge=1)
    dataset_column: str
    metric_column: str
    dataset_map: dict[str, str]
    metric_map: dict[str, SpreadsheetMetricConfig]
    value_scale: float = Field(default=1.0, gt=0.0)

    @field_validator("name", "dataset_column", "metric_column")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        """Normalize a required non-empty spreadsheet setting."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("Spreadsheet settings must not be empty.")
        return normalized

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: Path) -> Path:
        """Require an absolute spreadsheet source path."""
        if not value.is_absolute():
            raise ValueError(
                "Spreadsheet source path must be absolute, but got "
                f"{value}."
            )
        return value


class SpreadsheetArtifactConfig(BaseModel):
    """Select one model column from a configured XLSX source."""

    source: str
    model_column: str

    @field_validator("source", "model_column")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        """Normalize one non-empty spreadsheet artifact setting."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("Spreadsheet artifact settings must not be empty.")
        return normalized


class ArtifactConfig(BaseModel):
    """One explicitly labelled artifact or spreadsheet model column.

    Parameters
    ----------
    label : str
        User-defined comparison label. It may describe a model or an
        experimental condition.
    root : pathlib.Path or None, optional, default=None
        Absolute root containing ``summary/test_runs.csv``.
    spreadsheet : SpreadsheetArtifactConfig or None, optional, default=None
        Configured XLSX source and model column. Exactly one source is required.
    """

    label: str
    root: Path | None = None
    spreadsheet: SpreadsheetArtifactConfig | None = None

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        """Require a non-empty artifact label."""
        if not value.strip():
            raise ValueError("Artifact label must not be empty.")
        return value

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: Path | None) -> Path | None:
        """Require an absolute artifact root path."""
        if value is None:
            return value
        if not value.is_absolute():
            raise ValueError(
                "Artifact root must be an absolute path, but got "
                f"{value}."
            )
        return value

    @model_validator(mode="after")
    def validate_source(self) -> "ArtifactConfig":
        """Require exactly one artifact-root or spreadsheet source."""
        if (self.root is None) == (self.spreadsheet is None):
            raise ValueError(
                "Each artifact must define exactly one of root or spreadsheet."
            )
        return self


class MetricConfig(BaseModel):
    """One test metric selected for comparison.

    Parameters
    ----------
    name : str
        Test-metric name from ``summary/test_runs.csv``.
    direction : {"maximize", "minimize"}
        Direction used to identify the numerically best mean.
    display_name : str or None, optional, default=None
        Label used in tables and metric figures. ``name`` is used when absent.
    """

    name: str
    direction: Literal["maximize", "minimize"]
    display_name: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Reject empty and neural-loop-only metrics."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("Metric name must not be empty.")
        if normalized.lower() == FORBIDDEN_METRIC:
            raise ValueError(
                "Metric 'epoch' is a training-loop coordinate and cannot "
                "be compared."
            )
        return normalized


class DatasetConfig(BaseModel):
    """One dataset's task type and optional display label.

    Parameters
    ----------
    name : str
        Dataset identifier in the artifact ``test_runs.csv`` files.
    task : {"binary", "multiclass"}
        Classification task that determines the displayed metric subset.
    display_name : str or None, optional, default=None
        Label used in tables and metric figures. ``name`` is used when absent.
    """

    name: str
    task: TaskType
    display_name: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Require a non-empty result dataset identifier."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("Dataset name must not be empty.")
        return normalized


class TaskMetricsConfig(BaseModel):
    """Map binary and multiclass tasks to configured metric names.

    Parameters
    ----------
    binary : list[str]
        Metrics displayed for every configured binary dataset.
    multiclass : list[str]
        Metrics displayed for every configured multiclass dataset.
    """

    binary: list[str]
    multiclass: list[str]

    @field_validator("binary", "multiclass")
    @classmethod
    def validate_metric_names(cls, values: list[str]) -> list[str]:
        """Require a non-empty, duplicate-free metric sequence."""
        normalized = [value.strip() for value in values]
        if not normalized or any(not value for value in normalized):
            raise ValueError("Task metrics must contain non-empty names.")
        if len(normalized) != len(set(normalized)):
            raise ValueError("Task metric names must be unique per task.")
        return normalized



class PlotConfig(BaseModel):
    """Optional figure-rendering settings.

    Parameters
    ----------
    show_individual_points : bool, optional, default=True
        Whether each seed value is plotted in addition to mean and error bars.
    naive_artifact : str or None, optional, default=None
        Artifact label rendered as a horizontal dotted baseline per dataset.
    """

    show_individual_points: bool = True
    naive_artifact: str | None = None

    @field_validator("naive_artifact")
    @classmethod
    def validate_naive_artifact(cls, value: str | None) -> str | None:
        """Normalize an optional naive-artifact label."""
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("plot.naive_artifact must not be empty.")
        return normalized


class StatisticsConfig(BaseModel):
    """Statistical settings for tables and plot annotations.

    Parameters
    ----------
    near_best_q_threshold : float
        Adjusted q-value at or above which a lower-ranked result receives a
        dagger in the summary table.
    """

    near_best_q_threshold: float = Field(
        ge=MIN_Q_THRESHOLD,
        le=MAX_Q_THRESHOLD,
    )


class ResultComparisonConfig(BaseModel):
    """Validated cross-artifact comparison configuration.

    Parameters
    ----------
    artifacts : list[ArtifactConfig]
        At least two labelled source artifact roots.
    metrics : list[MetricConfig]
        Explicit metrics, ranking directions, and optional display labels.
    task_metrics : TaskMetricsConfig
        Metric subsets displayed for binary and multiclass datasets.
    datasets : list[DatasetConfig]
        Dataset task types and optional display labels.
    output_dir : pathlib.Path
        Absolute directory owned by this comparison workflow.
    plot : PlotConfig, optional
        Figure settings including seed dots and a naive-artifact reference.
    statistics : StatisticsConfig
        Required statistical display settings.
    """

    artifacts: list[ArtifactConfig]
    spreadsheet_sources: list[SpreadsheetSourceConfig] = Field(
        default_factory=list,
    )
    metrics: list[MetricConfig]
    task_metrics: TaskMetricsConfig
    datasets: list[DatasetConfig]
    output_dir: Path
    plot: PlotConfig = Field(default_factory=PlotConfig)
    statistics: StatisticsConfig

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, value: Path) -> Path:
        """Require an absolute output directory."""
        if not value.is_absolute():
            raise ValueError(
                "Comparison output_dir must be an absolute path, but got "
                f"{value}."
            )
        return value

    @model_validator(mode="after")
    def validate_unique_fields(self) -> "ResultComparisonConfig":
        """Require sufficient and unambiguous comparison inputs."""
        if len(self.artifacts) < MIN_ARTIFACT_COUNT:
            raise ValueError(
                "Expected at least two artifacts for comparison, but got "
                f"{len(self.artifacts)}."
            )
        labels = [artifact.label for artifact in self.artifacts]
        if len(labels) != len(set(labels)):
            raise ValueError("Artifact labels must be unique.")
        source_names = [source.name for source in self.spreadsheet_sources]
        if len(source_names) != len(set(source_names)):
            raise ValueError("Spreadsheet source names must be unique.")
        sources = {source.name: source for source in self.spreadsheet_sources}
        for artifact in self.artifacts:
            if artifact.spreadsheet is None:
                continue
            source = sources.get(artifact.spreadsheet.source)
            if source is None:
                raise ValueError(
                    "Spreadsheet artifact references an unknown source: "
                    f"{artifact.spreadsheet.source!r}."
                )
            targets = set(source.dataset_map.values())
            unknown_targets = sorted(
                targets - {dataset.name for dataset in self.datasets}
            )
            if unknown_targets:
                raise ValueError(
                    "Spreadsheet dataset_map targets require configured "
                    f"datasets, but missing {unknown_targets}."
                )
        if not self.metrics:
            raise ValueError(
                "Expected at least one explicitly selected metric."
            )
        metric_names = [metric.name for metric in self.metrics]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("Selected metric names must be unique.")
        task_metric_names = set(self.task_metrics.binary).union(
            self.task_metrics.multiclass,
        )
        unknown_metrics = sorted(task_metric_names - set(metric_names))
        if unknown_metrics:
            raise ValueError(
                "Task metrics must reference configured metrics, but got "
                f"{unknown_metrics}."
            )
        unassigned_metrics = sorted(set(metric_names) - task_metric_names)
        if unassigned_metrics:
            raise ValueError(
                "Every configured metric must be assigned to a task, but "
                f"got {unassigned_metrics}."
            )
        if not self.datasets:
            raise ValueError("Expected at least one configured dataset.")
        dataset_names = [dataset.name for dataset in self.datasets]
        if len(dataset_names) != len(set(dataset_names)):
            raise ValueError("Configured dataset names must be unique.")
        if (
            self.plot.naive_artifact is not None
            and self.plot.naive_artifact not in labels
        ):
            raise ValueError(
                "plot.naive_artifact must match one artifact label, but got "
                f"{self.plot.naive_artifact!r}."
            )
        return self

    @property
    def selected_metric_names(self) -> set[str]:
        """Return every metric referenced by one configured task."""
        return set(self.task_metrics.binary).union(
            self.task_metrics.multiclass,
        )

    @property
    def spreadsheet_source_lookup(self) -> dict[str, SpreadsheetSourceConfig]:
        """Return spreadsheet source settings keyed by their unique name."""
        return {source.name: source for source in self.spreadsheet_sources}





def load_result_comparison_config(
    config_path: Path,
) -> ResultComparisonConfig:
    """Load and validate one local result-comparison YAML mapping.

    Parameters
    ----------
    config_path : pathlib.Path
        Existing YAML path supplied to ``result_vis.py``.

    Returns
    -------
    ResultComparisonConfig
        Fully resolved and validated comparison settings.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist.
    ValueError
        If YAML content is not a mapping accepted by the schema.
    """
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Comparison configuration does not exist: "
            f"{config_path.resolve()}."
        )
    loaded = OmegaConf.load(config_path)
    resolved = OmegaConf.to_container(
        loaded,
        resolve=True,
        throw_on_missing=True,
    )
    if not isinstance(resolved, dict):
        raise ValueError(
            "Expected comparison YAML mapping at "
            f"{config_path.resolve()}, but got "
            f"{type(resolved).__name__}."
        )
    return ResultComparisonConfig.model_validate(resolved)
