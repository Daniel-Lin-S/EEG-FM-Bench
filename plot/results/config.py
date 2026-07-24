"""Validate YAML input for cross-artifact result comparisons.

The comparison YAML provides local artifact roots, user-defined labels,
explicit metric directions, and a separate output root. It is intentionally
local because artifact paths must not appear in public configuration files.
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


class ArtifactConfig(BaseModel):
    """One explicitly labelled artifact root.

    Parameters
    ----------
    label : str
        User-defined comparison label. It may describe a model or an
        experimental condition.
    root : pathlib.Path
        Absolute root containing ``summary/test_runs.csv``.
    """

    label: str
    root: Path

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        """Require a non-empty artifact label."""
        if not value.strip():
            raise ValueError("Artifact label must not be empty.")
        return value

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: Path) -> Path:
        """Require an absolute artifact root path."""
        if not value.is_absolute():
            raise ValueError(
                "Artifact root must be an absolute path, but got "
                f"{value}."
            )
        return value


class MetricConfig(BaseModel):
    """One test metric selected for comparison.

    Parameters
    ----------
    name : str
        Test-metric name from ``summary/test_runs.csv``.
    direction : {"maximize", "minimize"}
        Direction used to identify the numerically best mean.
    """

    name: str
    direction: Literal["maximize", "minimize"]

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
        Explicit test metrics and their ranking directions.
    output_dir : pathlib.Path
        Absolute directory owned by this comparison workflow.
    statistics : StatisticsConfig
        Required statistical display settings.
    """

    artifacts: list[ArtifactConfig]
    metrics: list[MetricConfig]
    output_dir: Path
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
        if not self.metrics:
            raise ValueError(
                "Expected at least one explicitly selected metric."
            )
        metric_names = [metric.name for metric in self.metrics]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("Selected metric names must be unique.")
        return self





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
