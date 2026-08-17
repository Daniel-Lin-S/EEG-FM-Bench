"""Configuration types for CPU feature-extractor baselines.

These baselines consume preprocessed EEG arrays shaped
``(n_trials, n_channels, n_timepoints)`` and fit sklearn classifiers. They
write metrics and completion metadata but intentionally do not save models.
"""

import math
from typing import Any, List, Literal, Mapping, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator

from baseline.abstract.config import (
    AbstractConfig,
    BaseDataArgs,
    BaseLoggingArgs,
)


RIDGE_ALPHA_MIN_EXPONENT = -3
RIDGE_ALPHA_MAX_EXPONENT = 3
RIDGE_ALPHA_COUNT = 10


def normalize_model_field_aliases(
    value: Any,
    aliases: Mapping[str, tuple[str, str]],
) -> Any:
    """Move historical model fields into current nested sections.

    Parameters
    ----------
    value : Any
        Raw Pydantic model input.
    aliases : Mapping[str, tuple[str, str]]
        Legacy field names mapped to ``(section, canonical_field)``.

    Returns
    -------
    Any
        Canonicalized input, or the original non-mapping value.
    """
    if not isinstance(value, Mapping):
        return value
    normalized = dict(value)
    for legacy_field, (section_name, canonical_field) in aliases.items():
        if legacy_field not in normalized:
            continue
        section_value = normalized.get(section_name, {})
        if not isinstance(section_value, Mapping):
            raise ValueError(
                f"Expected model.{section_name} to be a mapping."
            )
        section = dict(section_value)
        legacy_value = normalized.pop(legacy_field)
        if (
            canonical_field in section
            and section[canonical_field] != legacy_value
        ):
            raise ValueError(
                f"Conflicting values exist for model.{legacy_field} and "
                f"model.{section_name}.{canonical_field}."
            )
        section.setdefault(canonical_field, legacy_value)
        normalized[section_name] = section
    return normalized


def get_default_ridge_alphas() -> List[float]:
    """Return the default Ridge regularization candidates."""
    return np.logspace(
        RIDGE_ALPHA_MIN_EXPONENT,
        RIDGE_ALPHA_MAX_EXPONENT,
        RIDGE_ALPHA_COUNT,
    ).tolist()


class FeatureExtractorModelArgs(BaseModel):
    """Settings shared by catch22 and miniROCKET classifiers.

    Parameters
    ----------
    ridge_alphas : list[float]
        Positive Ridge regularization strengths evaluated on the validation
        split after fitting each candidate on training features.
    ridge_selection_metric : {"balanced_accuracy", "accuracy", "f1_weighted"}
        Validation metric used to select the Ridge regularization strength.
    """

    ridge_alphas: List[float] = Field(default_factory=get_default_ridge_alphas)
    ridge_selection_metric: Literal[
        "balanced_accuracy",
        "accuracy",
        "f1_weighted",
    ] = "balanced_accuracy"

    @field_validator("ridge_alphas")
    @classmethod
    def validate_ridge_alphas(cls, ridge_alphas: List[float]) -> List[float]:
        """Validate Ridge regularization candidates."""
        if not ridge_alphas:
            raise ValueError(
                "model.ridge_alphas must contain at least one alpha."
            )
        if any(not np.isfinite(alpha) or alpha <= 0 for alpha in ridge_alphas):
            raise ValueError(
                "model.ridge_alphas must contain only finite positive values."
            )
        if len(ridge_alphas) != len(set(ridge_alphas)):
            raise ValueError("model.ridge_alphas must not contain duplicates.")
        return ridge_alphas


class MiniRocketModelArgs(FeatureExtractorModelArgs):
    """Settings for the external multivariate miniROCKET implementation.

    Parameters
    ----------
    minirocket_source_path : str, optional
        Root of a clone of the upstream miniROCKET repository. The trainer
        loads ``code/minirocket_multivariate.py`` from this directory.
    minirocket_num_features : int
        Requested number of miniROCKET output features.
    minirocket_max_dilations_per_kernel : int
        Maximum number of dilations allocated to each miniROCKET kernel.
    """

    minirocket_source_path: Optional[str] = None
    minirocket_num_features: int = 10_000
    minirocket_max_dilations_per_kernel: int = 32

    @field_validator(
        "minirocket_num_features",
        "minirocket_max_dilations_per_kernel",
    )
    @classmethod
    def validate_positive_integer(cls, value: int) -> int:
        """Validate positive miniROCKET integer settings."""
        if value <= 0:
            raise ValueError("miniROCKET feature settings must be positive.")
        return value


class FeatureExtractorTrainingArgs(BaseModel):
    """Empty namespace retained for the common baseline configuration shape."""


class FeatureExtractorLoggingArgs(BaseLoggingArgs):
    """Logging settings for non-checkpointing feature-extractor baselines."""

    experiment_name: str = "feature-extractor"
    project: Optional[str] = "feature-extractor"


class FeatureExtractorDataArgs(BaseDataArgs):
    """Data and resource settings for classical feature extractors.

    Parameters
    ----------
    load_batch_size : int, optional, default=256
        Arrow rows converted to NumPy together.
    feature_batch_size : int, optional, default=1024
        Trials transformed or predicted together after Arrow loading.
    memory_limit_gib : float, optional, default=64.0
        Linux address-space ceiling applied to one baseline process.
    scratch_dir : str or None, optional, default=None
        Temporary memmap root. ``None`` resolves below the ignored local
        project output root.
    """

    load_batch_size: int = 256
    feature_batch_size: int = 1024
    memory_limit_gib: float = 64.0
    scratch_dir: Optional[str] = None

    @field_validator(
        "load_batch_size",
        "feature_batch_size",
        "memory_limit_gib",
        mode="before",
    )
    @classmethod
    def reject_boolean_numeric_settings(cls, value: Any) -> Any:
        """Reject booleans masquerading as classical numeric settings."""
        if isinstance(value, bool):
            raise ValueError(
                "Classical numeric runtime settings must not be booleans."
            )
        return value

    @field_validator("load_batch_size", "feature_batch_size")
    @classmethod
    def validate_batch_size(cls, value: int) -> int:
        """Require positive classical processing batch sizes."""
        if value <= 0:
            raise ValueError(
                "Classical data and feature batch sizes must be positive."
            )
        return value

    @field_validator("memory_limit_gib")
    @classmethod
    def validate_memory_limit_gib(cls, value: float) -> float:
        """Require a finite positive address-space limit."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                "data.memory_limit_gib must be finite and positive."
            )
        return value

    @field_validator("scratch_dir")
    @classmethod
    def validate_scratch_dir(cls, value: Optional[str]) -> Optional[str]:
        """Reject an explicitly empty scratch-directory setting."""
        if value is not None and not value.strip():
            raise ValueError(
                "data.scratch_dir must be null or a non-empty path."
            )
        return value


class FeatureExtractorConfig(AbstractConfig):
    """Base configuration for per-dataset feature-extractor evaluation."""

    multitask: bool = False
    data: FeatureExtractorDataArgs = Field(
        default_factory=FeatureExtractorDataArgs
    )
    training: FeatureExtractorTrainingArgs = Field(
        default_factory=FeatureExtractorTrainingArgs
    )
    logging: FeatureExtractorLoggingArgs = Field(
        default_factory=FeatureExtractorLoggingArgs
    )

    def validate_config(self) -> bool:
        """Validate feature-extractor constraints after config merging."""
        if self.multitask:
            raise ValueError(
                f"{self.model_type} supports only multitask=false because "
                "each dataset has an independent feature dimension."
            )
        if not self.data.datasets:
            raise ValueError(
                f"{self.model_type} requires at least one configured dataset."
            )
        if len(self.seeds) != 1:
            raise ValueError(
                f"{self.model_type} supports exactly one seed because its "
                "feature extraction and classifier evaluation are "
                "deterministic."
            )
        return True
