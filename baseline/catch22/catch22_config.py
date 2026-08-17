"""Configuration types for the catch22 feature-extractor baseline."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from baseline.feature_extractor.classifier import RidgeClassifierArgs
from baseline.feature_extractor.config import (
    FeatureExtractorConfig,
    normalize_model_field_aliases,
)


DEFAULT_CATCH22_WORKERS = 8


class Catch22ExtractorArgs(BaseModel):
    """Execution settings for canonical per-channel catch22 extraction."""

    model_config = ConfigDict(extra="forbid")

    n_jobs: int = Field(default=DEFAULT_CATCH22_WORKERS, ge=1)


class Catch22ModelArgs(BaseModel):
    """Compose canonical catch22 extraction with a Ridge classifier."""

    model_config = ConfigDict(extra="forbid")

    extractor: Catch22ExtractorArgs = Field(
        default_factory=Catch22ExtractorArgs
    )
    classifier: RidgeClassifierArgs = Field(
        default_factory=RidgeClassifierArgs
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, value: Any) -> Any:
        """Canonicalize the historical flat Ridge model structure."""
        return normalize_model_field_aliases(
            value,
            {
                "ridge_alphas": ("classifier", "alphas"),
                "ridge_selection_metric": (
                    "classifier",
                    "selection_metric",
                ),
            },
        )


class Catch22Config(FeatureExtractorConfig):
    """Configuration for canonical per-channel catch22 extraction."""

    model_type: str = "catch22"
    model: Catch22ModelArgs = Field(default_factory=Catch22ModelArgs)
