"""Configuration types for the external miniROCKET baseline."""

from typing import Any, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from baseline.feature_extractor.classifier import RidgeClassifierArgs
from baseline.feature_extractor.config import (
    FeatureExtractorConfig,
    normalize_model_field_aliases,
)


DEFAULT_MINIROCKET_THREADS = 8


class MiniRocketExtractorArgs(BaseModel):
    """Configuration for an external multivariate miniROCKET clone."""

    model_config = ConfigDict(extra="forbid")

    source_path: Optional[str] = None
    num_features: int = 10_000
    max_dilations_per_kernel: int = 32
    n_jobs: int = Field(default=DEFAULT_MINIROCKET_THREADS, ge=1)

    @field_validator("num_features", "max_dilations_per_kernel", "n_jobs")
    @classmethod
    def validate_positive_integer(cls, value: int) -> int:
        """Require positive miniROCKET integer settings."""
        if value <= 0:
            raise ValueError("miniROCKET extractor settings must be positive.")
        return value


class MiniRocketModelArgs(BaseModel):
    """Compose miniROCKET extraction with a Ridge classifier."""

    model_config = ConfigDict(extra="forbid")

    extractor: MiniRocketExtractorArgs = Field(
        default_factory=MiniRocketExtractorArgs
    )
    classifier: RidgeClassifierArgs = Field(
        default_factory=RidgeClassifierArgs
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, value: Any) -> Any:
        """Canonicalize historical flat MiniROCKET model fields."""
        return normalize_model_field_aliases(
            value,
            {
                "minirocket_source_path": ("extractor", "source_path"),
                "minirocket_num_features": ("extractor", "num_features"),
                "minirocket_max_dilations_per_kernel": (
                    "extractor",
                    "max_dilations_per_kernel",
                ),
                "ridge_alphas": ("classifier", "alphas"),
                "ridge_selection_metric": (
                    "classifier",
                    "selection_metric",
                ),
            },
        )


class MiniRocketConfig(FeatureExtractorConfig):
    """Configuration for external multivariate miniROCKET extraction."""

    model_type: str = "minirocket"
    model: MiniRocketModelArgs = Field(default_factory=MiniRocketModelArgs)
