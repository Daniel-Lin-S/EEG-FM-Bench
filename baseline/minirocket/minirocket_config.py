"""Configuration types for the external miniROCKET baseline."""

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from baseline.feature_extractor.classifier import RidgeClassifierArgs
from baseline.feature_extractor.config import FeatureExtractorConfig


class MiniRocketExtractorArgs(BaseModel):
    """Configuration for an external multivariate miniROCKET clone."""

    source_path: Optional[str] = None
    num_features: int = 10_000
    max_dilations_per_kernel: int = 32

    @field_validator("num_features", "max_dilations_per_kernel")
    @classmethod
    def validate_positive_integer(cls, value: int) -> int:
        """Require positive miniROCKET integer settings."""
        if value <= 0:
            raise ValueError("miniROCKET extractor settings must be positive.")
        return value


class MiniRocketModelArgs(BaseModel):
    """Compose miniROCKET extraction with a Ridge classifier."""

    extractor: MiniRocketExtractorArgs = Field(
        default_factory=MiniRocketExtractorArgs
    )
    classifier: RidgeClassifierArgs = Field(
        default_factory=RidgeClassifierArgs
    )


class MiniRocketConfig(FeatureExtractorConfig):
    """Configuration for external multivariate miniROCKET extraction."""

    model_type: str = "minirocket"
    model: MiniRocketModelArgs = Field(default_factory=MiniRocketModelArgs)
