"""Configuration types for the catch22 feature-extractor baseline."""

from pydantic import BaseModel, Field

from baseline.feature_extractor.classifier import RidgeClassifierArgs
from baseline.feature_extractor.config import FeatureExtractorConfig


DEFAULT_CATCH22_WORKERS = 8


class Catch22ExtractorArgs(BaseModel):
    """Execution settings for canonical per-channel catch22 extraction."""

    n_jobs: int = Field(default=DEFAULT_CATCH22_WORKERS, ge=1)


class Catch22ModelArgs(BaseModel):
    """Compose canonical catch22 extraction with a Ridge classifier."""

    extractor: Catch22ExtractorArgs = Field(
        default_factory=Catch22ExtractorArgs
    )
    classifier: RidgeClassifierArgs = Field(
        default_factory=RidgeClassifierArgs
    )


class Catch22Config(FeatureExtractorConfig):
    """Configuration for canonical per-channel catch22 extraction."""

    model_type: str = "catch22"
    model: Catch22ModelArgs = Field(default_factory=Catch22ModelArgs)
