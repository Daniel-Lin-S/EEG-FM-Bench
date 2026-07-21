"""Configuration types for the catch22 feature-extractor baseline."""

from pydantic import BaseModel, Field

from baseline.feature_extractor.classifier import RidgeClassifierArgs
from baseline.feature_extractor.config import FeatureExtractorConfig


class Catch22ModelArgs(BaseModel):
    """Compose canonical catch22 extraction with a Ridge classifier."""

    classifier: RidgeClassifierArgs = Field(
        default_factory=RidgeClassifierArgs
    )


class Catch22Config(FeatureExtractorConfig):
    """Configuration for canonical per-channel catch22 extraction."""

    model_type: str = "catch22"
    model: Catch22ModelArgs = Field(default_factory=Catch22ModelArgs)
