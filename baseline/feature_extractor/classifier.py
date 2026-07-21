"""Downstream feature classifiers independent of EEG feature extraction."""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RIDGE_ALPHA_MIN_EXPONENT = -3
RIDGE_ALPHA_MAX_EXPONENT = 3
RIDGE_ALPHA_COUNT = 10


def get_default_ridge_alphas() -> list[float]:
    """Return the default Ridge regularization candidates."""
    return np.logspace(
        RIDGE_ALPHA_MIN_EXPONENT,
        RIDGE_ALPHA_MAX_EXPONENT,
        RIDGE_ALPHA_COUNT,
    ).tolist()


class RidgeClassifierArgs(BaseModel):
    """Configuration for validation-selected standardized Ridge classification."""

    alphas: list[float] = Field(default_factory=get_default_ridge_alphas)
    selection_metric: Literal[
        "balanced_accuracy",
        "accuracy",
        "f1_weighted",
    ] = "balanced_accuracy"

    @field_validator("alphas")
    @classmethod
    def validate_alphas(cls, alphas: list[float]) -> list[float]:
        """Require a non-empty, finite, unique positive alpha grid."""
        if not alphas:
            raise ValueError("classifier.alphas must contain at least one value.")
        if any(not np.isfinite(alpha) or alpha <= 0 for alpha in alphas):
            raise ValueError(
                "classifier.alphas must contain finite positive values."
            )
        if len(alphas) != len(set(alphas)):
            raise ValueError("classifier.alphas must not contain duplicates.")
        return alphas


class FeatureClassifier(ABC):
    """Fit a classifier from features and expose prediction scores."""

    @abstractmethod
    def fit(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        validation_features: np.ndarray,
        validation_labels: np.ndarray,
    ) -> "FeatureClassifier":
        """Fit classifier candidates and select state using validation data."""

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict labels from extracted features."""

    @abstractmethod
    def decision_function(self, features: np.ndarray) -> np.ndarray:
        """Return classification scores from extracted features."""

    @property
    @abstractmethod
    def classes_(self) -> np.ndarray:
        """Return fitted class labels in score-column order."""


class ValidationSelectedRidgeClassifier(FeatureClassifier):
    """Standardize feature columns and select Ridge alpha on validation data."""

    def __init__(self, args: RidgeClassifierArgs):
        self.args = args
        self.pipeline: Pipeline | None = None
        self.selected_alpha: float | None = None

    def fit(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        validation_features: np.ndarray,
        validation_labels: np.ndarray,
    ) -> "ValidationSelectedRidgeClassifier":
        """Fit all candidate pipelines and retain the best validation score."""
        best_score = -np.inf
        for alpha in sorted(self.args.alphas):
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("ridge", RidgeClassifier(alpha=alpha)),
                ]
            )
            pipeline.fit(train_features, train_labels)
            score = self._selection_score(
                validation_labels,
                pipeline.predict(validation_features),
            )
            if not np.isfinite(score):
                raise ValueError(
                    f"Validation {self.args.selection_metric} is NaN for "
                    f"Ridge alpha {alpha}."
                )
            if score > best_score:
                self.pipeline = pipeline
                self.selected_alpha = float(alpha)
                best_score = score
        if self.pipeline is None or self.selected_alpha is None:
            raise RuntimeError("No Ridge classifier candidate was fitted.")
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict labels using the selected standardized Ridge pipeline."""
        return self._require_pipeline().predict(features)

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        """Return decision values using the selected Ridge pipeline."""
        return self._require_pipeline().decision_function(features)

    @property
    def classes_(self) -> np.ndarray:
        """Return classes learned by the selected Ridge estimator."""
        return self._require_pipeline().named_steps["ridge"].classes_

    def _selection_score(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
    ) -> float:
        """Calculate the configured model-selection metric."""
        if self.args.selection_metric == "balanced_accuracy":
            return float(balanced_accuracy_score(labels, predictions))
        if self.args.selection_metric == "accuracy":
            return float(accuracy_score(labels, predictions))
        return float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        )

    def _require_pipeline(self) -> Pipeline:
        """Return the fitted sklearn pipeline or raise a clear error."""
        if self.pipeline is None:
            raise RuntimeError("Ridge classifier must be fitted before prediction.")
        return self.pipeline
