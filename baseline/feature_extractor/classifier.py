"""Downstream feature classifiers independent of EEG feature extraction."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Literal, Mapping, Optional

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RIDGE_ALPHA_MIN_EXPONENT = -3
RIDGE_ALPHA_MAX_EXPONENT = 3
RIDGE_ALPHA_COUNT = 10
RIDGE_RUNTIME_OVERHEAD_BYTES = 2 * (1 << 30)


def get_default_ridge_alphas() -> list[float]:
    """Return the default Ridge regularization candidates."""
    return np.logspace(
        RIDGE_ALPHA_MIN_EXPONENT,
        RIDGE_ALPHA_MAX_EXPONENT,
        RIDGE_ALPHA_COUNT,
    ).tolist()


class RidgeLogspaceArgs(BaseModel):
    """Parameters for a base-10 logarithmic Ridge alpha grid.

    Parameters
    ----------
    start : float
        Base-10 exponent of the first alpha.
    stop : float
        Base-10 exponent of the last alpha.
    num : int
        Number of alpha candidates, including both endpoints.
    """

    model_config = ConfigDict(extra="forbid")

    start: float
    stop: float
    num: int

    @field_validator("start", "stop")
    @classmethod
    def validate_finite_exponent(cls, value: float) -> float:
        """Require a finite log-space exponent."""
        if not np.isfinite(value):
            raise ValueError("classifier.logspace exponents must be finite.")
        return value

    @field_validator("num", mode="before")
    @classmethod
    def reject_boolean_num(cls, value: Any) -> Any:
        """Reject booleans masquerading as a log-space candidate count."""
        if isinstance(value, bool):
            raise ValueError("classifier.logspace.num must not be a boolean.")
        return value

    @field_validator("num")
    @classmethod
    def validate_num(cls, value: int) -> int:
        """Require at least one log-space candidate."""
        if value <= 0:
            raise ValueError("classifier.logspace.num must be positive.")
        return value

    def build_alphas(self) -> list[float]:
        """Expand the declarative grid to the exact fitted alpha values."""
        alphas = np.logspace(self.start, self.stop, self.num)
        if not np.isfinite(alphas).all():
            raise ValueError(
                "classifier.logspace produces alpha values outside the "
                "finite floating-point range."
            )
        return alphas.tolist()


class RidgeClassifierArgs(BaseModel):
    """Configuration for validation-selected standardized Ridge.

    Provide either ``alphas`` or ``logspace`` as input, never both. A
    log-space specification is expanded to ``alphas`` before persistence and
    identity calculation, so equivalent forms have the same semantics.
    """

    model_config = ConfigDict(extra="forbid")

    alphas: list[float] = Field(default_factory=get_default_ridge_alphas)
    selection_metric: Literal[
        "balanced_accuracy",
        "accuracy",
        "f1_weighted",
    ] = "balanced_accuracy"

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, value: Any) -> Any:
        """Normalize historical names and expand a declared alpha grid."""
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        aliases = {
            "ridge_alphas": "alphas",
            "ridge_selection_metric": "selection_metric",
        }
        for legacy_field, canonical_field in aliases.items():
            if legacy_field not in normalized:
                continue
            legacy_value = normalized.pop(legacy_field)
            if (
                canonical_field in normalized
                and normalized[canonical_field] != legacy_value
            ):
                raise ValueError(
                    f"Conflicting classifier values exist for "
                    f"{legacy_field!r} and {canonical_field!r}."
                )
            normalized.setdefault(canonical_field, legacy_value)
        has_alphas = "alphas" in normalized
        has_logspace = "logspace" in normalized
        if has_alphas and has_logspace:
            raise ValueError(
                "classifier.alphas and classifier.logspace are mutually "
                "exclusive."
            )
        if has_logspace:
            logspace = RidgeLogspaceArgs.model_validate(
                normalized.pop("logspace")
            )
            normalized["alphas"] = logspace.build_alphas()
        return normalized

    @field_validator("alphas")
    @classmethod
    def validate_alphas(cls, alphas: list[float]) -> list[float]:
        """Require a non-empty, finite, unique positive alpha grid."""
        if not alphas:
            raise ValueError(
                "classifier.alphas must contain at least one value."
            )
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

    def __init__(
        self,
        args: RidgeClassifierArgs,
        batch_size: int = 1024,
    ):
        if batch_size <= 0:
            raise ValueError(
                f"Expected a positive classifier batch size, but got "
                f"{batch_size}."
            )
        self.args = args
        self.batch_size = batch_size
        self.pipeline: Pipeline | None = None
        self.selected_alpha: float | None = None
        self._memory_check: Optional[Callable[[str, int], None]] = None

    def configure_memory_check(
        self,
        memory_check: Optional[Callable[[str, int], None]],
    ) -> None:
        """Configure an invocation-local pre-allocation callback."""
        self._memory_check = memory_check

    def fit(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        validation_features: np.ndarray,
        validation_labels: np.ndarray,
    ) -> "ValidationSelectedRidgeClassifier":
        """Fit one scaler and select a Ridge alpha on scaled features."""
        self._validate_feature_pair(train_features, validation_features)
        self._check_ridge_workspace(train_features, validation_features)
        scaler, scaled_train_features = self._scale_training_features(
            train_features
        )
        best_ridge = None
        best_score = -np.inf
        for alpha in sorted(self.args.alphas):
            ridge = RidgeClassifier(alpha=alpha)
            ridge.fit(scaled_train_features, train_labels)
            score = self._selection_score(
                validation_labels,
                self._predict_with_scaler(
                    scaler,
                    ridge,
                    validation_features,
                ),
            )
            if not np.isfinite(score):
                raise ValueError(
                    f"Validation {self.args.selection_metric} is NaN for "
                    f"Ridge alpha {alpha}."
                )
            if score > best_score:
                best_ridge = ridge
                self.selected_alpha = float(alpha)
                best_score = score
        if best_ridge is None or self.selected_alpha is None:
            raise RuntimeError("No Ridge classifier candidate was fitted.")
        self.pipeline = Pipeline(
            [("scaler", scaler), ("ridge", best_ridge)]
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict labels using the selected standardized Ridge pipeline."""
        pipeline = self._require_pipeline()
        return self._predict_with_scaler(
            pipeline.named_steps["scaler"],
            pipeline.named_steps["ridge"],
            features,
        )

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        """Return decision values using the selected Ridge pipeline."""
        pipeline = self._require_pipeline()
        outputs = []
        for feature_batch in self._feature_batches(features):
            scaled = pipeline.named_steps["scaler"].transform(
                feature_batch,
                copy=True,
            )
            outputs.append(
                np.asarray(
                    pipeline.named_steps["ridge"].decision_function(scaled)
                )
            )
        return np.concatenate(outputs, axis=0)

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
            f1_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
        )

    def _scale_training_features(
        self,
        train_features: np.ndarray,
    ) -> tuple[StandardScaler, np.ndarray]:
        """Fit once and standardize a memmap in-place in bounded chunks."""
        if isinstance(train_features, np.memmap):
            if not train_features.flags.writeable:
                raise ValueError(
                    "Training feature memmap must be writable for bounded "
                    "in-place standardization."
                )
            scaler = StandardScaler(copy=False)
            scaler.fit(train_features)
            for feature_batch in self._feature_batches(train_features):
                scaler.transform(feature_batch, copy=False)
            scaler.copy = True
            return scaler, train_features
        scaler = StandardScaler()
        return scaler, scaler.fit_transform(train_features)

    def _predict_with_scaler(
        self,
        scaler: StandardScaler,
        ridge: RidgeClassifier,
        features: np.ndarray,
    ) -> np.ndarray:
        """Scale and classify one feature matrix in bounded row batches."""
        predictions = []
        for feature_batch in self._feature_batches(features):
            scaled = scaler.transform(feature_batch, copy=True)
            predictions.append(np.asarray(ridge.predict(scaled)))
        return np.concatenate(predictions, axis=0)

    def _feature_batches(self, features: np.ndarray):
        """Yield non-empty row batches from one feature matrix."""
        if features.ndim != 2 or len(features) == 0:
            raise ValueError(
                "Expected a non-empty feature matrix with shape "
                f"(trials, features), but got {features.shape}."
            )
        for start in range(0, len(features), self.batch_size):
            yield features[start:start + self.batch_size]

    def _validate_feature_pair(
        self,
        train_features: np.ndarray,
        validation_features: np.ndarray,
    ) -> None:
        """Require compatible finite training and validation features."""
        if train_features.ndim != 2 or validation_features.ndim != 2:
            raise ValueError(
                "Training and validation features must both be matrices."
            )
        if (
            len(train_features) == 0
            or len(validation_features) == 0
            or train_features.shape[1] == 0
        ):
            raise ValueError(
                "Training and validation feature matrices must be non-empty."
            )
        if train_features.shape[1] != validation_features.shape[1]:
            raise ValueError(
                "Expected matching training and validation feature widths, "
                f"but got {train_features.shape[1]} and "
                f"{validation_features.shape[1]}."
            )
        if not self._features_are_finite(train_features):
            raise ValueError("Training features contain NaN or inf.")
        if not self._features_are_finite(validation_features):
            raise ValueError("Validation features contain NaN or inf.")

    def _features_are_finite(self, features: np.ndarray) -> bool:
        """Check finite feature values without a whole-matrix temporary."""
        return all(
            np.isfinite(feature_batch).all()
            for feature_batch in self._feature_batches(features)
        )

    def _check_ridge_workspace(
        self,
        train_features: np.ndarray,
        validation_features: np.ndarray,
    ) -> None:
        """Preflight Ridge's dense copy and normal-equation workspace."""
        if self._memory_check is None:
            return
        n_samples, n_features = train_features.shape
        itemsize = train_features.dtype.itemsize
        system_dimension = min(n_samples, n_features)
        solver_bytes = system_dimension * system_dimension * itemsize
        validation_rows = min(self.batch_size, len(validation_features))
        validation_bytes = validation_rows * n_features * itemsize
        additional_bytes = (
            int(train_features.nbytes)
            + solver_bytes
            + validation_bytes
            + RIDGE_RUNTIME_OVERHEAD_BYTES
        )
        self._memory_check("fit Ridge classifier", additional_bytes)

    def _require_pipeline(self) -> Pipeline:
        """Return the fitted sklearn pipeline or raise a clear error."""
        if self.pipeline is None:
            raise RuntimeError(
                "Ridge classifier must be fitted before prediction."
            )
        return self.pipeline
