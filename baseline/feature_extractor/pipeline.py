"""Compose independent EEG feature extractors and feature classifiers."""

from dataclasses import dataclass

import numpy as np

from baseline.feature_extractor.classifier import FeatureClassifier
from baseline.feature_extractor.extractor import EEGFeatureExtractor


@dataclass(frozen=True)
class FeatureExtractionFitResult:
    """Feature matrices retained after fitting one extraction pipeline.

    Parameters
    ----------
    validation_features : numpy.ndarray
        Validation feature matrix with shape ``(n_trials, n_features)``.
    """

    validation_features: np.ndarray


class FeatureExtractionPipeline:
    """Fit feature extraction and downstream classification in sequence."""

    def __init__(
        self,
        extractor: EEGFeatureExtractor,
        classifier: FeatureClassifier,
    ):
        self.extractor = extractor
        self.classifier = classifier

    def fit(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        validation_data: np.ndarray,
        validation_labels: np.ndarray,
    ) -> FeatureExtractionFitResult:
        """Fit extraction/classification and retain validation features.

        Parameters
        ----------
        train_data : numpy.ndarray
            Training EEG with shape ``(n_trials, n_channels, n_timepoints)``.
        train_labels : numpy.ndarray
            Training labels with shape ``(n_trials,)``.
        validation_data : numpy.ndarray
            Validation EEG with shape ``(n_trials, n_channels, n_timepoints)``.
        validation_labels : numpy.ndarray
            Validation labels with shape ``(n_trials,)``.

        Returns
        -------
        FeatureExtractionFitResult
            Cached validation features used for alpha selection and final
            validation metrics.
        """
        self.extractor.fit(train_data)
        train_features = self._extract(train_data, "training")
        validation_features = self._extract(validation_data, "validation")
        self.classifier.fit(
            train_features,
            train_labels,
            validation_features,
            validation_labels,
        )
        return FeatureExtractionFitResult(validation_features)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Extract and validate features without invoking the classifier."""
        return self._extract(data, "transformation")

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Extract features and predict labels for EEG data."""
        return self.classifier.predict(self._extract(data, "prediction"))

    def decision_function(self, data: np.ndarray) -> np.ndarray:
        """Extract features and return classifier decision scores."""
        return self.classifier.decision_function(self._extract(data, "scoring"))

    def close(self) -> None:
        """Release runtime resources held by the feature extractor."""
        self.extractor.close()

    def _extract(self, data: np.ndarray, split_name: str) -> np.ndarray:
        """Extract and validate one dense numeric feature matrix."""
        features = np.asarray(self.extractor.transform(data))
        if not np.issubdtype(features.dtype, np.number):
            raise TypeError(
                f"Expected {split_name} numeric features, but got dtype "
                f"{features.dtype}."
            )
        if features.ndim != 2:
            raise ValueError(
                f"Expected {split_name} feature shape (trials, features), "
                f"but got {features.shape}."
            )
        if features.shape[0] != data.shape[0] or features.shape[1] == 0:
            raise ValueError(
                f"Expected {split_name} features with shape "
                f"({data.shape[0]}, n_features > 0), but got "
                f"{features.shape}."
            )
        if not np.isfinite(features).all():
            raise ValueError(
                f"{split_name} extracted features contain NaN or inf."
            )
        return features
