"""Compose independent EEG feature extractors and feature classifiers."""

import numpy as np

from baseline.feature_extractor.classifier import FeatureClassifier
from baseline.feature_extractor.extractor import EEGFeatureExtractor


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
    ) -> "FeatureExtractionPipeline":
        """Fit extractor on raw training data then classifier on features."""
        self.extractor.fit(train_data)
        train_features = self._extract(train_data, "training")
        validation_features = self._extract(validation_data, "validation")
        self.classifier.fit(
            train_features,
            train_labels,
            validation_features,
            validation_labels,
        )
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Extract and validate features without invoking the classifier."""
        return self._extract(data, "transformation")

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Extract features and predict labels for EEG data."""
        return self.classifier.predict(self._extract(data, "prediction"))

    def decision_function(self, data: np.ndarray) -> np.ndarray:
        """Extract features and return classifier decision scores."""
        return self.classifier.decision_function(self._extract(data, "scoring"))

    def _extract(self, data: np.ndarray, split_name: str) -> np.ndarray:
        """Extract and validate a dense feature matrix."""
        features = np.asarray(self.extractor.transform(data), dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(
                f"Expected {split_name} feature shape (trials, features), "
                f"but got {features.shape}."
            )
        if features.shape[0] != data.shape[0] or features.shape[1] == 0:
            raise ValueError(
                f"Expected {split_name} features with shape "
                f"({data.shape[0]}, n_features > 0), but got {features.shape}."
            )
        if not np.isfinite(features).all():
            raise ValueError(
                f"{split_name} extracted features contain NaN or inf."
            )
        return features
