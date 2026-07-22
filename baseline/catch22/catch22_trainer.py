"""Composition root for canonical per-channel catch22 classification."""

import numpy as np

from baseline.catch22.catch22_config import Catch22Config
from baseline.catch22.extractor import Catch22FeatureExtractor
from baseline.feature_extractor.classifier import (
    ValidationSelectedRidgeClassifier,
)
from baseline.feature_extractor.pipeline import FeatureExtractionPipeline
from baseline.feature_extractor.trainer import FeatureExtractorTrainer


class Catch22Trainer(FeatureExtractorTrainer):
    """Compose catch22 extraction with validation-selected Ridge.

    The classifier receives the deterministic per-channel feature matrix.
    """

    def __init__(self, cfg: Catch22Config):
        self.extractor = Catch22FeatureExtractor(cfg.model.extractor.n_jobs)
        pipeline = FeatureExtractionPipeline(
            self.extractor,
            ValidationSelectedRidgeClassifier(cfg.model.classifier),
        )
        super().__init__(cfg, pipeline)

    def fit_extractor(self, train_data: np.ndarray) -> None:
        """Fit deterministic catch22 state from float32 EEG trials.

        Parameters
        ----------
        train_data : numpy.ndarray
            EEG array with shape ``(n_trials, n_channels, n_timepoints)`` and
            dtype ``float32``.
        """
        self.extractor.fit(train_data)

    def transform_features(self, data: np.ndarray) -> np.ndarray:
        """Transform float32 EEG trials with the configured catch22 extractor.

        Parameters
        ----------
        data : numpy.ndarray
            EEG array with shape ``(n_trials, n_channels, n_timepoints)`` and
            dtype ``float32``.

        Returns
        -------
        numpy.ndarray
            Feature matrix with shape ``(n_trials, 22 * n_channels)``.
        """
        return self.extractor.transform(data)
