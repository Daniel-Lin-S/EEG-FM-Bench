"""Composition root for external multivariate miniROCKET classification.

The upstream GPL-3.0 miniROCKET implementation is loaded from a user-provided
clone at runtime rather than copied into this Apache-2.0 repository.
"""

import numpy as np

from baseline.feature_extractor.classifier import (
    ValidationSelectedRidgeClassifier,
)
from baseline.feature_extractor.pipeline import FeatureExtractionPipeline
from baseline.feature_extractor.trainer import FeatureExtractorTrainer
from baseline.minirocket.extractor import MiniRocketFeatureExtractor
from baseline.minirocket.minirocket_config import MiniRocketConfig


class MiniRocketTrainer(FeatureExtractorTrainer):
    """Compose miniROCKET extraction with validation-selected Ridge."""

    def __init__(self, cfg: MiniRocketConfig):
        self.extractor = MiniRocketFeatureExtractor(
            cfg.model.extractor,
            cfg.seed,
        )
        pipeline = FeatureExtractionPipeline(
            self.extractor,
            ValidationSelectedRidgeClassifier(cfg.model.classifier),
        )
        super().__init__(cfg, pipeline)

    def fit_extractor(self, train_data: np.ndarray) -> None:
        """Fit miniROCKET parameters from float32 EEG training trials.

        Parameters
        ----------
        train_data : numpy.ndarray
            EEG array with shape ``(n_trials, n_channels, n_timepoints)`` and
            dtype ``float32``.
        """
        self.extractor.fit(train_data)

    def transform_features(self, data: np.ndarray) -> np.ndarray:
        """Transform float32 EEG trials with native parallel miniROCKET.

        Parameters
        ----------
        data : numpy.ndarray
            EEG array with shape ``(n_trials, n_channels, n_timepoints)`` and
            dtype ``float32``.

        Returns
        -------
        numpy.ndarray
            Feature matrix with shape ``(n_trials, n_features)``.
        """
        return self.extractor.transform(data)
