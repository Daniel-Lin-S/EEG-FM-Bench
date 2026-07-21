"""Per-channel canonical catch22 EEG feature extraction.

For each trial, all 22 canonical catch22 values are extracted from each EEG
channel and concatenated in channel order before Ridge classification.
"""

import logging

import numpy as np

from baseline.catch22.catch22_config import Catch22Config
from baseline.catch22.extractor import Catch22FeatureExtractor
from baseline.feature_extractor.classifier import ValidationSelectedRidgeClassifier
from baseline.feature_extractor.pipeline import FeatureExtractionPipeline
from baseline.feature_extractor.trainer import FeatureExtractorTrainer


logger = logging.getLogger("baseline")

CATCH22_FEATURE_COUNT = 22


class Catch22Trainer(FeatureExtractorTrainer):
    """Extract canonical catch22 values independently from EEG channels."""

    def __init__(self, cfg: Catch22Config):
        pipeline = FeatureExtractionPipeline(
            Catch22FeatureExtractor(),
            ValidationSelectedRidgeClassifier(cfg.model.classifier),
        )
        super().__init__(cfg, pipeline)

    def fit_extractor(self, train_data: np.ndarray) -> None:
        """Validate deterministic catch22 training data without fitting state.

        Parameters
        ----------
        train_data : numpy.ndarray
            EEG array with shape ``(n_trials, n_channels, n_timepoints)`` and
            dtype ``float32``.
        """
        if train_data.dtype != np.float32:
            raise ValueError(
                "catch22 expected float32 EEG data, but got "
                f"{train_data.dtype}."
            )

    def transform_features(self, data: np.ndarray) -> np.ndarray:
        """Concatenate 22 canonical catch22 values from each trial channel.

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
        try:
            from pycatch22 import catch22_all
        except ModuleNotFoundError as exc:
            raise ImportError(
                "catch22 requires pycatch22. Install it with: pip install "
                "-r requirements/feature_extractors.txt"
            ) from exc

        n_trials, n_channels, _ = data.shape
        features = np.empty(
            (n_trials, n_channels * CATCH22_FEATURE_COUNT),
            dtype=np.float64,
        )
        for trial_index in range(n_trials):
            for channel_index in range(n_channels):
                result = catch22_all(
                    data[trial_index, channel_index],
                    catch24=False,
                )
                values = np.asarray(result["values"], dtype=np.float64)
                if values.shape != (CATCH22_FEATURE_COUNT,):
                    raise ValueError(
                        "Expected 22 catch22 values for trial "
                        f"{trial_index}, channel {channel_index}, but got "
                        f"shape {values.shape}."
                    )
                features[
                    trial_index,
                    channel_index * CATCH22_FEATURE_COUNT:
                    (channel_index + 1) * CATCH22_FEATURE_COUNT,
                ] = values
        return features
