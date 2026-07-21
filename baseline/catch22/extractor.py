"""Canonical per-channel catch22 feature extraction for EEG trials."""

import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Iterable

import numpy as np

from baseline.feature_extractor.extractor import EEGFeatureExtractor


CATCH22_FEATURE_COUNT = 22
CATCH22_PROGRESS_INTERVAL = 100

logger = logging.getLogger("baseline")


def _extract_catch22_trial(
    indexed_trial: tuple[int, np.ndarray],
) -> tuple[int, np.ndarray]:
    """Extract all canonical catch22 features for one EEG trial.

    Parameters
    ----------
    indexed_trial : tuple[int, numpy.ndarray]
        Trial index and float32 EEG trial with shape
        ``(n_channels, n_timepoints)``.

    Returns
    -------
    tuple[int, numpy.ndarray]
        Trial index and a feature vector with shape
        ``(22 * n_channels,)``.
    """
    try:
        from pycatch22 import catch22_all
    except ModuleNotFoundError as exc:
        raise ImportError(
            "catch22 requires pycatch22. Install it with: pip install "
            "-r requirements/feature_extractors.txt"
        ) from exc

    trial_index, trial = indexed_trial
    n_channels = trial.shape[0]
    features = np.empty(n_channels * CATCH22_FEATURE_COUNT, dtype=np.float64)
    for channel_index, channel in enumerate(trial):
        result = catch22_all(channel, catch24=False)
        values = np.asarray(result["values"], dtype=np.float64)
        if values.shape != (CATCH22_FEATURE_COUNT,):
            raise ValueError(
                "Expected 22 catch22 values for trial "
                f"{trial_index}, channel {channel_index}, but got shape "
                f"{values.shape}."
            )
        feature_start = channel_index * CATCH22_FEATURE_COUNT
        feature_end = feature_start + CATCH22_FEATURE_COUNT
        features[feature_start:feature_end] = values
    return trial_index, features


class Catch22FeatureExtractor(EEGFeatureExtractor):
    """Extract canonical catch22 features independently for every channel.

    Parameters
    ----------
    n_jobs : int
        Number of CPU processes used to extract independent trials.
    """

    def __init__(self, n_jobs: int):
        if n_jobs <= 0:
            raise ValueError(
                f"Expected positive catch22 n_jobs, but got {n_jobs}."
            )
        self.n_jobs = n_jobs

    def _fit(self, train_data: np.ndarray) -> None:
        """Retain no state because canonical catch22 is deterministic."""

    def _transform(self, data: np.ndarray) -> np.ndarray:
        """Concatenate 22 canonical features for each channel in order."""
        n_trials, n_channels, _ = data.shape
        features = np.empty(
            (n_trials, n_channels * CATCH22_FEATURE_COUNT),
            dtype=np.float64,
        )
        indexed_trials = enumerate(data)
        if self.n_jobs == 1:
            extracted_trials = map(_extract_catch22_trial, indexed_trials)
            self._collect_features(extracted_trials, features)
            return features

        context = get_context("fork")
        with ProcessPoolExecutor(
            max_workers=self.n_jobs,
            mp_context=context,
        ) as executor:
            extracted_trials = executor.map(
                _extract_catch22_trial,
                indexed_trials,
                chunksize=1,
            )
            self._collect_features(extracted_trials, features)
        return features

    @staticmethod
    def _collect_features(
        extracted_trials: Iterable[tuple[int, np.ndarray]],
        features: np.ndarray,
    ) -> None:
        """Store completed trial features and report extraction progress."""
        for completed_trials, (trial_index, trial_features) in enumerate(
            extracted_trials,
            start=1,
        ):
            features[trial_index] = trial_features
            if completed_trials % CATCH22_PROGRESS_INTERVAL == 0:
                logger.info(
                    "catch22 extracted %d/%d trials.",
                    completed_trials,
                    features.shape[0],
                )
