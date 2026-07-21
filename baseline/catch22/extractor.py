"""Canonical per-channel catch22 feature extraction for EEG trials."""

import numpy as np

from baseline.feature_extractor.extractor import EEGFeatureExtractor


CATCH22_FEATURE_COUNT = 22


class Catch22FeatureExtractor(EEGFeatureExtractor):
    """Extract canonical catch22 features independently for every channel."""

    def _fit(self, train_data: np.ndarray) -> None:
        """Retain no state because canonical catch22 is deterministic."""

    def _transform(self, data: np.ndarray) -> np.ndarray:
        """Concatenate 22 canonical features for each channel in order."""
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
                feature_start = channel_index * CATCH22_FEATURE_COUNT
                feature_end = feature_start + CATCH22_FEATURE_COUNT
                features[trial_index, feature_start:feature_end] = values
        return features
