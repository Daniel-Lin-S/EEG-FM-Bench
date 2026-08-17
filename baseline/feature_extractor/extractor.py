"""Interfaces for transforming EEG trials into fixed-width features.

Extractors receive raw EEG arrays with shape ``(trials, channels, time)``.
They may learn state from the training split, but must not access labels.
"""

from abc import ABC, abstractmethod

import numpy as np


EEG_FINITE_CHECK_BATCH_SIZE = 256


class EEGFeatureExtractor(ABC):
    """Convert batches of EEG trials into one feature vector per trial."""

    @property
    def requires_random_access_training_data(self) -> bool:
        """Return whether fitting requires the complete training EEG array.

        Stateless or genuinely streaming extractors should override this
        property with ``False``. The trainer otherwise supplies a temporary
        disk-backed C-contiguous array rather than a RAM-resident split.
        """
        return True

    def fit(self, train_data: np.ndarray) -> "EEGFeatureExtractor":
        """Fit state from training EEG data.

        Parameters
        ----------
        train_data : numpy.ndarray
            EEG with shape ``(n_trials, n_channels, n_timepoints)`` and dtype
            ``float32``.

        Returns
        -------
        EEGFeatureExtractor
            This fitted extractor.
        """
        self._validate_eeg(train_data, "training")
        self._fit(train_data)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Extract fixed-width features from EEG data.

        Parameters
        ----------
        data : numpy.ndarray
            EEG with shape ``(n_trials, n_channels, n_timepoints)`` and dtype
            ``float32``.

        Returns
        -------
        numpy.ndarray
            Features with shape ``(n_trials, n_features)``.
        """
        self._validate_eeg(data, "input")
        return self._transform(data)

    def close(self) -> None:
        """Release optional runtime resources owned by the extractor."""

    @staticmethod
    def _validate_eeg(data: np.ndarray, name: str) -> None:
        """Validate one dense EEG batch before extractor use."""
        if data.ndim != 3:
            raise ValueError(
                f"Expected {name} EEG shape (trials, channels, timepoints), "
                f"but got {data.shape}."
            )
        if data.dtype != np.float32:
            raise ValueError(
                f"Expected {name} EEG dtype float32, but got {data.dtype}."
            )
        if len(data) == 0:
            raise ValueError(f"Expected non-empty {name} EEG data.")
        for start in range(0, len(data), EEG_FINITE_CHECK_BATCH_SIZE):
            stop = min(start + EEG_FINITE_CHECK_BATCH_SIZE, len(data))
            if not np.isfinite(data[start:stop]).all():
                raise ValueError(f"{name} EEG data contains NaN or inf.")

    @abstractmethod
    def _fit(self, train_data: np.ndarray) -> None:
        """Fit extractor-specific state from validated training EEG."""

    @abstractmethod
    def _transform(self, data: np.ndarray) -> np.ndarray:
        """Transform validated EEG into a feature matrix."""
