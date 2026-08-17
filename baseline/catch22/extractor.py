"""Canonical per-channel catch22 feature extraction for EEG trials."""

import sys
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    wait,
)
from multiprocessing import get_context
from typing import Iterable, Optional

import numpy as np

from baseline.feature_extractor.extractor import EEGFeatureExtractor


CATCH22_FEATURE_COUNT = 22
CATCH22_PROGRESS_INTERVAL = 100
CATCH22_TRIAL_CHUNKSIZE = 8
CATCH22_PENDING_CHUNKS_PER_WORKER = 2


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


def _extract_catch22_chunk(
    indexed_chunk: tuple[int, np.ndarray],
) -> tuple[int, np.ndarray]:
    """Extract one bounded contiguous chunk of EEG trials.

    Parameters
    ----------
    indexed_chunk : tuple[int, numpy.ndarray]
        Starting trial index and float32 data with shape
        ``(chunk_trials, channels, timepoints)``.

    Returns
    -------
    tuple[int, numpy.ndarray]
        Starting index and float64 features with shape
        ``(chunk_trials, 22 * channels)``.
    """
    start_index, trials = indexed_chunk
    rows = [
        _extract_catch22_trial((start_index + offset, trial))[1]
        for offset, trial in enumerate(trials)
    ]
    return start_index, np.stack(rows, axis=0)


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
        self._executor: Optional[ProcessPoolExecutor] = None

    @property
    def requires_random_access_training_data(self) -> bool:
        """Return false because canonical catch22 has no fitted state."""
        return False

    def _fit(self, train_data: np.ndarray) -> None:
        """Retain no state because canonical catch22 is deterministic."""

    def _transform(self, data: np.ndarray) -> np.ndarray:
        """Concatenate 22 canonical features for each channel in order."""
        n_trials, n_channels, _ = data.shape
        features = np.empty(
            (n_trials, n_channels * CATCH22_FEATURE_COUNT),
            dtype=np.float64,
        )
        if self.n_jobs == 1:
            extracted_trials = map(_extract_catch22_trial, enumerate(data))
            self._collect_features(extracted_trials, features)
            return features

        try:
            self._collect_parallel_features(data, features)
        except BaseException:
            self.close()
            raise
        return features

    def close(self) -> None:
        """Stop the persistent spawn worker pool."""
        if self._executor is None:
            return
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._executor = None

    def _get_executor(self) -> ProcessPoolExecutor:
        """Return the persistent bounded spawn process pool."""
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.n_jobs,
                mp_context=get_context("spawn"),
            )
        return self._executor

    def _collect_parallel_features(
        self,
        data: np.ndarray,
        features: np.ndarray,
    ) -> None:
        """Submit bounded trial chunks and collect out-of-order results."""
        executor = self._get_executor()
        chunks = iter(self._indexed_chunks(data))
        pending: set[Future[tuple[int, np.ndarray]]] = set()
        max_pending = self.n_jobs * CATCH22_PENDING_CHUNKS_PER_WORKER

        def submit_next() -> bool:
            """Submit one chunk if input remains."""
            try:
                indexed_chunk = next(chunks)
            except StopIteration:
                return False
            pending.add(executor.submit(_extract_catch22_chunk, indexed_chunk))
            return True

        for _ in range(max_pending):
            if not submit_next():
                break

        completed_trials = 0
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                pending.remove(future)
                start_index, chunk_features = future.result()
                stop_index = start_index + len(chunk_features)
                features[start_index:stop_index] = chunk_features
                completed_trials += len(chunk_features)
                if completed_trials % CATCH22_PROGRESS_INTERVAL == 0:
                    _write_terminal_progress(
                        completed_trials,
                        features.shape[0],
                    )
                submit_next()
        _write_terminal_progress(
            features.shape[0],
            features.shape[0],
            final=True,
        )

    @staticmethod
    def _indexed_chunks(
        data: np.ndarray,
    ) -> Iterable[tuple[int, np.ndarray]]:
        """Yield bounded contiguous trial views with their start indices."""
        for start_index in range(0, len(data), CATCH22_TRIAL_CHUNKSIZE):
            stop_index = min(
                start_index + CATCH22_TRIAL_CHUNKSIZE,
                len(data),
            )
            yield start_index, data[start_index:stop_index]

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
                _write_terminal_progress(completed_trials, features.shape[0])
        _write_terminal_progress(
            features.shape[0],
            features.shape[0],
            final=True,
        )


def _write_terminal_progress(
    completed_trials: int,
    total_trials: int,
    final: bool = False,
) -> None:
    """Display extraction progress only for an interactive stderr terminal."""
    if not sys.stderr.isatty():
        return
    suffix = "\n" if final else ""
    sys.stderr.write(
        f"\rcatch22 extracted {completed_trials}/{total_trials} trials.{suffix}"
    )
    sys.stderr.flush()
