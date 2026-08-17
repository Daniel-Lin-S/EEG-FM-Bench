"""Batched Arrow input for classical EEG feature extractors.

The reader returns aligned NumPy batches and labels. It intentionally avoids
the neural loader's Torch formatting, class weighting, cached casting, and
dataset concatenation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import datasets
import numpy as np

from data.processor.wrapper import load_eeg_dataset_for_classical_ml


@dataclass(frozen=True)
class AlignedEEGBatch:
    """One aligned classical EEG input batch.

    Parameters
    ----------
    data : numpy.ndarray
        Float32 EEG with shape ``(batch, channels, timepoints)``.
    labels : numpy.ndarray
        Int64 labels with shape ``(batch,)``.
    start : int
        Inclusive split row offset.
    stop : int
        Exclusive split row offset.
    """

    data: np.ndarray
    labels: np.ndarray
    start: int
    stop: int


class FeatureSplitReader:
    """Stream one Arrow split as fixed-layout NumPy batches."""

    def __init__(
        self,
        dataset_name: str,
        dataset_config: str,
        split: datasets.NamedSplit,
        fs: int,
        batch_size: int,
        n_class: int,
        channel_layout: list[str],
        montages: dict[str, list[str]],
        expected_trial_shape: Optional[tuple[int, int]] = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        if batch_size <= 0:
            raise ValueError(
                f"Expected a positive Arrow batch size, but got "
                f"{batch_size}."
            )
        if n_class <= 1:
            raise ValueError(
                f"Expected at least two classes, but got {n_class}."
            )
        if not channel_layout:
            raise ValueError("Classical EEG channel layout must not be empty.")
        self.batch_size = batch_size
        self.n_class = n_class
        self.trial_shape = expected_trial_shape
        self.dataset = load_eeg_dataset_for_classical_ml(
            dataset_name,
            dataset_config,
            split,
            fs,
        )
        if len(self.dataset) == 0:
            raise ValueError(
                f"{dataset_name} {split} split contains no trials."
            )
        self._selectors = self._build_selectors(channel_layout, montages)
        self._source_channel_counts = {
            montage: len(source_layout)
            for montage, source_layout in montages.items()
        }
        self._channel_count = len(channel_layout)

    def __len__(self) -> int:
        """Return the number of trials in this split."""
        return len(self.dataset)

    def batches(self) -> Iterator[AlignedEEGBatch]:
        """Yield validated NumPy batches without retaining earlier batches."""
        formatted = self.dataset.with_format(
            "numpy",
            columns=["data", "label", "montage"],
        )
        for start in range(0, len(formatted), self.batch_size):
            stop = min(start + self.batch_size, len(formatted))
            batch = formatted[start:stop]
            labels = self._labels(batch["label"], stop - start)
            data = self._align_batch(batch["data"], batch["montage"], start)
            yield AlignedEEGBatch(data, labels, start, stop)

    def _build_selectors(
        self,
        channel_layout: list[str],
        montages: dict[str, list[str]],
    ) -> dict[str, np.ndarray]:
        """Build reusable source-to-shared channel selectors."""
        selectors: dict[str, np.ndarray] = {}
        for montage, source_layout in montages.items():
            missing_channels = [
                channel
                for channel in channel_layout
                if channel not in source_layout
            ]
            if missing_channels:
                raise ValueError(
                    f"Montage {montage} is missing shared channels "
                    f"{missing_channels}."
                )
            selectors[montage] = np.asarray(
                [source_layout.index(channel) for channel in channel_layout],
                dtype=np.intp,
            )
        return selectors

    def _labels(self, values: object, expected_rows: int) -> np.ndarray:
        """Validate and return one int64 label batch."""
        labels = np.asarray(values)
        if labels.shape != (expected_rows,):
            raise ValueError(
                f"Expected {self.dataset_name} {self.split} labels with shape "
                f"({expected_rows},), but got {labels.shape}."
            )
        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError(
                f"Expected {self.dataset_name} {self.split} integer labels, "
                f"but got dtype {labels.dtype}."
            )
        invalid_labels = labels[(labels < 0) | (labels >= self.n_class)]
        if invalid_labels.size:
            raise ValueError(
                f"Expected {self.dataset_name} {self.split} labels in "
                f"[0, {self.n_class}), but got "
                f"{np.unique(invalid_labels).tolist()}."
            )
        return labels.astype(np.int64, copy=False)

    def _align_batch(
        self,
        values: object,
        montage_values: object,
        start: int,
    ) -> np.ndarray:
        """Align one possibly mixed-montage batch to the shared layout."""
        montage_batch = np.asarray(montage_values)
        if montage_batch.ndim != 1:
            raise ValueError(
                f"Expected {self.dataset_name} {self.split} montage values "
                f"with shape (rows,), but got {montage_batch.shape}."
            )
        row_count = len(montage_batch)
        try:
            source_row_count = len(values)
        except TypeError as exc:
            raise ValueError(
                f"Expected {self.dataset_name} {self.split} batched EEG "
                "rows."
            ) from exc
        if source_row_count != row_count:
            raise ValueError(
                f"Expected {self.dataset_name} {self.split} EEG and montage "
                f"rows to match, but got {source_row_count} and "
                f"{row_count}."
            )
        aligned: Optional[np.ndarray] = None
        for offset in range(row_count):
            montage = str(montage_batch[offset])
            selector = self._selectors.get(montage)
            if selector is None:
                raise ValueError(
                    f"{self.dataset_name} {self.split} has unknown montage "
                    f"{montage}."
                )
            source = np.asarray(values[offset])
            expected_channels = self._source_channel_counts[montage]
            if source.ndim != 2 or source.shape[0] != expected_channels:
                raise ValueError(
                    f"Expected {montage} data shape ({expected_channels}, "
                    f"timepoints), but got {source.shape} at row "
                    f"{start + offset}."
                )
            if source.dtype != np.float32:
                raise TypeError(
                    f"Expected {montage} float32 EEG data, but got dtype "
                    f"{source.dtype} at row {start + offset}."
                )
            if not np.isfinite(source).all():
                raise ValueError(
                    f"{self.dataset_name} {self.split} EEG data contains "
                    f"NaN or inf at row {start + offset}."
                )
            trial_shape = (self._channel_count, source.shape[1])
            if self.trial_shape is None:
                self.trial_shape = trial_shape
            if trial_shape != self.trial_shape:
                raise ValueError(
                    f"Expected {self.dataset_name} {self.split} aligned EEG "
                    f"shape {self.trial_shape}, but got {trial_shape} at row "
                    f"{start + offset}."
                )
            if aligned is None:
                aligned = np.empty(
                    (row_count, *self.trial_shape),
                    dtype=np.float32,
                )
            aligned[offset] = source[selector]
        if aligned is None:
            raise ValueError(
                f"{self.dataset_name} {self.split} produced an empty batch."
            )
        return aligned
