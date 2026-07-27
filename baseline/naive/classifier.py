"""Label-distribution majority classifier with no EEG input.

The classifier fits one dataset by counting labels in its training split. It
returns a constant class prediction and, for binary metrics, a constant score
equal to the observed training positive-class frequency.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


HASH_BYTES = 8
POSITIVE_CLASS_INDEX = 1


@dataclass
class MajorityVoteClassifier:
    """Fit a constant class prediction from one training label vector.

    Parameters
    ----------
    seed : int
        Configured run seed used only for exact-majority ties.
    dataset_name : str
        Dataset identifier included in the tie seed so configured dataset
        ordering cannot influence the result.
    """

    seed: int
    dataset_name: str
    class_counts: np.ndarray = field(init=False)
    class_probabilities: np.ndarray = field(init=False)
    predicted_class: int = field(init=False)
    tied_classes: np.ndarray = field(init=False)

    def fit(
        self,
        train_labels: np.ndarray,
        n_class: int,
    ) -> "MajorityVoteClassifier":
        """Count training labels and resolve a constant prediction.

        Parameters
        ----------
        train_labels : numpy.ndarray
            One-dimensional integer training labels with shape
            ``(n_trials,)``.
        n_class : int
            Expected number of contiguous classes.

        Returns
        -------
        MajorityVoteClassifier
            Fitted classifier.
        """
        _validate_labels(train_labels, n_class, self.dataset_name, "training")
        self.class_counts = np.bincount(train_labels, minlength=n_class)
        self.class_probabilities = self.class_counts / train_labels.size
        maximum_count = int(self.class_counts.max())
        self.tied_classes = np.flatnonzero(self.class_counts == maximum_count)
        if self.tied_classes.size == 1:
            self.predicted_class = int(self.tied_classes[0])
            return self
        rng = np.random.default_rng(self._tie_seed())
        self.predicted_class = int(rng.choice(self.tied_classes))
        return self

    def predict(self, n_trials: int) -> np.ndarray:
        """Return the fitted constant label for every trial.

        Parameters
        ----------
        n_trials : int
            Number of validation or test examples to classify.

        Returns
        -------
        numpy.ndarray
            Integer predictions with shape ``(n_trials,)``.
        """
        if n_trials <= 0:
            raise ValueError(
                "MajorityVoteClassifier requires a positive prediction "
                f"count, but got {n_trials}."
            )
        return np.full(n_trials, self.predicted_class, dtype=np.int64)

    def positive_scores(self, n_trials: int) -> np.ndarray:
        """Return constant binary positive-class probabilities.

        Parameters
        ----------
        n_trials : int
            Number of validation or test examples to score.

        Returns
        -------
        numpy.ndarray
            Positive-class scores with shape ``(n_trials,)``.
        """
        if self.class_probabilities.size != 2:
            raise ValueError(
                "Binary positive-class scores require exactly two classes, "
                f"but got {self.class_probabilities.size}."
            )
        return np.full(
            n_trials,
            self.class_probabilities[POSITIVE_CLASS_INDEX],
            dtype=np.float64,
        )

    def metadata(self) -> dict[str, object]:
        """Return JSON-compatible fitted distribution details."""
        return {
            "class_counts": self.class_counts.astype(int).tolist(),
            "class_probabilities": self.class_probabilities.tolist(),
            "predicted_class": self.predicted_class,
            "tied_classes": self.tied_classes.astype(int).tolist(),
            "tie_breaker": "seeded_uniform",
        }

    def _tie_seed(self) -> int:
        """Derive an order-independent random seed for one dataset tie."""
        payload = f"{self.seed}:{self.dataset_name}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:HASH_BYTES], byteorder="big")


def _validate_labels(
    labels: np.ndarray,
    n_class: int,
    dataset_name: str,
    split_name: str,
) -> None:
    """Require one non-empty vector of contiguous integer class labels."""
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError(
            f"Expected non-empty {dataset_name} {split_name} labels with "
            f"shape (n_trials,), but got {labels.shape}."
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            f"Expected integer {dataset_name} {split_name} labels, but got "
            f"dtype {labels.dtype}."
        )
    if np.issubdtype(labels.dtype, np.bool_):
        raise ValueError(
            f"Expected integer {dataset_name} {split_name} labels, but got "
            "boolean labels."
        )
    if n_class <= 1:
        raise ValueError(
            f"naive supports classification with at least two classes, but "
            f"{dataset_name} declares {n_class}."
        )
    expected_classes = np.arange(n_class)
    observed_classes = np.unique(labels)
    if split_name == "training" and not np.array_equal(
        observed_classes,
        expected_classes,
    ):
        raise ValueError(
            f"Expected {dataset_name} training labels "
            f"{expected_classes.tolist()}, but got "
            f"{observed_classes.tolist()}."
        )
    if labels.min() < 0 or labels.max() >= n_class:
        raise ValueError(
            f"Expected {dataset_name} {split_name} labels in [0, "
            f"{n_class - 1}], but got {observed_classes.tolist()}."
        )
