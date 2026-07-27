"""Tests for the label-only naive majority-vote baseline."""

import json
from pathlib import Path

import datasets
import numpy as np
import pytest

import baseline.naive.naive_trainer as naive_trainer_module
from baseline.naive.classifier import MajorityVoteClassifier
from baseline.naive.naive_config import NaiveConfig
from baseline.naive.naive_trainer import NaiveTrainer


DATASET_NAME = "toy"
DATASET_CONFIG = "finetune"
BINARY_CLASS_COUNT = 2
MULTICLASS_COUNT = 3


class LabelOnlyDataset:
    """Minimal dataset that proves the trainer never reads EEG signals."""

    column_names = ["label", "data"]

    def __init__(self, labels: list[int]):
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return self.labels.size

    def __getitem__(self, key: str) -> np.ndarray:
        if key != "label":
            raise AssertionError(
                "Naive baseline accessed forbidden column: "
                f"{key}"
            )
        return self.labels


def make_config(tmp_path: Path) -> NaiveConfig:
    """Build one minimal label-only baseline configuration."""
    return NaiveConfig(
        data={"datasets": {DATASET_NAME: DATASET_CONFIG}},
        logging={
            "run_dir": str(tmp_path),
            "use_cloud": False,
            "outputs": ["csv"],
        },
    )


def test_majority_prediction_and_training_probability() -> None:
    """Fit uses only unweighted training label counts."""
    classifier = MajorityVoteClassifier(42, DATASET_NAME).fit(
        np.array([0, 1, 1, 1], dtype=np.int64),
        BINARY_CLASS_COUNT,
    )

    assert classifier.predicted_class == 1
    np.testing.assert_array_equal(
        classifier.predict(3),
        np.array([1, 1, 1], dtype=np.int64),
    )
    np.testing.assert_allclose(
        classifier.positive_scores(2),
        np.array([0.75, 0.75]),
    )
    assert classifier.metadata()["class_counts"] == [1, 3]


def test_tied_majority_is_reproducible_per_dataset() -> None:
    """Exact ties use a stable dataset-specific seeded draw."""
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    first = MajorityVoteClassifier(42, DATASET_NAME).fit(
        labels,
        BINARY_CLASS_COUNT,
    )
    second = MajorityVoteClassifier(42, DATASET_NAME).fit(
        labels,
        BINARY_CLASS_COUNT,
    )

    assert first.predicted_class == second.predicted_class
    assert first.metadata()["tied_classes"] == [0, 1]


def test_training_requires_all_declared_classes() -> None:
    """Missing training classes fail instead of silently changing the task."""
    with pytest.raises(ValueError, match="Expected toy training labels"):
        MajorityVoteClassifier(42, DATASET_NAME).fit(
            np.array([0, 0], dtype=np.int64),
            BINARY_CLASS_COUNT,
        )


def test_binary_metrics_use_constant_training_probability(
    tmp_path: Path,
) -> None:
    """Binary AUROC and AUPR use a score inferred only from training labels."""
    trainer = NaiveTrainer(make_config(tmp_path))
    classifier = MajorityVoteClassifier(42, DATASET_NAME).fit(
        np.array([0, 1, 1, 1], dtype=np.int64),
        BINARY_CLASS_COUNT,
    )
    metrics = trainer._evaluate(
        classifier,
        np.array([0, 1], dtype=np.int64),
        DATASET_NAME,
        "test",
        BINARY_CLASS_COUNT,
    )

    assert metrics[f"{DATASET_NAME}/test/acc"] == 0.5
    assert metrics[f"{DATASET_NAME}/test/balanced_acc"] == 0.5
    assert metrics[f"{DATASET_NAME}/test/auroc"] == 0.5
    assert metrics[f"{DATASET_NAME}/test/auc_pr"] == 0.5


def test_multiclass_metrics_are_finite(tmp_path: Path) -> None:
    """Multiclass results include kappa and weighted F1."""
    trainer = NaiveTrainer(make_config(tmp_path))
    classifier = MajorityVoteClassifier(42, DATASET_NAME).fit(
        np.array([0, 1, 2, 2], dtype=np.int64),
        MULTICLASS_COUNT,
    )
    metrics = trainer._evaluate(
        classifier,
        np.array([0, 1, 2], dtype=np.int64),
        DATASET_NAME,
        "test",
        MULTICLASS_COUNT,
    )

    assert set(metrics) == {
        f"{DATASET_NAME}/test/acc",
        f"{DATASET_NAME}/test/balanced_acc",
        f"{DATASET_NAME}/test/cohen_kappa",
        f"{DATASET_NAME}/test/f1",
    }
    assert all(np.isfinite(value) for value in metrics.values())


def test_run_reads_only_labels_and_writes_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete run does not request a signal column and writes artifacts."""
    split_labels = {
        datasets.Split.TRAIN: [0, 1, 1, 1],
        datasets.Split.VALIDATION: [0, 1],
        datasets.Split.TEST: [0, 1, 1],
    }

    def fake_loader(
        dataset_names: list[str],
        builder_configs: list[str],
        split: datasets.NamedSplit,
        **_: object,
    ) -> tuple[LabelOnlyDataset, list[object]]:
        assert dataset_names == [DATASET_NAME]
        assert builder_configs == [DATASET_CONFIG]
        return LabelOnlyDataset(split_labels[split]), []

    monkeypatch.setattr(
        naive_trainer_module,
        "load_concat_eeg_datasets",
        fake_loader,
    )
    monkeypatch.setattr(
        naive_trainer_module,
        "get_dataset_n_class",
        lambda *_: BINARY_CLASS_COUNT,
    )
    trainer = NaiveTrainer(make_config(tmp_path))

    trainer.run()

    root = next((tmp_path / "log" / "baseline" / "naive").iterdir())
    completion_path = root / "datasets" / DATASET_NAME / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert completion["majority_vote"]["class_counts"] == [1, 3]
    assert completion["majority_vote"]["predicted_class"] == 1
    assert (root / "summary" / "test_summary.csv").is_file()
