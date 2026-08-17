"""Tests for bounded classical Arrow and memmap feature processing."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import datasets
import numpy as np
import pytest
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

import baseline.feature_extractor.data as data_module
import baseline.feature_extractor.runtime as runtime_module
import baseline.feature_extractor.storage as storage_module
import baseline.feature_extractor.trainer as trainer_module
from baseline.catch22.extractor import Catch22FeatureExtractor
from baseline.catch22.catch22_config import Catch22Config
from baseline.feature_extractor.classifier import (
    RidgeClassifierArgs,
    ValidationSelectedRidgeClassifier,
)
from baseline.feature_extractor.data import AlignedEEGBatch, FeatureSplitReader
from baseline.feature_extractor.extractor import EEGFeatureExtractor
from baseline.feature_extractor.pipeline import FeatureExtractionPipeline
from baseline.feature_extractor.runtime import AddressSpaceGuard, ModelRunLock
from baseline.feature_extractor.storage import ScratchSpace
from baseline.feature_extractor.trainer import FeatureExtractorTrainer
from baseline.minirocket.extractor import MiniRocketFeatureExtractor
from baseline.minirocket.minirocket_config import MiniRocketExtractorArgs


def _arrow_dataset() -> datasets.Dataset:
    """Return a small mixed-montage Arrow dataset."""
    return datasets.Dataset.from_dict(
        {
            "data": [
                np.array([[1.0, 2.0], [10.0, 20.0]], dtype=np.float32),
                np.array(
                    [
                        [30.0, 40.0],
                        [3.0, 4.0],
                        [300.0, 400.0],
                    ],
                    dtype=np.float32,
                ),
                np.array([[5.0, 6.0], [50.0, 60.0]], dtype=np.float32),
            ],
            "label": [0, 1, 0],
            "montage": ["toy/a", "toy/b", "toy/a"],
        }
    )


def test_classical_reader_streams_numpy_batches_and_aligns_montages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The classical reader aligns Arrow batches without neural loading."""
    calls = []

    def load_classical(name, config, split, fs):
        calls.append((name, config, split, fs))
        return _arrow_dataset()

    monkeypatch.setattr(
        data_module,
        "load_eeg_dataset_for_classical_ml",
        load_classical,
    )
    reader = FeatureSplitReader(
        dataset_name="toy",
        dataset_config="finetune",
        split=datasets.Split.TRAIN,
        fs=256,
        batch_size=2,
        n_class=2,
        channel_layout=["Cz"],
        montages={
            "toy/a": ["Cz", "Pz"],
            "toy/b": ["Fz", "Cz", "Pz"],
        },
    )

    batches = list(reader.batches())

    assert calls == [("toy", "finetune", datasets.Split.TRAIN, 256)]
    assert [batch.data.shape for batch in batches] == [(2, 1, 2), (1, 1, 2)]
    np.testing.assert_array_equal(
        np.concatenate([batch.data for batch in batches])[:, 0],
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )
    np.testing.assert_array_equal(
        np.concatenate([batch.labels for batch in batches]),
        np.array([0, 1, 0]),
    )


def test_classical_reader_rejects_non_finite_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-finite EEG fails in its Arrow batch before feature extraction."""
    dataset = datasets.Dataset.from_dict(
        {
            "data": [
                np.array(
                    [[1.0, 2.0], [np.nan, 20.0]],
                    dtype=np.float32,
                ),
                np.array(
                    [[30.0, 40.0], [3.0, 4.0]],
                    dtype=np.float32,
                ),
                np.array(
                    [[5.0, 6.0], [50.0, 60.0]],
                    dtype=np.float32,
                ),
            ],
            "label": [0, 1, 0],
            "montage": ["toy/a", "toy/b", "toy/a"],
        }
    )
    monkeypatch.setattr(
        data_module,
        "load_eeg_dataset_for_classical_ml",
        lambda *args: dataset,
    )
    reader = FeatureSplitReader(
        "toy",
        "finetune",
        datasets.Split.TRAIN,
        256,
        2,
        2,
        ["Cz"],
        {"toy/a": ["Cz", "Pz"], "toy/b": ["Fz", "Cz"]},
    )

    with pytest.raises(ValueError, match="contains NaN or inf"):
        list(reader.batches())


def test_classical_reader_rejects_empty_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty Arrow split fails before any memmap is allocated."""
    dataset = datasets.Dataset.from_dict(
        {"data": [], "label": [], "montage": []}
    )
    monkeypatch.setattr(
        data_module,
        "load_eeg_dataset_for_classical_ml",
        lambda *args: dataset,
    )

    with pytest.raises(ValueError, match="contains no trials"):
        FeatureSplitReader(
            "toy",
            "finetune",
            datasets.Split.TRAIN,
            256,
            2,
            2,
            ["Cz"],
            {"toy/a": ["Cz", "Pz"]},
        )


def test_classical_reader_rejects_invalid_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Labels outside the configured class range fail in their batch."""
    dataset = _arrow_dataset().remove_columns("label").add_column(
        "label",
        [0, 2, 0],
    )
    monkeypatch.setattr(
        data_module,
        "load_eeg_dataset_for_classical_ml",
        lambda *args: dataset,
    )
    reader = FeatureSplitReader(
        "toy",
        "finetune",
        datasets.Split.TRAIN,
        256,
        2,
        2,
        ["Cz"],
        {"toy/a": ["Cz", "Pz"], "toy/b": ["Fz", "Cz", "Pz"]},
    )

    with pytest.raises(ValueError, match=r"labels in \[0, 2\)"):
        next(reader.batches())


def test_classical_reader_rejects_inconsistent_trial_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A split inconsistent with the training trial shape fails early."""
    monkeypatch.setattr(
        data_module,
        "load_eeg_dataset_for_classical_ml",
        lambda *args: _arrow_dataset(),
    )
    reader = FeatureSplitReader(
        "toy",
        "finetune",
        datasets.Split.VALIDATION,
        256,
        2,
        2,
        ["Cz"],
        {"toy/a": ["Cz", "Pz"], "toy/b": ["Fz", "Cz", "Pz"]},
        expected_trial_shape=(1, 3),
    )

    with pytest.raises(ValueError, match="aligned EEG shape"):
        next(reader.batches())


class _PermissiveMemoryGuard:
    """Record requested mappings without applying a process limit."""

    def __init__(self):
        self.requests = []

    def require_additional(self, phase: str, requested_bytes: int) -> None:
        self.requests.append((phase, requested_bytes))


def test_scratch_memmap_is_removed_and_accounted(
    tmp_path: Path,
) -> None:
    """Scratch arrays are C-contiguous and removed before context exit."""
    guard = _PermissiveMemoryGuard()
    scratch = ScratchSpace(tmp_path, "test", guard)
    with scratch:
        store = scratch.create_array("features", (5, 3), np.float32)
        path = store.path
        store.array[:] = 1.0
        assert store.array.flags.c_contiguous
        assert path.is_file()
        assert scratch.current_bytes == 60

    assert not path.exists()
    assert scratch.current_bytes == 0
    assert scratch.peak_bytes == 60
    assert guard.requests == [("map scratch array features", 60)]


def test_scratch_memmap_is_removed_after_interruption(tmp_path: Path) -> None:
    """Invocation-local scratch is removed after an interruption."""
    scratch = ScratchSpace(tmp_path, "interrupted", _PermissiveMemoryGuard())
    with pytest.raises(KeyboardInterrupt):
        with scratch:
            store = scratch.create_array("features", (5, 3), np.float32)
            path = store.path
            raise KeyboardInterrupt

    assert not path.exists()
    assert scratch.directory is not None
    assert not scratch.directory.exists()
    assert scratch.current_bytes == 0


def test_scratch_preflight_preserves_free_disk_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scratch allocation fails before creating a file when disk is low."""
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        storage_module.shutil,
        "disk_usage",
        lambda _: usage(100, 90, 10),
    )
    scratch = ScratchSpace(tmp_path, "test", _PermissiveMemoryGuard())
    with scratch:
        with pytest.raises(RuntimeError, match="scratch reserve"):
            scratch.create_array("features", (5, 3), np.float32)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_memmap_ridge_matches_in_memory_workflow(
    tmp_path: Path,
    dtype: type[np.floating],
) -> None:
    """Chunked memmap scaling preserves Ridge selection and predictions."""
    rng = np.random.default_rng(42)
    train = rng.normal(size=(128, 17)).astype(dtype)
    validation = rng.normal(size=(31, 17)).astype(dtype)
    train_labels = np.tile(np.arange(4), 32)
    validation_labels = np.arange(31) % 4
    alphas = [0.1, 1.0, 10.0]

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train)
    scaled_validation = scaler.transform(validation)
    best_ridge = None
    best_alpha = None
    best_score = -np.inf
    for alpha in alphas:
        ridge = RidgeClassifier(alpha=alpha).fit(scaled_train, train_labels)
        score = np.mean(ridge.predict(scaled_validation) == validation_labels)
        if score > best_score:
            best_ridge = ridge
            best_alpha = alpha
            best_score = score

    path = tmp_path / "features.npy"
    mapped = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=train.shape,
    )
    mapped[:] = train
    classifier = ValidationSelectedRidgeClassifier(
        RidgeClassifierArgs(
            alphas=alphas,
            selection_metric="accuracy",
        ),
        batch_size=7,
    )
    classifier.fit(
        mapped,
        train_labels,
        validation,
        validation_labels,
    )

    assert classifier.selected_alpha == best_alpha
    np.testing.assert_array_equal(mapped, scaled_train)
    np.testing.assert_array_equal(
        classifier.pipeline.named_steps["scaler"].mean_,
        scaler.mean_,
    )
    np.testing.assert_array_equal(
        classifier.pipeline.named_steps["ridge"].coef_,
        best_ridge.coef_,
    )
    np.testing.assert_array_equal(
        classifier.predict(validation),
        best_ridge.predict(scaled_validation),
    )


def test_minirocket_memmap_matches_in_memory_input(tmp_path: Path) -> None:
    """MiniROCKET fitting and features are unchanged for a C memmap."""

    class FakeMiniRocketModule:
        @staticmethod
        def fit(data, num_features, max_dilations_per_kernel):
            return (
                data.mean(axis=(0, 2)),
                num_features,
                max_dilations_per_kernel,
            )

        @staticmethod
        def transform(data, parameters):
            return data.mean(axis=2) + parameters[0]

    rng = np.random.default_rng(7)
    train = rng.normal(size=(6, 2, 9)).astype(np.float32)
    path = tmp_path / "training.npy"
    mapped = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=train.shape,
    )
    mapped[:] = train
    args = MiniRocketExtractorArgs(source_path="/unused", n_jobs=1)
    in_memory = MiniRocketFeatureExtractor(args, seed=42)
    disk_backed = MiniRocketFeatureExtractor(args, seed=42)
    module = FakeMiniRocketModule()
    in_memory._load_module = lambda: module
    disk_backed._load_module = lambda: module

    in_memory.fit(train)
    disk_backed.fit(mapped)

    np.testing.assert_array_equal(
        in_memory.parameters[0],
        disk_backed.parameters[0],
    )
    assert in_memory.parameters[1:] == disk_backed.parameters[1:]
    np.testing.assert_array_equal(
        in_memory.transform(train),
        disk_backed.transform(mapped),
    )


def test_address_space_guard_preflights_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The memory guard lowers, checks, and restores the process limit."""
    infinity = runtime_module.resource.RLIM_INFINITY
    calls = []
    monkeypatch.setattr(
        runtime_module,
        "_virtual_memory_bytes",
        lambda: 2 * (1 << 30),
    )
    monkeypatch.setattr(
        runtime_module.resource,
        "getrlimit",
        lambda _: (infinity, infinity),
    )
    monkeypatch.setattr(
        runtime_module.resource,
        "setrlimit",
        lambda resource_name, limits: calls.append((resource_name, limits)),
    )

    with AddressSpaceGuard(4.0) as guard:
        guard.require_additional("small", 1 << 30)
        with pytest.raises(MemoryError, match="phase 'large'"):
            guard.require_additional("large", 3 * (1 << 30))

    assert calls[0][1][0] == 4 * (1 << 30)
    assert calls[-1][1] == (infinity, infinity)


def test_model_runtime_lock_is_per_model(tmp_path: Path) -> None:
    """Duplicate models conflict while different models can coexist."""
    with ModelRunLock(str(tmp_path), "catch22"):
        with pytest.raises(RuntimeError, match="Another catch22 invocation"):
            with ModelRunLock(str(tmp_path), "catch22"):
                pass
        with ModelRunLock(str(tmp_path), "minirocket"):
            pass
    with ModelRunLock(str(tmp_path), "catch22"):
        pass


def test_extractor_training_data_requirements() -> None:
    """Only MiniROCKET requests a random-access training EEG store."""
    catch22 = Catch22FeatureExtractor(n_jobs=1)
    minirocket = MiniRocketFeatureExtractor(
        MiniRocketExtractorArgs(source_path="/unused"),
        seed=42,
    )

    assert catch22.requires_random_access_training_data is False
    assert minirocket.requires_random_access_training_data is True


def test_non_finite_metrics_cannot_be_completed() -> None:
    """Completion is rejected before persisting a non-finite metric."""
    with pytest.raises(ValueError, match="is not finite"):
        FeatureExtractorTrainer._require_finite_metrics(
            {"toy/test/acc": np.nan}
        )


class _MeanExtractor(EEGFeatureExtractor):
    """Small extractor with configurable random-access fit requirements."""

    def __init__(self, requires_training: bool):
        self._requires_training = requires_training
        self.fit_input_is_memmap = None
        self.fit_input_snapshot = None

    @property
    def requires_random_access_training_data(self) -> bool:
        return self._requires_training

    def _fit(self, train_data: np.ndarray) -> None:
        self.fit_input_is_memmap = isinstance(train_data, np.memmap)
        self.fit_input_snapshot = np.array(train_data, copy=True)

    def _transform(self, data: np.ndarray) -> np.ndarray:
        return data.mean(axis=-1, dtype=np.float32)


class _ToyFeatureTrainer(FeatureExtractorTrainer):
    """Compose the shared classical workflow around a tiny extractor."""

    def __init__(self, extractor: _MeanExtractor, scratch_dir: Path):
        config = Catch22Config(
            fs=256,
            data={
                "datasets": {"toy": "finetune"},
                "scratch_dir": str(scratch_dir),
                "feature_batch_size": 2,
            },
            model={"classifier": {"alphas": [0.1, 1.0]}},
            logging={"use_cloud": False, "outputs": ["csv"]},
        )
        classifier = ValidationSelectedRidgeClassifier(
            config.model.classifier,
            batch_size=2,
        )
        pipeline = FeatureExtractionPipeline(extractor, classifier)
        self.extractor = extractor
        super().__init__(config, pipeline)
        self.ds_info = {"toy": {"n_class": 2}}

    def fit_extractor(self, train_data: np.ndarray) -> None:
        self.extractor.fit(train_data)

    def transform_features(self, data: np.ndarray) -> np.ndarray:
        return self.extractor.transform(data)


class _ArrayReader:
    """Yield bounded batches from one synthetic split."""

    def __init__(self, split: datasets.NamedSplit):
        self.dataset_name = "toy"
        self.split = split
        self.trial_shape = (2, 3)
        if split == datasets.Split.TRAIN:
            self.data = np.array(
                [
                    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    [[5.0, 5.0, 5.0], [5.0, 6.0, 5.0]],
                    [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                    [[6.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
                ],
                dtype=np.float32,
            )
            self.labels = np.array([0, 1, 0, 1])
        else:
            self.data = np.array(
                [
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                    [[5.0, 5.0, 6.0], [5.0, 5.0, 5.0]],
                ],
                dtype=np.float32,
            )
            self.labels = np.array([0, 1])

    def __len__(self) -> int:
        return len(self.data)

    def batches(self):
        for start in range(0, len(self.data), 2):
            stop = min(start + 2, len(self.data))
            yield AlignedEEGBatch(
                self.data[start:stop],
                self.labels[start:stop],
                start,
                stop,
            )


class _PartialArrayReader(_ArrayReader):
    """Incorrect reader that omits the final declared training rows."""

    def batches(self):
        yield AlignedEEGBatch(
            self.data[:2],
            self.labels[:2],
            0,
            2,
        )


@pytest.mark.parametrize("requires_training", [False, True])
def test_classical_workflow_streams_or_memmaps_by_fit_requirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requires_training: bool,
) -> None:
    """Only stateful extractors materialize random-access training EEG."""
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        storage_module.shutil,
        "disk_usage",
        lambda _: usage(64 << 30, 1 << 30, 63 << 30),
    )
    monkeypatch.setattr(trainer_module, "get_dataset_n_class", lambda *args: 2)
    extractor = _MeanExtractor(requires_training)
    trainer = _ToyFeatureTrainer(extractor, tmp_path)
    trainer._open_split_reader = lambda *args, **kwargs: _ArrayReader(args[2])
    guard = _PermissiveMemoryGuard()
    scratch = ScratchSpace(tmp_path, "workflow", guard)

    with scratch:
        validation_metrics, test_metrics = trainer._run_dataset_classical(
            "toy",
            "finetune",
            scratch,
        )

    if requires_training:
        assert extractor.fit_input_is_memmap is True
        np.testing.assert_array_equal(
            extractor.fit_input_snapshot,
            _ArrayReader(datasets.Split.TRAIN).data,
        )
    else:
        assert extractor.fit_input_is_memmap is None
    assert validation_metrics["toy/eval/acc"] == 1.0
    assert test_metrics["toy/test/acc"] == 1.0
    assert scratch.current_bytes == 0
    expected_bytes = 144 if requires_training else 48
    assert scratch.total_allocated_bytes == expected_bytes


@pytest.mark.parametrize("requires_training", [False, True])
def test_classical_workflow_rejects_partial_training_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requires_training: bool,
) -> None:
    """Neither extractor path may fit after omitting training rows."""
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        storage_module.shutil,
        "disk_usage",
        lambda _: usage(64 << 30, 1 << 30, 63 << 30),
    )
    extractor = _MeanExtractor(requires_training)
    trainer = _ToyFeatureTrainer(extractor, tmp_path)
    trainer._open_split_reader = (
        lambda *args, **kwargs: _PartialArrayReader(args[2])
    )
    scratch = ScratchSpace(
        tmp_path,
        "partial-workflow",
        _PermissiveMemoryGuard(),
    )

    with scratch:
        with pytest.raises(ValueError, match="consumed 2"):
            trainer._run_dataset_classical("toy", "finetune", scratch)

    assert extractor.fit_input_snapshot is None
