"""Unit tests for sklearn EEG feature-extractor baselines."""

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import baseline_main
from baseline.catch22.catch22_config import Catch22Config
from baseline.catch22.catch22_trainer import Catch22Trainer
import baseline.catch22.extractor as catch22_extractor_module
import baseline.feature_extractor.classifier as classifier_module
import baseline.feature_extractor.trainer as trainer_module
from baseline.catch22.extractor import (
    CATCH22_TRIAL_CHUNKSIZE,
    Catch22FeatureExtractor,
    _write_terminal_progress,
)
from baseline.feature_extractor.classifier import (
    RidgeClassifierArgs,
    ValidationSelectedRidgeClassifier,
)
from baseline.feature_extractor.artifacts import (
    find_matching_artifact_root,
)
from baseline.feature_extractor.extractor import EEGFeatureExtractor
from baseline.feature_extractor.pipeline import FeatureExtractionPipeline
from baseline.feature_extractor.summary import (
    write_feature_extractor_summary,
)
from baseline.feature_extractor.trainer import FeatureExtractorTrainer
from baseline.hpo.config import HpoConfig
from baseline.minirocket.minirocket_config import MiniRocketConfig
from baseline.minirocket.minirocket_trainer import MiniRocketTrainer
from baseline.utils.run_artifacts import load_final_checkpoint


DATASET_NAME = "toy"
DATASET_CONFIG = "finetune"


class MeanFeatureTrainer(FeatureExtractorTrainer):
    """Small feature extractor for testing shared Ridge behavior."""

    def fit_extractor(self, train_data: np.ndarray) -> None:
        """Retain no trainable extractor state."""

    def transform_features(self, data: np.ndarray) -> np.ndarray:
        """Return channel means as dense features."""
        return data.mean(axis=-1)


def make_config(
    model_type: str = "catch22",
    model: dict | None = None,
) -> Catch22Config:
    """Create one minimal feature-extractor configuration."""
    return Catch22Config(
        model_type=model_type,
        data={"datasets": {DATASET_NAME: DATASET_CONFIG}},
        model=model or {},
        logging={"use_cloud": False, "outputs": ["csv"]},
    )


def test_feature_extractor_logging_honors_configured_level(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed feature runs never reset verbosity to hard-coded INFO."""
    config = make_config()
    config.logging.level = "error"
    trainer = MeanFeatureTrainer(config)
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        trainer,
        "get_train_io_path",
        lambda _: (str(tmp_path.resolve()), ""),
    )
    monkeypatch.setattr(trainer_module, "get_is_master", lambda: True)

    def record_setup(**kwargs: Any) -> None:
        """Record the effective setup options."""
        calls.update(kwargs)

    monkeypatch.setattr(trainer_module, "setup_log", record_setup)

    trainer.setup_logging()

    assert calls["level"] == "ERROR"


def test_catch22_concatenates_canonical_features(monkeypatch):
    """catch22 extracts 22 values per channel in channel order."""
    fake_module = types.ModuleType("pycatch22")

    def catch22_all(values, catch24):
        assert not catch24
        return {"values": [values[0] + index for index in range(22)]}

    fake_module.catch22_all = catch22_all
    monkeypatch.setitem(sys.modules, "pycatch22", fake_module)
    trainer = Catch22Trainer(
        make_config(model={"extractor": {"n_jobs": 1}})
    )
    data = np.array([[[1.0, 2.0], [10.0, 20.0]]], dtype=np.float32)

    trainer.fit_extractor(data)
    features = trainer.transform_features(data)

    assert features.shape == (1, 44)
    np.testing.assert_array_equal(features[0, :22], np.arange(1.0, 23.0))
    np.testing.assert_array_equal(features[0, 22:], np.arange(10.0, 32.0))


class CountingFloat32Extractor(EEGFeatureExtractor):
    """Count feature-extraction calls while preserving float32 output."""

    def __init__(self):
        self.fit_calls = 0
        self.transform_calls = 0

    def _fit(self, train_data: np.ndarray) -> None:
        self.fit_calls += 1

    def _transform(self, data: np.ndarray) -> np.ndarray:
        self.transform_calls += 1
        return data.mean(axis=-1, dtype=np.float32)


def test_pipeline_caches_validation_features_and_preserves_dtype():
    """Pipeline fitting transforms validation EEG once without widening.

    The returned matrix retains the extractor output dtype.
    """
    extractor = CountingFloat32Extractor()
    classifier = ValidationSelectedRidgeClassifier(
        RidgeClassifierArgs(alphas=[0.1, 1.0])
    )
    pipeline = FeatureExtractionPipeline(extractor, classifier)
    train_data = np.array(
        [[[0.0]], [[1.0]], [[10.0]], [[11.0]]],
        dtype=np.float32,
    )
    validation_data = np.array([[[0.5]], [[10.5]]], dtype=np.float32)
    fit_result = pipeline.fit(
        train_data,
        np.array([0, 0, 1, 1]),
        validation_data,
        np.array([0, 1]),
    )

    assert extractor.fit_calls == 1
    assert extractor.transform_calls == 2
    assert fit_result.validation_features.dtype == np.float32
    np.testing.assert_array_equal(
        fit_result.validation_features,
        np.array([[0.5], [10.5]], dtype=np.float32),
    )


def test_ridge_alpha_search_fits_scaler_once(monkeypatch):
    """Ridge candidates share one fitted scaler and fit independently."""
    original_scaler = classifier_module.StandardScaler
    original_ridge = classifier_module.RidgeClassifier

    class CountingScaler(original_scaler):
        fit_calls = 0

        def fit(self, features, labels=None, sample_weight=None):
            type(self).fit_calls += 1
            return super().fit(features, labels, sample_weight)

    class CountingRidge(original_ridge):
        fit_calls = 0

        def fit(self, features, labels, sample_weight=None):
            type(self).fit_calls += 1
            return super().fit(features, labels, sample_weight)

    monkeypatch.setattr(classifier_module, "StandardScaler", CountingScaler)
    monkeypatch.setattr(classifier_module, "RidgeClassifier", CountingRidge)
    classifier = classifier_module.ValidationSelectedRidgeClassifier(
        RidgeClassifierArgs(alphas=[0.01, 0.1, 1.0])
    )
    classifier.fit(
        np.array([[0.0], [1.0], [10.0], [11.0]]),
        np.array([0, 0, 1, 1]),
        np.array([[0.5], [10.5]]),
        np.array([0, 1]),
    )

    assert CountingScaler.fit_calls == 1
    assert CountingRidge.fit_calls == 3


def test_catch22_parallel_dispatch_uses_trial_chunks(monkeypatch):
    """Parallel catch22 extraction batches independent trial tasks."""
    fake_module = types.ModuleType("pycatch22")

    def catch22_all(values, catch24):
        assert not catch24
        return {"values": [values[0] + index for index in range(22)]}

    fake_module.catch22_all = catch22_all

    class ImmediateExecutor:
        chunksize = None

        def __init__(self, max_workers, mp_context):
            self.max_workers = max_workers
            self.mp_context = mp_context

        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception, traceback):
            return False

        def map(self, function, items, chunksize):
            type(self).chunksize = chunksize
            return map(function, items)

    monkeypatch.setitem(sys.modules, "pycatch22", fake_module)
    monkeypatch.setattr(
        catch22_extractor_module,
        "ProcessPoolExecutor",
        ImmediateExecutor,
    )
    monkeypatch.setattr(
        catch22_extractor_module,
        "get_context",
        lambda _: object(),
    )
    data = np.array([[[1.0]], [[2.0]]], dtype=np.float32)
    serial_extractor = Catch22FeatureExtractor(n_jobs=1)
    parallel_extractor = Catch22FeatureExtractor(n_jobs=2)

    serial_extractor.fit(data)
    parallel_extractor.fit(data)
    serial_features = serial_extractor.transform(data)
    parallel_features = parallel_extractor.transform(data)

    np.testing.assert_array_equal(serial_features, parallel_features)
    assert ImmediateExecutor.chunksize == CATCH22_TRIAL_CHUNKSIZE


def test_ridge_selection_uses_validation_and_scales_features():
    """Ridge candidates fit train data and select the smallest tied alpha."""
    trainer = MeanFeatureTrainer(
        make_config(model={"classifier": {"alphas": [10.0, 1.0]}})
    )
    train_features = np.array([[0.0], [1.0], [10.0], [11.0]])
    train_labels = np.array([0, 0, 1, 1])
    validation_features = np.array([[0.5], [10.5]])
    validation_labels = np.array([0, 1])

    classifier, selected_alpha = trainer._fit_classifier(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    )

    assert selected_alpha == 1.0
    assert classifier.named_steps["scaler"].mean_[0] == pytest.approx(5.5)
    assert classifier.predict(validation_features).tolist() == [0, 1]


@pytest.mark.parametrize(
    "metric",
    ["balanced_accuracy", "accuracy", "f1_weighted"],
)
def test_supported_ridge_selection_metrics(metric):
    """Every configured selection metric calculates a finite score."""
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])

    score = MeanFeatureTrainer._selection_score(labels, predictions, metric)

    assert np.isfinite(score)


def test_feature_extractor_config_rejects_multitask_and_bad_alphas():
    """Feature baselines reject incompatible and ambiguous configuration."""
    multitask_config = make_config()
    multitask_config.multitask = True
    with pytest.raises(ValueError, match="multitask=false"):
        multitask_config.validate_config()
    with pytest.raises(ValueError, match="positive"):
        make_config(model={"classifier": {"alphas": [1.0, 0.0]}})
    with pytest.raises(ValueError, match="duplicates"):
        make_config(model={"classifier": {"alphas": [1.0, 1.0]}})


def test_minirocket_loads_external_multivariate_source(tmp_path):
    """miniROCKET loads only the configured external source clone."""
    source_root = tmp_path / "minirocket"
    source_dir = source_root / "code"
    source_dir.mkdir(parents=True)
    module_path = source_dir / "minirocket_multivariate.py"
    module_path.write_text(
        "import numpy as np\n"
        "seen = {'fit_calls': 0, 'transform_calls': 0}\n"
        "def fit(X, num_features, max_dilations_per_kernel):\n"
        "    seen['fit_calls'] += 1\n"
        "    seen['dtype'] = X.dtype\n"
        "    seen['shape'] = X.shape\n"
        "    return (num_features, max_dilations_per_kernel)\n"
        "def transform(X, parameters):\n"
        "    seen['transform_calls'] += 1\n"
        "    return np.full((X.shape[0], 2), parameters[0], "
        "dtype=np.float32)\n",
        encoding="utf-8",
    )
    config = MiniRocketConfig(
        data={"datasets": {DATASET_NAME: DATASET_CONFIG}},
        model={
            "extractor": {
                "source_path": str(source_root),
                "num_features": 12,
                "max_dilations_per_kernel": 3,
            },
        },
    )
    trainer = MiniRocketTrainer(config)
    data = np.ones((3, 2, 9), dtype=np.float32)

    fit_result = trainer.pipeline.fit(
        data,
        np.array([0, 1, 0]),
        data[:2],
        np.array([0, 1]),
    )
    features = trainer.transform_features(data)

    assert trainer.extractor._module.seen["dtype"] == np.dtype("float32")
    assert trainer.extractor._module.seen["shape"] == (3, 2, 9)
    assert trainer.extractor._module.seen["fit_calls"] == 1
    assert trainer.extractor._module.seen["transform_calls"] == 3
    assert fit_result.validation_features.dtype == np.float32
    assert features.shape == (3, 2)
    assert np.all(features == 12)


def test_minirocket_requires_source_path():
    """miniROCKET reports a clear error when no clone path is configured."""
    config = MiniRocketConfig(
        data={"datasets": {DATASET_NAME: DATASET_CONFIG}},
    )
    trainer = MiniRocketTrainer(config)

    with pytest.raises(ValueError, match="model.extractor.source_path"):
        trainer.fit_extractor(np.ones((2, 1, 9), dtype=np.float32))



def test_catch22_progress_is_silent_without_a_terminal(monkeypatch, capsys):
    """catch22 progress must not enter captured logs or redirected stderr."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)

    _write_terminal_progress(100, 200)

    assert capsys.readouterr().err == ""


def test_feature_extractor_csv_omits_neural_training_coordinates(tmp_path):
    """Feature-extractor CSV metrics have no epoch or optimizer-step fields."""
    trainer = MeanFeatureTrainer(make_config())
    trainer.log_dir = str(tmp_path)
    trainer.current_dataset = DATASET_NAME
    trainer._open_csv_writer(DATASET_NAME)
    trainer._write_csv_metrics(
        {f"{DATASET_NAME}/eval/balanced_acc": 0.5}
    )
    trainer._close_csv_writer()

    csv_content = (tmp_path / "csv" / f"{DATASET_NAME}.csv").read_text(
        encoding="utf-8"
    )
    assert csv_content.splitlines()[0] == "timestamp,dataset,split,metric,value"
    assert "epoch" not in csv_content
    assert "step" not in csv_content

def test_no_checkpoint_completion_skips_and_reports_clear_error(tmp_path):
    """Feature completion metadata supports rerun skipping without a model."""
    trainer = MeanFeatureTrainer(make_config())
    trainer.log_dir = str(tmp_path)
    trainer.execution_id = "test-run"
    validation_metrics = {f"{DATASET_NAME}/eval/balanced_acc": 0.5}
    test_metrics = {f"{DATASET_NAME}/test/balanced_acc": 0.5}
    trainer.final_validation_metrics[DATASET_NAME] = validation_metrics
    trainer.final_test_metrics[DATASET_NAME] = test_metrics

    trainer._write_completion(DATASET_NAME, DATASET_CONFIG)

    assert trainer._dataset_is_complete(DATASET_NAME, DATASET_CONFIG)
    with pytest.raises(FileNotFoundError, match="retention was disabled"):
        load_final_checkpoint(tmp_path, DATASET_NAME)


def _write_feature_completion(
    artifact_root: Path,
    test_metrics: dict[str, float],
) -> None:
    """Write one completed deterministic extractor result for tests."""
    trainer = MeanFeatureTrainer(make_config())
    trainer.log_dir = str(artifact_root)
    trainer.execution_id = "test-run"
    trainer.final_validation_metrics[DATASET_NAME] = {
        f"{DATASET_NAME}/eval/balanced_acc": 0.5,
    }
    trainer.final_test_metrics[DATASET_NAME] = test_metrics
    trainer._write_completion(DATASET_NAME, DATASET_CONFIG)


def test_feature_summary_has_campaign_csv_schema_without_std(tmp_path):
    """One-seed extractor summaries omit neural fields and standard deviation."""
    test_metrics = {
        f"{DATASET_NAME}/test/acc": 0.6,
        f"{DATASET_NAME}/test/balanced_acc": 0.55,
    }
    _write_feature_completion(tmp_path, test_metrics)

    write_feature_extractor_summary(
        tmp_path,
        "catch22",
        42,
        {DATASET_NAME: DATASET_CONFIG},
        "config-identity",
    )

    test_runs = (tmp_path / "summary" / "test_runs.csv").read_text(
        encoding="utf-8"
    )
    test_summary = (tmp_path / "summary" / "test_summary.csv").read_text(
        encoding="utf-8"
    )
    status = json.loads(
        (tmp_path / "summary" / "summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert test_runs.splitlines()[0] == "dataset,seed,metric,value"
    assert test_summary.splitlines()[0] == (
        "dataset,metric,count,mean,median"
    )
    assert "std" not in test_summary
    assert "epoch" not in test_runs
    assert "loss" not in test_runs
    assert status["status"] == "complete"
    assert status["dataset_pairs"]["completed"] == 1


def test_feature_summary_rejects_neural_test_metrics(tmp_path):
    """Feature summaries reject neural-only metrics instead of exporting them."""
    _write_feature_completion(
        tmp_path,
        {f"{DATASET_NAME}/test/loss": 1.0},
    )

    with pytest.raises(ValueError, match="neural-only"):
        write_feature_extractor_summary(
            tmp_path,
            "catch22",
            42,
            {DATASET_NAME: DATASET_CONFIG},
            "config-identity",
        )


def test_matching_artifact_root_normalizes_legacy_scalar_seed(tmp_path):
    """Existing scalar-seed extractor artifacts are reused on rerun."""
    config = make_config()
    config.logging.run_dir = str(tmp_path)
    artifact_root = (
        tmp_path
        / "log"
        / "baseline"
        / "catch22"
        / "catch22-legacy"
    )
    config_path = artifact_root / "configs" / "legacy.yaml"
    config_path.parent.mkdir(parents=True)
    saved_config = config.model_dump(mode="json")
    saved_config["seed"] = saved_config.pop("seeds")[0]
    config_path.write_text(
        yaml.safe_dump(saved_config, sort_keys=False),
        encoding="utf-8",
    )

    assert find_matching_artifact_root(config) == artifact_root


def test_matching_artifact_root_rejects_ambiguity(tmp_path):
    """Reruns fail instead of selecting between matching legacy roots."""
    config = make_config()
    config.logging.run_dir = str(tmp_path)
    for suffix in ("first", "second"):
        config_path = (
            tmp_path
            / "log"
            / "baseline"
            / "catch22"
            / suffix
            / "configs"
            / "legacy.yaml"
        )
        config_path.parent.mkdir(parents=True)
        config_path.write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )

    with pytest.raises(RuntimeError, match="Multiple feature-extractor"):
        find_matching_artifact_root(config)


def test_all_skipped_feature_rerun_writes_summary(tmp_path):
    """Completed extractor datasets are skipped while their summary refreshes."""
    config = make_config()
    trainer = MeanFeatureTrainer(config)
    trainer.log_dir = str(tmp_path)
    trainer.execution_id = "completed-run"
    trainer.final_validation_metrics[DATASET_NAME] = {
        f"{DATASET_NAME}/eval/balanced_acc": 0.5,
    }
    trainer.final_test_metrics[DATASET_NAME] = {
        f"{DATASET_NAME}/test/balanced_acc": 0.5,
    }
    trainer._write_completion(DATASET_NAME, DATASET_CONFIG)

    rerun = MeanFeatureTrainer(config)
    rerun.log_dir_override = tmp_path
    rerun.fit_extractor = lambda _: pytest.fail("Extractor must be skipped")
    rerun.run()

    assert (tmp_path / "summary" / "test_runs.csv").is_file()
    assert (tmp_path / "summary" / "test_summary.csv").is_file()
    assert (tmp_path / "summary" / "summary.json").is_file()


def test_feature_entry_point_requires_one_seed():
    """The direct feature-extractor path rejects repeated seeds."""
    config = make_config()
    config.seeds = [42, 43]

    with pytest.raises(ValueError, match="exactly one deterministic seed"):
        baseline_main._run_feature_extractor(config, HpoConfig())


def test_feature_matrix_validation_rejects_nan():
    """Feature extraction rejects non-finite values before Ridge fitting."""
    trainer = MeanFeatureTrainer(make_config())
    data = np.ones((2, 1, 2), dtype=np.float32)
    trainer.transform_features = lambda _: np.array([[np.nan], [1.0]])

    with pytest.raises(ValueError, match="NaN or inf"):
        trainer._extract_features(data, "validation")
