"""Unit tests for sklearn EEG feature-extractor baselines."""

import sys
import types

import numpy as np
import pytest

from baseline.catch22.catch22_config import Catch22Config
from baseline.catch22.catch22_trainer import Catch22Trainer
from baseline.catch22.extractor import _write_terminal_progress
from baseline.feature_extractor.trainer import FeatureExtractorTrainer
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
        "seen = {}\n"
        "def fit(X, num_features, max_dilations_per_kernel):\n"
        "    seen['dtype'] = X.dtype\n"
        "    seen['shape'] = X.shape\n"
        "    return (num_features, max_dilations_per_kernel)\n"
        "def transform(X, parameters):\n"
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

    trainer.fit_extractor(data)
    features = trainer.transform_features(data)

    assert trainer.extractor._module.seen["dtype"] == np.dtype("float32")
    assert trainer.extractor._module.seen["shape"] == (3, 2, 9)
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
    with pytest.raises(FileNotFoundError, match="does not support checkpoint"):
        load_final_checkpoint(tmp_path, DATASET_NAME)


def test_feature_matrix_validation_rejects_nan():
    """Feature extraction rejects non-finite values before Ridge fitting."""
    trainer = MeanFeatureTrainer(make_config())
    data = np.ones((2, 1, 2), dtype=np.float32)
    trainer.transform_features = lambda _: np.array([[np.nan], [1.0]])

    with pytest.raises(ValueError, match="NaN or inf"):
        trainer._extract_features(data, "validation")
