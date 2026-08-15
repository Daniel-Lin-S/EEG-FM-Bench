"""Tests for persistent baseline run artifacts."""

from pathlib import Path

import pytest

from baseline.brainomni.brainomni_config import BrainOmniConfig
from baseline.abstract.trainer import AbstractTrainer
from baseline.brainomni.brainomni_config import BrainOmniLoggingArgs
from baseline.utils.run_artifacts import get_config_hash
from baseline.utils.run_artifacts import load_saved_run_config
from baseline.utils.run_artifacts import save_resolved_config


TEST_FS = 256


def test_performance_diagnostics_aggregate_finite_passes() -> None:
    """Timing diagnostics retain compact finite split summaries."""
    class PerformanceTrainer(AbstractTrainer):
        def setup_model(self):
            raise NotImplementedError

        def load_checkpoint(self, checkpoint_path: str):
            raise NotImplementedError

    trainer = object.__new__(PerformanceTrainer)
    trainer.multitask = False
    trainer._loader_build_seconds = {}
    trainer._evaluation_timings = {}

    trainer._record_loader_build_seconds("validation", 1.5)
    trainer._record_loader_build_seconds("validation", 0.5)
    trainer._record_evaluation_seconds("eval", 2.0)
    trainer._record_evaluation_seconds("eval", 4.0)

    assert trainer.get_performance_diagnostics() == {
        "scope": "dataset",
        "loader_build_seconds": {"validation": 2.0},
        "evaluation": {
            "validation": {
                "passes": 2,
                "total_seconds": 6.0,
                "mean_seconds": 3.0,
                "latest_seconds": 4.0,
            }
        },
    }


def test_checkpoint_retention_default_and_identity_exclusion() -> None:
    """Checkpoint retention defaults off and remains operational."""
    logging_args = BrainOmniLoggingArgs()
    assert logging_args.save_checkpoints is False
    assert BrainOmniLoggingArgs(
        save_checkpoints=True
    ).save_checkpoints is True

    config = BrainOmniConfig(fs=TEST_FS).model_dump(mode="json")
    original = get_config_hash(config, multitask=False)
    config["logging"]["save_checkpoints"] = True
    assert get_config_hash(config, multitask=False) == original


def test_outputs_default_and_validation() -> None:
    """Local trace outputs default to CSV and reject empty selections."""
    assert BrainOmniLoggingArgs().outputs == ['csv']
    with pytest.raises(ValueError, match='at least one trace type'):
        BrainOmniLoggingArgs(outputs=[])

def test_logging_level_is_normalized_and_not_part_of_identity() -> None:
    """Operational verbosity is validated without invalidating artifacts."""
    assert BrainOmniLoggingArgs().level == 'info'
    assert BrainOmniLoggingArgs(level='DEBUG').level == 'debug'
    with pytest.raises(ValueError, match='Input should be'):
        BrainOmniLoggingArgs(level='trace')

    config = BrainOmniConfig(fs=TEST_FS).model_dump(mode='json')
    original = get_config_hash(config, multitask=False)
    config['logging']['level'] = 'debug'

    assert get_config_hash(config, multitask=False) == original



def test_final_run_identity_excludes_separate_dataset_selection() -> None:
    """Separate tasks share one campaign while multitask retains membership."""
    config = BrainOmniConfig(fs=TEST_FS).model_dump(mode='json')
    separate_hash = get_config_hash(config, multitask=False)
    multitask_hash = get_config_hash(config, multitask=True)
    config['data']['datasets'] = {'bcic_1a': 'finetune'}

    assert separate_hash == get_config_hash(config, multitask=False)
    assert multitask_hash != get_config_hash(config, multitask=True)


def test_saved_config_round_trip(tmp_path: Path) -> None:
    """Saved run configurations load without source-YAML merging."""
    config = BrainOmniConfig(fs=TEST_FS).model_dump(mode='json')
    save_resolved_config(config, tmp_path / 'configs' / 'run.yaml')
    loaded = load_saved_run_config(tmp_path)
    assert loaded.model_dump(mode='json') == config
