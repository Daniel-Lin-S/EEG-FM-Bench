"""Tests for persistent baseline run artifacts."""

from pathlib import Path

import pytest

from baseline.brainomni.brainomni_config import BrainOmniConfig
from baseline.brainomni.brainomni_config import BrainOmniLoggingArgs
from baseline.utils.run_artifacts import get_config_hash
from baseline.utils.run_artifacts import load_saved_run_config
from baseline.utils.run_artifacts import save_resolved_config


def test_outputs_default_and_validation() -> None:
    """Local trace outputs default to CSV and reject empty selections."""
    assert BrainOmniLoggingArgs().outputs == ['csv']
    with pytest.raises(ValueError, match='at least one trace type'):
        BrainOmniLoggingArgs(outputs=[])


def test_single_task_hash_ignores_dataset_selection() -> None:
    """Single-task experiment identity remains stable as datasets are added."""
    config = BrainOmniConfig().model_dump(mode='json')
    first_hash = get_config_hash(config, multitask=False)
    config['data']['datasets'] = {'bcic_1a': 'finetune'}
    assert first_hash == get_config_hash(config, multitask=False)
    assert first_hash != get_config_hash(config, multitask=True)


def test_saved_config_round_trip(tmp_path: Path) -> None:
    """Saved run configurations load without source-YAML merging."""
    config = BrainOmniConfig().model_dump(mode='json')
    save_resolved_config(config, tmp_path / 'configs' / 'run.yaml')
    loaded = load_saved_run_config(tmp_path)
    assert loaded.model_dump(mode='json') == config
