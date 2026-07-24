"""Tests for canonical and legacy campaign completion recovery.

Inputs are synthetic resolved configurations, completion JSON files, and
checkpoint paths. Outputs verify safe pair-level reuse and summary recovery.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from baseline.brainomni.brainomni_config import BrainOmniConfig
from baseline.hpo.artifacts import (
    CampaignPaths,
    check_completion_compatibility,
    write_campaign_summary,
)
from baseline.hpo.orchestrator import _force_local_trial_logging
from baseline.utils.run_artifacts import get_config_hash
from baseline.utils.run_artifacts import save_resolved_config


DATASET_NAME = "adftd"
DATASET_CONFIG = "finetune"
CAMPAIGN_HASH = "campaign"
SEED = 42


def _selected_config(batch_size: int = 128) -> dict:
    """Return one resolved selected configuration for compatibility tests."""
    return BrainOmniConfig(
        seeds=[SEED],
        data={
            "datasets": {DATASET_NAME: DATASET_CONFIG},
            "batch_size": batch_size,
        },
    ).model_dump(mode="json")


def _completion_path(root: Path) -> Path:
    """Return the synthetic dataset completion path."""
    return (
        root
        / "logs"
        / f"seed_{SEED}"
        / "datasets"
        / DATASET_NAME
        / "completion.json"
    )


def _write_completion(
    root: Path,
    config_hash: str,
    execution_id: str = "execution",
    metric: float = 0.75,
) -> Path:
    """Write one completion and its required checkpoint artifact."""
    path = _completion_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.touch()
    path.write_text(
        json.dumps({
            "status": "completed",
            "campaign_hash": CAMPAIGN_HASH,
            "config_hash": config_hash,
            "seed": SEED,
            "dataset_config": DATASET_CONFIG,
            "execution_id": execution_id,
            "checkpoint_path": str(checkpoint.resolve()),
            "test_metrics": {f"{DATASET_NAME}/test/acc": metric},
        }),
        encoding="utf-8",
    )
    return path


def _save_legacy_config(root: Path, config: dict) -> str:
    """Save one historical resolved config and return its stored hash."""
    config_path = (
        root
        / "logs"
        / f"seed_{SEED}"
        / "configs"
        / "execution.yaml"
    )
    save_resolved_config(config, config_path)
    return get_config_hash(config, multitask=False)


def test_exact_canonical_completion_is_compatible(tmp_path: Path) -> None:
    """A valid canonical hash is accepted without legacy inspection."""
    selected = _selected_config()
    config_hash = get_config_hash(selected, multitask=False)
    path = _write_completion(tmp_path, config_hash)

    result = check_completion_compatibility(
        path,
        CAMPAIGN_HASH,
        SEED,
        selected,
    )

    assert result.compatible is True
    assert result.mode == "exact_canonical_hash"


def test_legacy_runtime_batch_is_recovered_read_only(
    tmp_path: Path,
) -> None:
    """A divisor-only historical batch mutation is safely recovered."""
    selected = _selected_config()
    saved = copy.deepcopy(selected)
    saved["data"]["batch_size"] = 64
    stored_hash = _save_legacy_config(tmp_path, saved)
    path = _write_completion(tmp_path, stored_hash)
    original_completion = path.read_text(encoding="utf-8")

    result = check_completion_compatibility(
        path,
        CAMPAIGN_HASH,
        SEED,
        selected,
    )

    assert result.compatible is True
    assert result.mode == "legacy_runtime_batch_compatible"
    assert path.read_text(encoding="utf-8") == original_completion


def test_legacy_recovery_rejects_semantic_or_invalid_batch_changes(
    tmp_path: Path,
) -> None:
    """Only a positive divisor batch may differ in a legacy config."""
    selected = _selected_config()
    saved = copy.deepcopy(selected)
    saved["data"]["batch_size"] = 64
    stored_hash = _save_legacy_config(tmp_path, saved)
    path = _write_completion(tmp_path, stored_hash)

    changed = copy.deepcopy(selected)
    changed["model"]["classifier_head"]["hidden_dims"] = [64]
    semantic = check_completion_compatibility(
        path,
        CAMPAIGN_HASH,
        SEED,
        changed,
    )
    assert semantic.compatible is False
    assert "beyond the runtime batch" in semantic.reason

    saved["data"]["batch_size"] = 48
    stored_hash = _save_legacy_config(tmp_path, saved)
    path = _write_completion(tmp_path, stored_hash)
    invalid_batch = check_completion_compatibility(
        path,
        CAMPAIGN_HASH,
        SEED,
        selected,
    )
    assert invalid_batch.compatible is False
    assert "positive divisor" in invalid_batch.reason


@pytest.mark.parametrize("failure", ["missing_checkpoint", "nonfinite"])
def test_completion_rejects_invalid_artifacts(
    tmp_path: Path,
    failure: str,
) -> None:
    """Missing checkpoints and non-finite metrics are rejected."""
    selected = _selected_config()
    config_hash = get_config_hash(selected, multitask=False)
    metric = float("nan") if failure == "nonfinite" else 0.75
    path = _write_completion(tmp_path, config_hash, metric=metric)
    if failure == "missing_checkpoint":
        completion = json.loads(path.read_text(encoding="utf-8"))
        Path(completion["checkpoint_path"]).unlink()

    result = check_completion_compatibility(
        path,
        CAMPAIGN_HASH,
        SEED,
        selected,
    )

    assert result.compatible is False


def test_partial_summary_records_pair_diagnostics(tmp_path: Path) -> None:
    """A valid pair is retained when a sibling dataset is missing."""
    selected = _selected_config()
    config_hash = get_config_hash(selected, multitask=False)
    _write_completion(tmp_path, config_hash)
    paths = CampaignPaths(tmp_path, tmp_path / "checkpoints")
    compatible = {
        (SEED, DATASET_NAME): selected,
        (SEED, "missing"): config_hash,
    }

    summary = write_campaign_summary(
        paths,
        CAMPAIGN_HASH,
        {"attempted": [SEED]},
        compatible,
    )

    assert summary is not None
    assert summary["status"] == "partial"
    assert len(summary["compatibility"]["accepted"]) == 1
    assert len(summary["compatibility"]["missing"]) == 1


def test_zero_row_summary_preserves_previous_summary(tmp_path: Path) -> None:
    """No compatible rows produce diagnostics without replacing summaries."""
    paths = CampaignPaths(tmp_path, tmp_path / "checkpoints")
    previous_path = paths.summary_root / "summary.json"
    previous_path.parent.mkdir(parents=True, exist_ok=True)
    previous_path.write_text("previous", encoding="utf-8")

    with pytest.warns(UserWarning, match="left unchanged"):
        summary = write_campaign_summary(
            paths,
            CAMPAIGN_HASH,
            {"attempted": [SEED]},
            {(SEED, DATASET_NAME): _selected_config()},
        )

    assert summary is None
    assert previous_path.read_text(encoding="utf-8") == "previous"
    report_path = paths.summary_root / "compatibility_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "no_compatible_results"
    assert report["compatibility"]["missing"]


def test_hpo_trial_text_logs_follow_debug_verbosity() -> None:
    """Normal HPO omits text logs while DEBUG retains them."""
    normal = {
        "logging": {
            "level": "info",
            "outputs": ["log", "tensorboard"],
        },
    }
    _force_local_trial_logging(normal)
    assert normal["logging"]["outputs"] == ["tensorboard", "csv"]

    debug = {
        "logging": {
            "level": "debug",
            "outputs": ["log"],
        },
    }
    _force_local_trial_logging(debug)
    assert debug["logging"]["outputs"] == ["log", "csv"]
