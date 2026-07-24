"""Tests for canonical and legacy campaign completion recovery.

Inputs are synthetic resolved configurations, completion JSON files, and
checkpoint paths. Outputs verify safe pair-level reuse and summary recovery.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import pytest

from baseline.brainomni.brainomni_config import BrainOmniConfig
from baseline.hpo.artifacts import (
    CampaignPaths,
    check_completion_compatibility,
    locate_completion,
    write_campaign_summary,
)
from baseline.hpo.orchestrator import (
    CampaignRunner,
    _acquire_execution_locks,
    _archive_completed_scope,
    _force_local_trial_logging,
    _release_execution_locks,
    _recover_multitask_from_csv,
    _restore_completed_scope,
    _scope_artifact_roots,
)
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


def _completion_path(
    root: Path,
    config_identity: str | None = None,
) -> Path:
    """Return the synthetic dataset completion path."""
    seed_root = root / "logs" / f"seed_{SEED}"
    if config_identity is not None:
        seed_root = (
            seed_root / "configurations" / config_identity
        )
    return (
        seed_root
        / "datasets"
        / DATASET_NAME
        / "completion.json"
    )


def _write_completion(
    root: Path,
    config_hash: str,
    execution_id: str = "execution",
    metric: float = 0.75,
    config_identity: str | None = None,
) -> Path:
    """Write one completion and its required checkpoint artifact."""
    path = _completion_path(root, config_identity)
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


def test_short_campaign_alias_requires_full_semantic_config(
    tmp_path: Path,
) -> None:
    """A colliding display hash cannot decide completion compatibility."""
    selected = _selected_config()
    config_hash = get_config_hash(selected, multitask=False)
    path = _write_completion(tmp_path, config_hash)
    completion = json.loads(path.read_text(encoding="utf-8"))
    completion["campaign_hash"] = "short-collision"
    path.write_text(json.dumps(completion), encoding="utf-8")

    rejected = check_completion_compatibility(
        path,
        CAMPAIGN_HASH,
        SEED,
        selected,
        campaign_aliases=("short-collision",),
    )

    assert rejected.compatible is False
    assert "resolved configuration does not exist" in rejected.reason

    _save_legacy_config(tmp_path, selected)
    accepted = check_completion_compatibility(
        path,
        CAMPAIGN_HASH,
        SEED,
        selected,
        campaign_aliases=("short-collision",),
    )

    assert accepted.compatible is True
    assert accepted.mode == "legacy_semantic_compatible"


def test_namespaced_completion_is_ignored(
    tmp_path: Path,
) -> None:
    """Legacy configuration namespaces never participate in discovery."""
    selected = _selected_config()
    config_hash = get_config_hash(selected, multitask=False)
    ignored_path = _write_completion(
        tmp_path,
        config_hash,
        config_identity=config_hash,
    )

    located = locate_completion(
        tmp_path,
        CAMPAIGN_HASH,
        SEED,
        DATASET_NAME,
        selected,
    )

    assert located.compatibility.mode == "missing"
    assert located.path != ignored_path
    assert ignored_path.is_file()


def test_direct_completion_wins_over_ignored_namespace(
    tmp_path: Path,
) -> None:
    """A legacy namespace cannot make one direct completion ambiguous."""
    selected = _selected_config()
    config_hash = get_config_hash(selected, multitask=False)
    direct_path = _write_completion(tmp_path, config_hash)
    _write_completion(
        tmp_path,
        config_hash,
        config_identity=config_hash,
    )

    located = locate_completion(
        tmp_path,
        CAMPAIGN_HASH,
        SEED,
        DATASET_NAME,
        selected,
    )

    assert located.compatibility.compatible is True
    assert located.path == direct_path


def test_changed_final_config_aborts_without_proven_budget_growth(
    tmp_path: Path,
) -> None:
    """A semantic mismatch cannot overwrite a completed direct result."""
    old_config = _selected_config()
    old_hash = get_config_hash(old_config, multitask=False)
    old_path = _write_completion(tmp_path, old_hash)
    original = old_path.read_bytes()
    selected = copy.deepcopy(old_config)
    selected["model"]["classifier_head"]["hidden_dims"] = [64]
    paths = CampaignPaths(tmp_path, tmp_path / "checkpoints")

    with pytest.raises(RuntimeError, match="cannot be overwritten"):
        _scope_artifact_roots(
            paths,
            CAMPAIGN_HASH,
            SEED,
            selected,
        )

    assert old_path.read_bytes() == original


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


def test_missing_checkpoint_does_not_invalidate_completion(
    tmp_path: Path,
) -> None:
    """Finite terminal metadata is authoritative without a checkpoint."""
    selected = _selected_config()
    config_hash = get_config_hash(selected, multitask=False)
    path = _write_completion(tmp_path, config_hash)
    completion = json.loads(path.read_text(encoding="utf-8"))
    Path(completion["checkpoint_path"]).unlink()

    result = check_completion_compatibility(
        path,
        CAMPAIGN_HASH,
        SEED,
        selected,
    )

    assert result.compatible is True


def test_completion_rejects_nonfinite_metrics(
    tmp_path: Path,
) -> None:
    """A non-finite terminal metric is incomplete and must be restarted."""
    selected = _selected_config()
    config_hash = get_config_hash(selected, multitask=False)
    path = _write_completion(
        tmp_path,
        config_hash,
        metric=float("nan"),
    )

    result = check_completion_compatibility(
        path,
        CAMPAIGN_HASH,
        SEED,
        selected,
    )

    assert result.compatible is False


def test_partial_summary_is_compact_and_logs_pair_diagnostics(
    tmp_path: Path,
    caplog,
    monkeypatch,
) -> None:
    """A valid pair is retained while missing details stay out of JSON."""
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(
        logging.getLogger("baseline"),
        "propagate",
        True,
    )
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

    assert summary.status["status"] == "partial"
    assert summary.status["dataset_pairs"] == {
        "expected": 2,
        "completed": 1,
        "missing": 1,
        "rejected": 0,
    }
    persisted = json.loads(
        (paths.summary_root / "summary.json").read_text(encoding="utf-8")
    )
    assert "compatibility" not in persisted
    assert "test_runs" not in persisted
    assert "dataset missing" in caplog.text


def test_zero_row_summary_preserves_previous_without_report(
    tmp_path: Path,
    caplog,
    monkeypatch,
) -> None:
    """No compatible rows log diagnostics without saving a report."""
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(
        logging.getLogger("baseline"),
        "propagate",
        True,
    )
    paths = CampaignPaths(tmp_path, tmp_path / "checkpoints")
    previous_path = paths.summary_root / "summary.json"
    previous_path.parent.mkdir(parents=True, exist_ok=True)
    previous_path.write_text("previous", encoding="utf-8")

    summary = write_campaign_summary(
        paths,
        CAMPAIGN_HASH,
        {"attempted": [SEED]},
        {(SEED, DATASET_NAME): _selected_config()},
    )

    assert summary.written is False
    assert previous_path.read_text(encoding="utf-8") == "previous"
    report_path = paths.summary_root / "compatibility_report.json"
    assert not report_path.exists()
    assert "completion.json does not exist" in caplog.text
    assert "left unchanged" in caplog.text


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


def test_execution_lock_rejects_a_concurrent_scope(
    tmp_path: Path,
) -> None:
    """Two invocations cannot execute one semantic seed scope together."""
    paths = CampaignPaths(tmp_path / "log", tmp_path / "checkpoints")
    scopes = {DATASET_NAME: _selected_config()}
    first = _acquire_execution_locks(paths, SEED, scopes)
    try:
        with pytest.raises(RuntimeError, match="execution lock"):
            _acquire_execution_locks(paths, SEED, scopes)
    finally:
        _release_execution_locks(first)
    second = _acquire_execution_locks(paths, SEED, scopes)
    _release_execution_locks(second)


def test_larger_hpo_budget_and_changed_winner_authorize_replacement(
    tmp_path: Path,
) -> None:
    """Replacement requires one study, budget growth, and a new winner."""
    old_config = _selected_config()
    completion_path = _write_completion(
        tmp_path,
        get_config_hash(old_config, multitask=False),
    )
    completion = json.loads(
        completion_path.read_text(encoding="utf-8")
    )
    completion["selection_provenance"] = {
        "source": "hpo",
        "scope": DATASET_NAME,
        "study_identity": "study",
        "effective_budget": 30,
        "parameter_digest": "old",
    }
    completion_path.write_text(
        json.dumps(completion),
        encoding="utf-8",
    )
    selected = copy.deepcopy(old_config)
    selected["model"]["classifier_head"]["hidden_dims"] = [64]
    located = locate_completion(
        tmp_path,
        CAMPAIGN_HASH,
        SEED,
        DATASET_NAME,
        selected,
    )
    runner = CampaignRunner.__new__(CampaignRunner)
    runner.paths = CampaignPaths(tmp_path, tmp_path / "checkpoints")
    runner.selection_provenance = {
        DATASET_NAME: {
            "source": "hpo",
            "scope": DATASET_NAME,
            "study_identity": "study",
            "effective_budget": 31,
            "parameter_digest": "new",
        },
    }

    assert runner._replacement_is_authorized(
        DATASET_NAME,
        SEED,
        [located],
    )
    runner.selection_provenance[DATASET_NAME]["effective_budget"] = 30
    assert not runner._replacement_is_authorized(
        DATASET_NAME,
        SEED,
        [located],
    )


def test_completed_replacement_archive_restores_after_failure(
    tmp_path: Path,
) -> None:
    """A failed new attempt restores the archived ordinary artifacts."""
    selected = _selected_config()
    completion_path = _write_completion(
        tmp_path,
        get_config_hash(selected, multitask=False),
    )
    original_completion = completion_path.read_bytes()
    csv_path = (
        tmp_path / "logs" / f"seed_{SEED}" / "csv"
        / f"{DATASET_NAME}.csv"
    )
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("old metrics", encoding="utf-8")
    checkpoint = (
        tmp_path / "checkpoints" / f"seed_{SEED}" / "seperated"
        / DATASET_NAME / "best.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("old checkpoint", encoding="utf-8")
    paths = CampaignPaths(tmp_path, tmp_path / "checkpoints")
    located = locate_completion(
        tmp_path,
        CAMPAIGN_HASH,
        SEED,
        DATASET_NAME,
        selected,
    )

    archive = _archive_completed_scope(
        paths,
        tmp_path / "invocations" / "replacement",
        SEED,
        DATASET_NAME,
        selected,
        [located],
    )
    assert archive is not None
    assert not completion_path.exists()
    completion_path.write_text("new partial", encoding="utf-8")
    csv_path.write_text("new partial", encoding="utf-8")
    checkpoint.write_text("new partial", encoding="utf-8")

    _restore_completed_scope(archive)

    assert completion_path.read_bytes() == original_completion
    assert csv_path.read_text(encoding="utf-8") == "old metrics"
    assert checkpoint.read_text(encoding="utf-8") == "old checkpoint"


def test_partial_multitask_completion_recovers_from_shared_csv(
    tmp_path: Path,
) -> None:
    """Unambiguous best-validation and test rows recover a missing pair."""
    second_dataset = "bcic_1a"
    selected = BrainOmniConfig(
        seeds=[SEED],
        multitask=True,
        data={
            "datasets": {
                DATASET_NAME: DATASET_CONFIG,
                second_dataset: DATASET_CONFIG,
            },
        },
    ).model_dump(mode="json")
    config_hash = get_config_hash(selected, multitask=True)
    first_path = _completion_path(tmp_path)
    first_path.parent.mkdir(parents=True, exist_ok=True)
    first_path.write_text(
        json.dumps({
            "status": "completed",
            "campaign_hash": CAMPAIGN_HASH,
            "config_hash": config_hash,
            "seed": SEED,
            "dataset_config": DATASET_CONFIG,
            "execution_id": "shared-run",
            "has_checkpoint": False,
            "checkpoint_path": None,
            "checkpoint_retention_requested": False,
            "validation_metrics": {
                f"{DATASET_NAME}/eval/epoch": 2,
                f"{DATASET_NAME}/eval/loss": 0.5,
            },
            "test_metrics": {
                f"{DATASET_NAME}/test/loss": 0.6,
            },
            "batching": {
                "requested_global_batch_size": 32,
                "world_size": 1,
                "micro_batch_size": 16,
                "accumulation_steps": 2,
            },
        }),
        encoding="utf-8",
    )
    second_path = (
        tmp_path / "logs" / f"seed_{SEED}" / "datasets"
        / second_dataset / "completion.json"
    )
    located = [
        locate_completion(
            tmp_path,
            CAMPAIGN_HASH,
            SEED,
            dataset_name,
            selected,
        )
        for dataset_name in (DATASET_NAME, second_dataset)
    ]
    csv_path = (
        tmp_path / "logs" / f"seed_{SEED}" / "csv" / "training.csv"
    )
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text(
        "timestamp,dataset,split,epoch,step,metric,value\n"
        f"now,{second_dataset},eval,2,3,loss,0.7\n"
        f"now,{second_dataset},test,4,3,loss,0.8\n",
        encoding="utf-8",
    )

    recovered = _recover_multitask_from_csv(
        CampaignPaths(tmp_path, tmp_path / "checkpoints"),
        CAMPAIGN_HASH,
        SEED,
        selected,
        located,
        {"source": "fixed"},
        "recovery-invocation",
    )

    assert recovered is True
    completion = json.loads(second_path.read_text(encoding="utf-8"))
    assert completion["test_metrics"] == {
        f"{second_dataset}/test/loss": 0.8,
    }
    assert completion["checkpoint_path"] is None
