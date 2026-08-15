"""Tests for semantic campaign identity and immutable campaign metadata."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from baseline.brainomni.brainomni_config import BrainOmniConfig
from baseline.catch22.catch22_config import Catch22Config
from baseline.hpo.artifacts import (
    build_campaign_paths,
    collect_artifact_test_rows_with_diagnostics,
    get_campaign_hash,
    locate_completion,
    resolve_campaign,
)
from baseline.hpo.config import HpoConfig
from baseline.hpo.orchestrator import CampaignRunner
from baseline.utils.identity import build_campaign_semantic_config


TEST_FS = 256


def _hpo_config() -> HpoConfig:
    """Return a minimal enabled HPO configuration."""
    return HpoConfig.model_validate({
        "enabled": True,
        "seed": 0,
        "n_trials": 3,
        "max_consecutive_failed_trials": 5,
        "search_space": {
            "training.max_lr": {
                "distribution": "float",
                "low": 2e-4,
                "high": 1e-3,
                "log": True,
            },
        },
    })


def _model_config(
    run_dir: Path,
    experiment_name: str = "brainomni-test",
) -> BrainOmniConfig:
    """Return a resolved BrainOmni configuration under a temporary root."""
    return BrainOmniConfig(
        fs=TEST_FS,
        data={"datasets": {"adftd": "finetune"}},
        logging={
            "run_dir": str(run_dir),
            "experiment_name": experiment_name,
        },
    )


def test_campaign_identity_excludes_all_invocation_fields(
    tmp_path: Path,
) -> None:
    """Paths, ports, seeds, logging, and budgets never split campaigns."""
    config = _model_config(tmp_path).model_dump(mode="json")
    hpo = _hpo_config().model_dump(mode="json")
    original = get_campaign_hash(config, hpo)

    changed = copy.deepcopy(config)
    changed["conf_file"] = "/different/configuration.local.yaml"
    changed["master_port"] += 1
    changed["seeds"] = [7, 8, 9]
    changed["logging"] = {
        **changed["logging"],
        "experiment_name": "renamed",
        "level": "debug",
        "outputs": ["csv", "tensorboard"],
        "run_dir": str(tmp_path / "moved"),
    }
    changed_hpo = copy.deepcopy(hpo)
    changed_hpo["n_trials"] = 100
    changed_hpo["max_consecutive_failed_trials"] = 9

    assert get_campaign_hash(changed, changed_hpo) == original


def test_campaign_identity_marks_searched_leaves_and_excludes_loaders(
    tmp_path: Path,
) -> None:
    """Unused starting values are ignored while fixed numerics remain."""
    config = _model_config(tmp_path).model_dump(mode="json")
    hpo = _hpo_config().model_dump(mode="json")
    original = get_campaign_hash(config, hpo)

    searched_default = copy.deepcopy(config)
    searched_default["training"]["max_lr"] = 8e-4
    assert get_campaign_hash(searched_default, hpo) == original

    fixed_numeric = copy.deepcopy(config)
    fixed_numeric["training"]["weight_decay"] *= 2
    assert get_campaign_hash(fixed_numeric, hpo) != original

    workers = copy.deepcopy(config)
    workers["data"]["num_workers"] += 1
    assert get_campaign_hash(workers, hpo) == original

    pinned = copy.deepcopy(config)
    assert pinned["data"]["pin_memory"] is False
    pinned["data"]["pin_memory"] = True
    assert get_campaign_hash(pinned, hpo) == original

    legacy_default = copy.deepcopy(config)
    legacy_default["data"].pop("pin_memory")
    assert get_campaign_hash(legacy_default, hpo) == original

    search_change = copy.deepcopy(hpo)
    search_change["search_space"]["training.max_lr"]["high"] = 2e-3
    assert get_campaign_hash(config, search_change) != original


@pytest.mark.parametrize("path", ("data.num_workers", "data.pin_memory"))
def test_hpo_rejects_runtime_only_loader_paths(path: str) -> None:
    """Runtime-only loader settings cannot be HPO search parameters."""
    with pytest.raises(ValueError, match="runtime-only data-loader field"):
        HpoConfig.model_validate({
            "enabled": True,
            "n_trials": 1,
            "search_space": {
                path: {
                    "distribution": "int",
                    "low": 1,
                    "high": 2,
                },
            },
        })


def test_separate_task_identity_excludes_dataset_membership(
    tmp_path: Path,
) -> None:
    """Separate-task additions reuse the campaign while multitask does not."""
    config = _model_config(tmp_path).model_dump(mode="json")
    hpo = _hpo_config().model_dump(mode="json")
    original = get_campaign_hash(config, hpo)

    expanded = copy.deepcopy(config)
    expanded["data"]["datasets"]["bcic_2a"] = "finetune"
    assert get_campaign_hash(expanded, hpo) == original

    expanded["multitask"] = True
    assert get_campaign_hash(expanded, hpo) != original


def _write_legacy_campaign(
    root: Path,
    config: dict,
    hpo: dict,
) -> bytes:
    """Write a synthetic legacy full-config manifest and return its bytes."""
    root.mkdir(parents=True)
    payload = copy.deepcopy(config)
    payload["hpo"] = copy.deepcopy(hpo)
    path = root / "campaign.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return path.read_bytes()


def test_resolver_reuses_unique_legacy_root_without_rewriting(
    tmp_path: Path,
) -> None:
    """A renamed source file resolves to its unique historical root."""
    config = _model_config(tmp_path).model_dump(mode="json")
    config["conf_file"] = "/old/name.local.yaml"
    config["data"]["num_workers"] = 1
    hpo = _hpo_config().model_dump(mode="json")
    legacy_root = (
        tmp_path
        / "log"
        / "baseline"
        / "brainomni"
        / "manually-renamed-a5f8f3d25a11"
    )
    original = _write_legacy_campaign(legacy_root, config, hpo)
    identity_path = legacy_root / "identity.json"
    identity_path.write_text(
        json.dumps({
            "campaign_identity": "a" * 64,
            "identity_version": 2,
        }),
        encoding="utf-8",
    )
    original_identity = identity_path.read_bytes()

    completion_path = (
        legacy_root
        / "logs"
        / "seed_42"
        / "datasets"
        / "adftd"
        / "completion.json"
    )
    completion_path.parent.mkdir(parents=True)
    completion_path.write_text(
        json.dumps({"campaign_hash": "3f68277ec5ec"}),
        encoding="utf-8",
    )


    requested = copy.deepcopy(config)
    requested["conf_file"] = "/new/name.local.yaml"
    requested["master_port"] += 10
    requested["logging"]["experiment_name"] = "new-label"
    requested["data"]["datasets"]["bcic_2a"] = "finetune"
    requested["data"]["num_workers"] = 8
    resolution = resolve_campaign(
        str(tmp_path),
        "brainomni",
        "new-label",
        requested,
        hpo,
    )

    assert resolution.paths.log_root == legacy_root
    assert resolution.legacy is True
    assert "a" * 64 in resolution.aliases
    assert "3f68277ec5ec" in resolution.aliases
    assert (legacy_root / "campaign.yaml").read_bytes() == original
    assert identity_path.read_bytes() == original_identity


def test_resolver_prefers_existing_current_campaign_root(
    tmp_path: Path,
) -> None:
    """Legacy discovery runs only after the canonical root is absent."""
    config = _model_config(tmp_path).model_dump(mode="json")
    hpo = _hpo_config().model_dump(mode="json")
    campaign_hash = get_campaign_hash(config, hpo)
    model_root = tmp_path / "log" / "baseline" / "brainomni"
    current_root = model_root / f"brainomni-test-{campaign_hash[:12]}"
    current_root.mkdir(parents=True)
    (current_root / "campaign.yaml").write_text(
        yaml.safe_dump(
            build_campaign_semantic_config(config, hpo),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_legacy_campaign(model_root / "legacy-root", config, hpo)

    resolution = resolve_campaign(
        str(tmp_path),
        "brainomni",
        "brainomni-test",
        config,
        hpo,
    )

    assert resolution.paths.log_root == current_root
    assert resolution.legacy is False


def test_resolver_aborts_for_ambiguous_semantic_roots(
    tmp_path: Path,
) -> None:
    """Two exact semantic roots require explicit human resolution."""
    config = _model_config(tmp_path).model_dump(mode="json")
    hpo = _hpo_config().model_dump(mode="json")
    model_root = tmp_path / "log" / "baseline" / "brainomni"
    _write_legacy_campaign(model_root / "first-oldhash", config, hpo)
    _write_legacy_campaign(model_root / "second-oldhash", config, hpo)

    with pytest.raises(RuntimeError, match="Multiple campaign roots"):
        resolve_campaign(
            str(tmp_path),
            "brainomni",
            "requested",
            config,
            hpo,
        )


def test_legacy_manifest_remains_immutable_when_invocation_is_added(
    tmp_path: Path,
) -> None:
    """Adoption adds identity and invocation files without rewriting YAML."""
    config = _model_config(tmp_path)
    config_dict = config.model_dump(mode="json")
    model_root = tmp_path / "log" / "baseline" / "brainomni"
    legacy_root = model_root / "legacy-123456789abc"
    original = _write_legacy_campaign(
        legacy_root,
        config_dict,
        HpoConfig().model_dump(mode="json"),
    )
    runner = CampaignRunner(config, HpoConfig(), BrainOmniConfig)

    runner._save_campaign_config()

    assert runner.paths.log_root == legacy_root
    assert (legacy_root / "campaign.yaml").read_bytes() == original
    assert (legacy_root / "identity.json").is_file()
    assert (runner.invocation_root / "invocation.yaml").is_file()
    assert (runner.invocation_root / "status.json").is_file()


def test_legacy_identity_remains_immutable_when_semantics_evolve(
    tmp_path: Path,
) -> None:
    """Dataset-independent adoption does not replace legacy identity data."""
    config = _model_config(tmp_path)
    config_dict = config.model_dump(mode="json")
    model_root = tmp_path / "log" / "baseline" / "brainomni"
    legacy_root = model_root / "legacy-identity"
    _write_legacy_campaign(
        legacy_root,
        config_dict,
        HpoConfig().model_dump(mode="json"),
    )
    identity_path = legacy_root / "identity.json"
    identity_path.write_text(
        json.dumps({
            "campaign_identity": "legacy-identity",
            "identity_version": 2,
        }),
        encoding="utf-8",
    )
    original_identity = identity_path.read_bytes()
    runner = CampaignRunner(config, HpoConfig(), BrainOmniConfig)

    runner._save_campaign_config()

    assert runner.resolution.legacy is True
    assert identity_path.read_bytes() == original_identity


def test_new_campaign_separates_semantic_and_invocation_parameters(
    tmp_path: Path,
) -> None:
    """New manifests are semantic-only and invocations are complete."""
    config = _model_config(tmp_path)
    runner = CampaignRunner(config, HpoConfig(), BrainOmniConfig)

    runner._save_campaign_config()

    campaign = yaml.safe_load(
        (runner.paths.log_root / "campaign.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert "conf_file" not in campaign
    assert "logging" not in campaign
    assert "master_port" not in campaign
    assert "seeds" not in campaign
    invocation = yaml.safe_load(
        (runner.invocation_root / "invocation.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert invocation["model_config"]["master_port"] == config.master_port
    assert invocation["model_config"]["seeds"] == config.seeds
    identity = json.loads(
        (runner.paths.log_root / "identity.json").read_text(
            encoding="utf-8"
        )
    )
    assert identity["campaign_identity"] == runner.campaign_hash


def test_audit_mode_does_not_create_campaign_artifacts(
    tmp_path: Path,
) -> None:
    """Audit resolves a future campaign without creating its directory."""
    config = _model_config(tmp_path)
    runner = CampaignRunner(config, HpoConfig(), BrainOmniConfig)

    report = runner.audit()

    assert report["campaign_identity"] == runner.campaign_hash
    assert not runner.paths.log_root.exists()
    assert not runner.paths.checkpoint_root.exists()


def test_deterministic_identity_excludes_declared_runtime_fields(
    tmp_path: Path,
) -> None:
    """Runtime-only deterministic settings do not split campaigns."""
    config = Catch22Config(
        fs=TEST_FS,
        data={"datasets": {"adftd": "finetune"}},
        logging={"run_dir": str(tmp_path)},
    ).model_dump(mode="json")
    original = get_campaign_hash(config, {})
    changed = copy.deepcopy(config)
    changed["data"]["batch_size"] = 64
    changed["data"]["load_batch_size"] = 64
    changed["data"]["num_workers"] = 8
    changed["data"]["datasets"]["bcic_2a"] = "finetune"
    changed["model"]["extractor"]["n_jobs"] = 2
    assert get_campaign_hash(changed, {}) == original

    semantic = copy.deepcopy(config)
    semantic["model"]["classifier"]["alphas"] = [0.1]
    assert get_campaign_hash(semantic, {}) != original

    paths = build_campaign_paths(
        str(tmp_path),
        "catch22",
        "catch22",
        original,
    )
    assert paths.flat_results is True
    assert paths.seed_log_root(42) == paths.log_root


def test_campaign_adopts_matching_legacy_flat_extractor_root(
    tmp_path: Path,
) -> None:
    """One completed flat extractor root is reused without result copying."""
    config = Catch22Config(
        fs=TEST_FS,
        data={"datasets": {"adftd": "finetune"}},
        logging={"run_dir": str(tmp_path), "experiment_name": "catch22"},
    ).model_dump(mode="json")
    legacy_root = tmp_path / "log" / "baseline" / "catch22" / "legacy"
    config_path = legacy_root / "configs" / "legacy-run.yaml"
    config_path.parent.mkdir(parents=True)
    saved = copy.deepcopy(config)
    saved["seed"] = saved.pop("seeds")[0]
    config_path.write_text(yaml.safe_dump(saved), encoding="utf-8")
    completion_path = legacy_root / "datasets" / "adftd" / "completion.json"
    completion_path.parent.mkdir(parents=True)
    completion_path.write_text(
        json.dumps({
            "status": "completed",
            "dataset_config": "finetune",
            "execution_id": "legacy-run",
            "invocation_id": "current-invocation",
            "model_type": "catch22",
            "has_checkpoint": False,
            "checkpoint_path": None,
            "test_metrics": {"adftd/test/balanced_acc": 0.5},
        }),
        encoding="utf-8",
    )

    resolution = resolve_campaign(
        str(tmp_path),
        "catch22",
        "catch22",
        config,
        {},
    )
    located = locate_completion(
        resolution.paths.log_root,
        resolution.campaign_identity,
        42,
        "adftd",
        config,
        resolution.aliases,
    )

    assert resolution.paths.log_root == legacy_root
    assert resolution.legacy is True
    assert resolution.paths.flat_results is True
    assert resolution.paths.seed_log_root(42) == legacy_root
    assert located.compatibility.compatible
    assert located.compatibility.mode == "legacy_flat_semantic_compatible"
    rows, diagnostics = collect_artifact_test_rows_with_diagnostics(
        legacy_root,
        resolution.campaign_identity,
        resolution.aliases,
        invocation_id="current-invocation",
    )
    assert len(rows) == 1
    assert len(diagnostics["accepted"]) == 1
