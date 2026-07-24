"""Tests for semantic campaign identity and immutable campaign metadata."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from baseline.brainomni.brainomni_config import BrainOmniConfig
from baseline.hpo.artifacts import get_campaign_hash, resolve_campaign
from baseline.hpo.config import HpoConfig
from baseline.hpo.orchestrator import CampaignRunner


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


def test_campaign_identity_marks_searched_leaves_but_retains_semantics(
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
    assert get_campaign_hash(workers, hpo) != original

    search_change = copy.deepcopy(hpo)
    search_change["search_space"]["training.max_lr"]["high"] = 2e-3
    assert get_campaign_hash(config, search_change) != original


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
    hpo = _hpo_config().model_dump(mode="json")
    legacy_root = (
        tmp_path
        / "log"
        / "baseline"
        / "brainomni"
        / "manually-renamed-a5f8f3d25a11"
    )
    original = _write_legacy_campaign(legacy_root, config, hpo)
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
    resolution = resolve_campaign(
        str(tmp_path),
        "brainomni",
        "new-label",
        requested,
        hpo,
    )

    assert resolution.paths.log_root == legacy_root
    assert resolution.legacy is True
    assert "3f68277ec5ec" in resolution.aliases
    assert (legacy_root / "campaign.yaml").read_bytes() == original


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
