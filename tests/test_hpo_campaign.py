"""Tests for HPO configuration, campaign failures, and seed summaries."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

import baseline.abstract.trainer as trainer_module
import baseline.hpo.orchestrator as orchestrator_module
from baseline.abstract.trainer import AbstractTrainer, HpoStopReason
from baseline.brainomni.brainomni_config import BrainOmniConfig
from baseline.hpo.artifacts import (
    CampaignPaths,
    collect_test_rows,
    failure_fingerprint,
    get_campaign_hash,
    summarize_test_rows,
)
from baseline.hpo.config import HpoConfig, ProgressiveHpoArgs
from baseline.hpo.progressive import (
    assess_progressive_study,
    has_complete_progressive_evidence,
    progressive_assessment_block,
)
from baseline.hpo.orchestrator import (
    CampaignExecutionError,
    CampaignRunner,
    ENCODER_LR_SCALE_PATH,
    _collect_completed_trials,
    _project_hpo_parameters,
    _hpo_scope_root,
    _study_record_matches_semantics,
    _study_progress,
    _study_scope_configs,
    _trial_metadata_matches,
    _validate_immutable_hpo_best,
)
from baseline.hpo.search import (
    reduce_objective,
    sample_config,
    validate_search_space,
)
from baseline.registry import register_builtin_models
from baseline_main import _load_configs, _normalize_legacy_seed
from common.utils import setup_yaml

TEST_FS = 256


class FakeTrial:
    """Deterministic trial implementing the search helper protocol."""

    def suggest_float(
        self,
        name,
        low,
        high,
        *,
        step=None,
        log=False,
    ):
        del name, high, step, log
        return low

    def suggest_int(
        self,
        name,
        low,
        high,
        *,
        step=1,
        log=False,
    ):
        del name, high, step, log
        return low

    def suggest_categorical(self, name, choices):
        del name
        return choices[-1]


def make_hpo_config() -> HpoConfig:
    """Return the representative BrainOmni search configuration."""
    return HpoConfig.model_validate({
        "enabled": True,
        "seed": 0,
        "n_trials": 5,
        "objective": {
            "metric": "loss",
            "direction": "minimize",
            "multitask_reduction": "macro_mean",
        },
        "search_space": {
            "training.max_lr": {
                "distribution": "float",
                "low": 2e-4,
                "high": 1e-3,
                "log": True,
            },
            "training.min_lr": {
                "distribution": "float",
                "low": 1e-6,
                "high": 1e-4,
                "log": True,
            },
            "training.encoder_lr_scale": {
                "distribution": "float",
                "low": 0.05,
                "high": 1.0,
                "log": False,
            },
            "model.classifier_head.hidden_dims": {
                "distribution": "categorical",
                "choices": [[64], [128], [256, 128]],
            },
        },
    })


def test_public_seeds_default_and_effective_seed_contract() -> None:
    """Public configs use a list while trainers receive one effective seed."""
    assert BrainOmniConfig(fs=TEST_FS).seeds == [42]
    assert BrainOmniConfig(fs=TEST_FS).seed == 42
    with pytest.raises(RuntimeError, match="exactly one effective seed"):
        _ = BrainOmniConfig(fs=TEST_FS, seeds=[42, 43]).seed


@pytest.mark.parametrize("seeds", ([True], ["42"], 42))
def test_public_seeds_reject_coerced_types(seeds) -> None:
    """Boolean, string, and scalar seed inputs are not silently coerced."""
    with pytest.raises(ValidationError, match="seeds must"):
        BrainOmniConfig(fs=TEST_FS, seeds=seeds)


def test_legacy_seed_converts_with_warning() -> None:
    """Entrypoint conversion retains old scalar YAMLs for one release."""
    config = OmegaConf.create({"seed": 7})
    with pytest.warns(DeprecationWarning, match="deprecated scalar"):
        _normalize_legacy_seed(config, "test")
    assert OmegaConf.to_container(config) == {"seeds": [7]}


def test_public_yaml_accepts_cli_hpo_budget_override(
    monkeypatch,
) -> None:
    """CLI HPO settings override YAML without affecting evaluation seeds."""
    config_path = (
        Path(__file__).parents[1]
        / "assets"
        / "conf"
        / "baseline"
        / "brainomni"
        / "brainomni_unified.yaml"
    ).resolve()
    register_builtin_models()
    setup_yaml()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "baseline_main.py",
            f"conf_file={config_path}",
            "model_type=brainomni",
            "hpo.n_trials=7",
        ],
    )

    _, config, hpo = _load_configs()

    assert config.seeds == [42]
    assert hpo.seed == 0
    assert hpo.n_trials == 7


def test_campaign_identity_excludes_seeds_and_trial_budget() -> None:
    """Adding evaluation seeds or HPO trials resumes the same campaign."""
    base = BrainOmniConfig(fs=TEST_FS, seeds=[42]).model_dump(mode="json")
    hpo = make_hpo_config().model_dump(mode="json")
    original = get_campaign_hash(base, hpo)

    base["seeds"] = [42, 43, 44]
    hpo["n_trials"] = 100
    hpo["max_consecutive_failed_trials"] = 9
    base["logging"]["level"] = "debug"
    assert get_campaign_hash(base, hpo) == original

    hpo["search_space"]["training.max_lr"]["high"] = 2e-3
    assert get_campaign_hash(base, hpo) != original


def test_enabled_hpo_requires_budget_and_search_space() -> None:
    """An enabled study cannot silently use an implicit compute budget."""
    with pytest.raises(ValidationError, match="hpo.n_trials is required"):
        HpoConfig.model_validate({
            "enabled": True,
            "search_space": {
                "training.max_lr": {
                    "distribution": "float",
                    "low": 1e-4,
                    "high": 1e-3,
                },
            },
        })


def test_search_sampling_changes_only_selected_paths() -> None:
    """Dotted search paths preserve every unselected model parameter."""
    base = BrainOmniConfig(fs=TEST_FS, seeds=[0]).model_dump(mode="json")
    hpo = make_hpo_config()
    validate_search_space(base, hpo, BrainOmniConfig)
    sampled, decoded = sample_config(base, hpo, FakeTrial())

    assert sampled["training"]["max_lr"] == 2e-4
    assert sampled["training"]["encoder_lr_scale"] == 0.05
    assert sampled["model"]["classifier_head"]["hidden_dims"] == [
        256,
        128,
    ]
    assert sampled["training"]["weight_decay"] == (
        base["training"]["weight_decay"]
    )
    assert decoded["training.encoder_lr_scale"] == 0.05
    assert (
        hpo.search_space["training.encoder_lr_scale"].log is False
    )


def test_search_rejects_overlapping_learning_rate_ranges() -> None:
    """Every sampled min_lr must remain at or below sampled max_lr."""
    base = BrainOmniConfig(fs=TEST_FS, seeds=[0]).model_dump(mode="json")
    raw = make_hpo_config().model_dump(mode="json")
    raw["search_space"]["training.min_lr"]["high"] = 3e-4
    hpo = HpoConfig.model_validate(raw)
    with pytest.raises(ValueError, match="min_lr.high"):
        validate_search_space(base, hpo, BrainOmniConfig)


def test_multitask_objective_reducers() -> None:
    """Macro and log-size reductions follow their documented formulas."""
    values = {"small": 1.0, "large": 3.0}
    assert reduce_objective(values, "macro_mean", {}) == 2.0
    reduced = reduce_objective(
        values,
        "log_train_size_weighted",
        {"small": 10, "large": 100},
    )
    expected = (
        math.log(10) + 3.0 * math.log(100)
    ) / (math.log(10) + math.log(100))
    assert reduced == pytest.approx(expected)


def test_summary_omits_std_until_two_seeds_exist() -> None:
    """Single-seed summaries contain no placeholder or non-finite std."""
    first = summarize_test_rows([
        {
            "dataset": "demo",
            "seed": 42,
            "metric": "acc",
            "value": 0.5,
        },
    ])
    assert first == [{
        "dataset": "demo",
        "metric": "acc",
        "count": 1,
        "mean": 0.5,
        "median": 0.5,
    }]

    second = summarize_test_rows([
        {
            "dataset": "demo",
            "seed": 42,
            "metric": "acc",
            "value": 0.5,
        },
        {
            "dataset": "demo",
            "seed": 43,
            "metric": "acc",
            "value": 0.7,
        },
    ])
    assert second[0]["std"] == pytest.approx(math.sqrt(0.02))


def test_failure_fingerprint_removes_runtime_details() -> None:
    """Equivalent seed failures receive one normalized fingerprint."""
    first = RuntimeError(
        "seed=42 rank=0 failed at /tmp/run/seed_42/model.pt pid 12345"
    )
    second = RuntimeError(
        "seed=43 rank=1 failed at /data/run/seed_43/model.pt pid 98765"
    )
    assert failure_fingerprint(first) == failure_fingerprint(second)


def make_failure_runner(
    tmp_path: Path,
    monkeypatch,
    outcomes,
) -> CampaignRunner:
    """Create a campaign runner whose seed executions use fake outcomes."""
    runner = CampaignRunner.__new__(CampaignRunner)
    runner.base_dict = {
        "multitask": True,
        "seeds": [42, 43, 44, 45],
    }
    runner.paths = CampaignPaths(
        log_root=tmp_path / "log",
        checkpoint_root=tmp_path / "ckpt",
    )
    runner.campaign_hash = "campaign"
    runner.campaign_aliases = frozenset()
    monkeypatch.setattr(
        orchestrator_module,
        "_seed_scope_is_complete",
        lambda *args, **kwargs: False,
    )

    def run_seed(seed, selected):
        del selected
        outcome = outcomes.get(seed)
        if outcome is False:
            return False
        if outcome is not None:
            raise outcome
        return True

    runner._run_seed = run_seed
    return runner


def test_repeated_consecutive_failure_stops_after_second(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A success permits partial summary eligibility before repeated failure."""
    runner = make_failure_runner(
        tmp_path,
        monkeypatch,
        {
            43: RuntimeError("rank=0 failed at /tmp/a pid 12345"),
            44: RuntimeError("rank=1 failed at /tmp/b pid 54321"),
        },
    )
    invocation, eligible = runner._run_seeds({"fixed": {}})
    assert invocation["attempted"] == [42, 43, 44]
    assert invocation["succeeded"] == [42]
    assert invocation["unattempted"] == [45]
    assert eligible is True


def test_first_attempt_failure_prevents_summary_update(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Later success cannot replace a summary when the first attempt failed."""
    runner = make_failure_runner(
        tmp_path,
        monkeypatch,
        {42: ValueError("first seed failed")},
    )
    invocation, eligible = runner._run_seeds({"fixed": {}})
    assert invocation["succeeded"] == [43, 44, 45]
    assert eligible is False


def test_success_resets_failure_streak(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Matching errors separated by success are not consecutive."""
    runner = make_failure_runner(
        tmp_path,
        monkeypatch,
        {
            43: RuntimeError("same failure"),
            45: RuntimeError("same failure"),
        },
    )
    invocation, eligible = runner._run_seeds({"fixed": {}})
    assert invocation["attempted"] == [42, 43, 44, 45]
    assert invocation["unattempted"] == []
    assert eligible is True


def test_skipped_seed_does_not_become_first_attempt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Summary eligibility uses the first executed seed, not a skipped one."""
    runner = make_failure_runner(
        tmp_path,
        monkeypatch,
        {
            42: False,
            43: ValueError("first attempted seed failed"),
        },
    )
    invocation, eligible = runner._run_seeds({"fixed": {}})
    assert invocation["skipped"] == [42]
    assert invocation["attempted"] == [43, 44, 45]
    assert invocation["succeeded"] == [44, 45]
    assert eligible is False


def _write_completion(
    campaign_root: Path,
    seed: int,
    dataset_name: str,
    config_hash: str,
    value: float,
) -> None:
    """Write one synthetic dataset completion for summary tests."""
    path = (
        campaign_root
        / "logs"
        / f"seed_{seed}"
        / "datasets"
        / dataset_name
        / "completion.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "status": "completed",
            "campaign_hash": "campaign",
            "config_hash": config_hash,
            "seed": seed,
            "has_checkpoint": False,
            "test_metrics": {
                f"{dataset_name}/test/acc": value,
            },
        }),
        encoding="utf-8",
    )


def test_summary_accepts_compatible_dataset_seed_pairs(
    tmp_path: Path,
) -> None:
    """Compatible pairs survive missing or stale sibling datasets."""
    _write_completion(tmp_path, 42, "alpha", "alpha-42", 0.1)
    _write_completion(tmp_path, 43, "alpha", "alpha-43", 0.2)
    _write_completion(tmp_path, 43, "beta", "beta-43", 0.3)
    _write_completion(tmp_path, 44, "alpha", "stale", 0.4)
    _write_completion(tmp_path, 44, "beta", "beta-44", 0.5)
    compatible = {
        (42, "alpha"): "alpha-42",
        (42, "beta"): "beta-42",
        (43, "alpha"): "alpha-43",
        (43, "beta"): "beta-43",
        (44, "alpha"): "alpha-44",
        (44, "beta"): "beta-44",
    }

    rows = collect_test_rows(tmp_path, "campaign", compatible)

    assert {
        (row["seed"], row["dataset"])
        for row in rows
    } == {
        (42, "alpha"),
        (43, "alpha"),
        (43, "beta"),
        (44, "beta"),
    }


def test_single_task_and_multitask_hpo_scopes_and_paths(
    tmp_path: Path,
) -> None:
    """Single-task studies split by dataset; multitask uses one joint path."""
    base = BrainOmniConfig(
        fs=TEST_FS,
        seeds=[42, 43],
        data={
            "datasets": {
                "alpha": "finetune",
                "beta": "finetune",
            },
        },
    ).model_dump(mode="json")
    scopes = _study_scope_configs(base)
    assert list(scopes) == ["alpha", "beta"]
    assert scopes["alpha"]["data"]["datasets"] == {
        "alpha": "finetune"
    }
    assert scopes["alpha"]["seeds"] == [42, 43]
    assert _hpo_scope_root(tmp_path, "alpha") == (
        tmp_path / "hpo" / "datasets" / "alpha"
    )

    base["multitask"] = True
    assert list(_study_scope_configs(base)) == ["multitask"]
    assert _hpo_scope_root(tmp_path, "multitask") == (
        tmp_path / "hpo" / "multitask"
    )


def test_optuna_sqlite_study_resumes_existing_trials(
    tmp_path: Path,
) -> None:
    """Recreating a campaign study loads its existing SQLite trial state."""
    runner = CampaignRunner.__new__(CampaignRunner)
    runner.paths = CampaignPaths(
        log_root=tmp_path / "log",
        checkpoint_root=tmp_path / "ckpt",
    )
    runner.campaign_hash = "campaign"
    runner.campaign_aliases = frozenset()
    runner.hpo_config = make_hpo_config()

    first_runtime = runner._create_study("alpha")
    first_runtime.study.optimize(lambda trial: 1.0, n_trials=1)
    runner._release_study(first_runtime)
    second_runtime = runner._create_study("alpha")

    assert len(second_runtime.study.trials) == 1
    runner._release_study(second_runtime)
    assert (
        tmp_path
        / "log"
        / "hpo"
        / "datasets"
        / "alpha"
        / "studies"
        / first_runtime.study_identity
        / "study.sqlite3"
    ).is_file()


def test_frozen_hpo_accepts_legacy_inactive_trial_parameter() -> None:
    """A frozen trial may contain the historical inert encoder LR scale."""
    import optuna

    requested = make_hpo_config()
    effective_payload = requested.model_dump(mode="json")
    effective_payload["search_space"].pop(ENCODER_LR_SCALE_PATH)
    effective = HpoConfig.model_validate(effective_payload)
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    decoded = {}
    for path, distribution in requested.search_space.items():
        if distribution.distribution == "categorical":
            encoded = trial.suggest_categorical(
                path,
                ["choice_0000", "choice_0001", "choice_0002"],
            )
            index = int(encoded.rsplit("_", maxsplit=1)[-1])
            decoded[path] = distribution.choices[index]
        else:
            decoded[path] = trial.suggest_float(
                path,
                float(distribution.low),
                float(distribution.high),
                log=distribution.log,
            )
    trial.set_user_attr("decoded_params", decoded)
    study.tell(trial, 1.0)

    assert _trial_metadata_matches(
        study.trials,
        effective,
        frozenset({ENCODER_LR_SCALE_PATH}),
    )
    projected = _project_hpo_parameters(
        decoded,
        effective,
        frozenset({ENCODER_LR_SCALE_PATH}),
    )
    assert set(projected) == set(effective.search_space)
    assert ENCODER_LR_SCALE_PATH not in projected


def _frozen_effective_hpo_config() -> HpoConfig:
    """Return the test HPO space without its inactive frozen parameter."""
    payload = make_hpo_config().model_dump(mode="json")
    payload["search_space"].pop(ENCODER_LR_SCALE_PATH)
    return HpoConfig.model_validate(payload)


def _winner_payload(parameters: dict[str, Any]) -> dict[str, Any]:
    """Return one deterministic immutable HPO winner payload."""
    return {
        "study_identity": "historical-study",
        "trial_number": 27,
        "objective": 0.5,
        "parameters": parameters,
    }


def test_immutable_winner_accepts_inactive_historical_parameter(
    tmp_path: Path,
) -> None:
    """An inactive historical parameter does not block winner reuse."""
    hpo = _frozen_effective_hpo_config()
    active_parameters = {
        "model.classifier_head.hidden_dims": [128],
        "training.max_lr": 5e-4,
        "training.min_lr": 5e-6,
    }
    historical_parameters = {
        **active_parameters,
        ENCODER_LR_SCALE_PATH: 0.75,
    }
    path = tmp_path / "best_trial_00027.json"
    original = json.dumps(
        _winner_payload(historical_parameters),
        indent=2,
        sort_keys=True,
    )
    path.write_text(original, encoding="utf-8")

    _validate_immutable_hpo_best(
        path,
        _winner_payload(active_parameters),
        hpo,
        frozenset({ENCODER_LR_SCALE_PATH}),
    )

    assert path.read_text(encoding="utf-8") == original


def test_immutable_winner_rejects_active_parameter_mismatch(
    tmp_path: Path,
) -> None:
    """A changed active parameter still blocks historical winner reuse."""
    hpo = _frozen_effective_hpo_config()
    expected_parameters = {
        "model.classifier_head.hidden_dims": [128],
        "training.max_lr": 5e-4,
        "training.min_lr": 5e-6,
    }
    historical_parameters = {
        **expected_parameters,
        "training.max_lr": 8e-4,
        ENCODER_LR_SCALE_PATH: 0.75,
    }
    path = tmp_path / "best_trial_00027.json"
    path.write_text(
        json.dumps(
            _winner_payload(historical_parameters),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="mismatching immutable HPO winner",
    ):
        _validate_immutable_hpo_best(
            path,
            _winner_payload(expected_parameters),
            hpo,
            frozenset({ENCODER_LR_SCALE_PATH}),
        )


def test_unfrozen_hpo_rejects_legacy_extra_trial_parameter() -> None:
    """An active search cannot silently accept an unrelated extra path."""
    import optuna

    hpo = make_hpo_config()
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    trial.suggest_float("training.unrelated", 0.1, 1.0)
    trial.set_user_attr("decoded_params", {"training.unrelated": 0.1})
    study.tell(trial, 1.0)

    assert not _trial_metadata_matches(study.trials, hpo)


def test_legacy_root_accepts_older_campaign_identity() -> None:
    """A compatible legacy root may contain an older identity schema."""
    record = {
        "semantic_payload_present": True,
        "semantic_payload_valid": True,
        "scope": "alpha",
        "campaign_identity": "older-campaign",
    }

    assert not _study_record_matches_semantics(
        record,
        "alpha",
        "current-study",
        frozenset({"current-campaign"}),
    )
    assert _study_record_matches_semantics(
        record,
        "alpha",
        "current-study",
        frozenset({"current-campaign"}),
        allow_historical_campaign_identity=True,
    )


def test_legacy_root_still_rejects_wrong_scope() -> None:
    """Legacy identity compatibility cannot cross HPO scopes."""
    record = {
        "semantic_payload_present": True,
        "semantic_payload_valid": True,
        "scope": "beta",
        "campaign_identity": "older-campaign",
    }

    assert not _study_record_matches_semantics(
        record,
        "alpha",
        "current-study",
        frozenset({"current-campaign"}),
        allow_historical_campaign_identity=True,
    )


def test_resuming_hpo_removes_failed_trials_and_their_artifacts(
    tmp_path: Path,
) -> None:
    """Failed and stale trials are removed before the next trial is asked."""
    import optuna

    runner = _study_test_runner(tmp_path)
    runner.hpo_config = make_hpo_config().model_copy(
        update={"n_trials": 5}
    )
    first_runtime = runner._create_study("alpha")
    for _ in range(2):
        trial = first_runtime.study.ask()
        first_runtime.study.tell(trial, 1.0)
    failed_numbers = []
    for _ in range(2):
        trial = first_runtime.study.ask()
        failed_numbers.append(trial.number)
        first_runtime.study.tell(
            trial,
            state=optuna.trial.TrialState.FAIL,
        )
    running_trial = first_runtime.study.ask()
    failed_numbers.append(running_trial.number)
    for trial_number in failed_numbers:
        artifact = (
            first_runtime.artifact_root
            / "trials"
            / f"trial_{trial_number:05d}"
        )
        artifact.mkdir(parents=True)
        (artifact / "trial.json").write_text("{}", encoding="utf-8")
        checkpoint = (
            first_runtime.checkpoint_root
            / f"trial_{trial_number:05d}"
        )
        checkpoint.mkdir(parents=True)
        (checkpoint / "checkpoint.pt").write_text("partial", encoding="utf-8")
    runner._release_study(first_runtime)

    second_runtime = runner._create_study("alpha")
    try:
        assert [trial.number for trial in second_runtime.study.trials] == [
            0,
            1,
        ]
        resumed_trial = second_runtime.study.ask()
        assert resumed_trial.number == 2
        second_runtime.study.tell(resumed_trial, 1.0)
        for trial_number in failed_numbers:
            assert not (
                second_runtime.artifact_root
                / "trials"
                / f"trial_{trial_number:05d}"
            ).exists()
            assert not (
                second_runtime.checkpoint_root
                / f"trial_{trial_number:05d}"
            ).exists()
    finally:
        runner._release_study(second_runtime)


def test_completed_hpo_budget_reuses_winner_without_trainer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A complete persisted budget selects its winner without calibration."""
    runner = CampaignRunner.__new__(CampaignRunner)
    runner.paths = CampaignPaths(
        log_root=tmp_path / "log",
        checkpoint_root=tmp_path / "ckpt",
    )
    runner.campaign_hash = "campaign"
    runner.campaign_aliases = frozenset()
    runner.hpo_config = make_hpo_config().model_copy(
        update={"n_trials": 1}
    )
    scope_config = BrainOmniConfig(fs=TEST_FS).model_dump(mode="json")
    runner.base_dict = scope_config
    runner.config_class = BrainOmniConfig

    runner.invocation_root = tmp_path / "log" / "invocations" / "test"
    runner.invocation_root.mkdir(parents=True)
    runner._active_study_runtime = None
    runtime = runner._create_study("alpha")
    trial = runtime.study.ask()
    _, decoded = sample_config(
        scope_config,
        runner.hpo_config,
        trial,
    )
    trial.set_user_attr("decoded_params", decoded)
    runtime.study.tell(trial, 1.0)
    runner._release_study(runtime)

    def reject_training(*args, **kwargs):
        del args, kwargs
        raise AssertionError("trainer must not be constructed")

    monkeypatch.setattr(runner, "_run_adaptive_trainer", reject_training)

    selected = runner._run_hpo_scope("alpha", scope_config)

    expected = json.loads(json.dumps(scope_config))
    for dotted_path, value in decoded.items():
        orchestrator_module.set_dotted_value(
            expected,
            dotted_path,
            value,
        )
    assert selected == expected
    assert (
        tmp_path
        / "log"
        / "hpo"
        / "datasets"
        / "alpha"
        / "studies"
        / runtime.study_identity
        / "best_trial_00000.json"
    ).is_file()


def test_completed_versioned_study_survives_identity_schema_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A completed predecessor is selected across identity versions."""
    import optuna

    runner = CampaignRunner.__new__(CampaignRunner)
    runner.paths = CampaignPaths(
        log_root=tmp_path / "log",
        checkpoint_root=tmp_path / "ckpt",
    )
    runner.campaign_hash = "current-campaign"
    runner.campaign_aliases = frozenset({"historic-campaign"})
    runner.hpo_config = make_hpo_config().model_copy(
        update={"n_trials": 1}
    )
    scope_config = BrainOmniConfig(fs=TEST_FS).model_dump(mode="json")
    runner.base_dict = scope_config
    runner.config_class = BrainOmniConfig
    runner.invocation_root = tmp_path / "log" / "invocations" / "test"
    runner.invocation_root.mkdir(parents=True)
    runner._active_study_runtime = None

    scope_root = _hpo_scope_root(runner.paths.log_root, "alpha")
    historic_identity = "historic-study"
    historic_root = scope_root / "studies" / historic_identity
    historic_root.mkdir(parents=True)
    storage_path = (historic_root / "study.sqlite3").resolve()
    historic = optuna.create_study(
        study_name=historic_identity,
        storage=f"sqlite:///{storage_path}",
        direction="minimize",
    )
    historic.set_user_attr("identity_version", 3)
    historic.set_user_attr("study_identity", historic_identity)
    historic.set_user_attr(
        "semantic_payload",
        json.dumps({
            "identity_version": 3,
            "campaign_identity": "historic-campaign",
            "scope": "alpha",
        }, sort_keys=True, separators=(",", ":")),
    )
    trial = historic.ask()
    _, decoded = sample_config(
        scope_config,
        runner.hpo_config,
        trial,
    )
    trial.set_user_attr("decoded_params", decoded)
    historic.tell(trial, 1.0)
    best_payload = {
        "study_identity": historic_identity,
        "trial_number": 0,
        "objective": 1.0,
        "parameters": decoded,
    }
    (historic_root / "best_trial_00000.json").write_text(
        json.dumps(best_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    current_identity, _ = runner._study_identity("alpha")
    current_root = scope_root / "studies" / current_identity
    current_root.mkdir(parents=True)
    current_storage_path = (current_root / "study.sqlite3").resolve()
    current = optuna.create_study(
        study_name=current_identity,
        storage=f"sqlite:///{current_storage_path}",
        direction="minimize",
    )
    current.set_user_attr("identity_version", 4)
    current.set_user_attr("study_identity", current_identity)
    current.set_user_attr(
        "semantic_payload",
        json.dumps({
            "identity_version": 4,
            "campaign_identity": "current-campaign",
            "scope": "alpha",
        }, sort_keys=True, separators=(",", ":")),
    )
    current_trial = current.ask()
    sample_config(scope_config, runner.hpo_config, current_trial)

    audit_report, audited = runner._audit_hpo_scope(
        "alpha",
        scope_config,
    )
    assert audit_report["status"] == "selected"
    assert audit_report["selected"]["study_name"] == historic_identity
    assert audited is not None

    def reject_training(*args, **kwargs):
        del args, kwargs
        raise AssertionError("trainer must not be constructed")

    monkeypatch.setattr(runner, "_run_adaptive_trainer", reject_training)
    selected = runner._run_hpo_scope("alpha", scope_config)

    expected = json.loads(json.dumps(scope_config))
    for dotted_path, value in decoded.items():
        orchestrator_module.set_dotted_value(
            expected,
            dotted_path,
            value,
        )
    assert selected == expected
    assert runner.selection_provenance["alpha"]["study_identity"] == (
        historic_identity
    )
    assert len(current.trials) == 1
    assert current.trials[0].state.name == "RUNNING"


def _study_test_runner(tmp_path: Path) -> CampaignRunner:
    """Return a minimal runner for study storage and lock tests."""
    runner = CampaignRunner.__new__(CampaignRunner)
    runner.paths = CampaignPaths(
        log_root=tmp_path / "log",
        checkpoint_root=tmp_path / "ckpt",
    )
    runner.campaign_hash = "campaign"
    runner.campaign_aliases = frozenset()
    runner.hpo_config = make_hpo_config().model_copy(
        update={"n_trials": 1}
    )
    return runner


def test_semantic_study_lock_rejects_second_writer(
    tmp_path: Path,
) -> None:
    """Only one process handle may mutate one semantic study."""
    runner = _study_test_runner(tmp_path)
    runtime = runner._create_study("alpha")
    try:
        with pytest.raises(RuntimeError, match="already writing"):
            runner._create_study("alpha")
    finally:
        runner._release_study(runtime)


def test_legacy_study_prefers_completed_alias_and_preserves_duplicate(
    tmp_path: Path,
) -> None:
    """A completed legacy alias wins while an accidental study is untouched."""
    import optuna

    runner = _study_test_runner(tmp_path)
    runner.campaign_aliases = frozenset({"accidental"})
    scope_root = _hpo_scope_root(runner.paths.log_root, "alpha")
    scope_root.mkdir(parents=True)
    storage_path = (scope_root / "study.sqlite3").resolve()
    storage = f"sqlite:///{storage_path}"
    scope_config = BrainOmniConfig(
        fs=TEST_FS,
        data={"datasets": {"alpha": "finetune"}},
    ).model_dump(mode="json")

    def objective(trial: Any) -> float:
        """Persist exact configured distributions and decoded parameters."""
        _, decoded = sample_config(
            scope_config,
            runner.hpo_config,
            trial,
        )
        trial.set_user_attr("decoded_params", decoded)
        return 1.0

    selected = optuna.create_study(
        study_name="legacy-alpha",
        storage=storage,
        direction="minimize",
    )
    selected.optimize(objective, n_trials=1)
    duplicate = optuna.create_study(
        study_name="accidental-alpha",
        storage=storage,
        direction="minimize",
    )
    duplicate.ask()

    runtime = runner._create_study("alpha")
    try:
        assert runtime.legacy is True
        assert runtime.study.study_name == "legacy-alpha"
        assert runtime.duplicate_names == ("accidental-alpha",)
    finally:
        runner._release_study(runtime)

    untouched = optuna.load_study(
        study_name="accidental-alpha",
        storage=storage,
    )
    assert untouched.trials[0].state.name == "RUNNING"


def test_legacy_study_rejects_mismatched_trial_distributions(
    tmp_path: Path,
) -> None:
    """A hash-like name cannot override persisted search semantics."""
    import optuna

    runner = _study_test_runner(tmp_path)
    runner.campaign_aliases = frozenset({"legacy"})
    scope_root = _hpo_scope_root(runner.paths.log_root, "alpha")
    scope_root.mkdir(parents=True)
    storage_path = (scope_root / "study.sqlite3").resolve()
    storage = f"sqlite:///{storage_path}"
    wrong = optuna.create_study(
        study_name="legacy-alpha",
        storage=storage,
        direction="minimize",
    )
    wrong.optimize(
        lambda trial: trial.suggest_float(
            "training.max_lr",
            0.1,
            0.2,
        ),
        n_trials=1,
    )

    runtime = runner._create_study("alpha")
    try:
        assert runtime.legacy is False
        assert runtime.duplicate_names == ("legacy-alpha",)
    finally:
        runner._release_study(runtime)


def test_failed_trials_do_not_consume_resumed_hpo_budget() -> None:
    """Only complete and pruned trials count toward a resumed target."""
    import optuna

    state = optuna.trial.TrialState
    trials = [
        SimpleNamespace(state=state.COMPLETE),
        SimpleNamespace(state=state.FAIL),
        SimpleNamespace(state=state.PRUNED),
        SimpleNamespace(state=state.FAIL),
        SimpleNamespace(state=state.FAIL),
    ]

    budgeted, consecutive_failures = _study_progress(trials)

    assert budgeted == 2
    assert consecutive_failures == 2


def test_progressive_collection_uses_completed_optuna_trials() -> None:
    """Progressive assessment excludes pruned Optuna trials."""
    import optuna

    study = optuna.create_study()
    complete = study.ask()
    complete.set_user_attr(
        "objective_history",
        [{"epoch": 0, "value": 1.0}],
    )
    study.tell(complete, 1.0)
    pruned = study.ask()
    study.tell(pruned, state=optuna.trial.TrialState.PRUNED)
    failed = study.ask()
    study.tell(failed, state=optuna.trial.TrialState.FAIL)
    study.ask()

    collected = _collect_completed_trials(study)

    assert [trial["state"] for trial in collected] == ["COMPLETE"]
    assert collected[0]["objective_history"] == [
        {"epoch": 0, "value": 1.0},
    ]


@pytest.mark.parametrize(
    ("callback_result", "expected"),
    [
        (HpoStopReason.PATIENCE, HpoStopReason.PATIENCE),
        (HpoStopReason.OPTUNA_PRUNED, HpoStopReason.OPTUNA_PRUNED),
    ],
)
def test_callback_stop_reason_preserves_hpo_outcome(
    monkeypatch: pytest.MonkeyPatch,
    callback_result: HpoStopReason,
    expected: HpoStopReason,
) -> None:
    """Patience and Optuna pruning remain distinct trainer outcomes."""

    class CallbackTrainer(AbstractTrainer):
        """Minimal concrete trainer exposing the managed callback contract."""

        def setup_model(self) -> None:
            """Satisfy the abstract model-construction contract."""
            return None

        def load_checkpoint(self, *args: Any, **kwargs: Any) -> None:
            """Satisfy the abstract checkpoint-loading contract."""
            del args, kwargs
            return None

    trainer = CallbackTrainer.__new__(CallbackTrainer)
    trainer.epoch = 3
    monkeypatch.setattr(trainer_module, "get_is_master", lambda: True)
    trainer.validation_callback = lambda *args: callback_result

    reason = trainer._callback_stop_reason({}, {})

    assert reason is expected


def test_progressive_pruner_waits_for_initial_evidence() -> None:
    """Progressive HPO delays pruning until its first evidence block."""
    runner = CampaignRunner.__new__(CampaignRunner)
    runner.hpo_config = make_hpo_config().model_copy(
        update={
            "progressive": ProgressiveHpoArgs(
                initial_trials=10,
                increment_trials=10,
            ),
        }
    )

    assert runner._effective_pruner_startup_trials() == 10


def test_pruned_only_study_uses_immutable_recovery_namespace(
    tmp_path: Path,
) -> None:
    """A pruned-only study is retained while a recovery study is created."""
    import optuna

    runner = _study_test_runner(tmp_path)
    initial = runner._create_study("alpha")
    initial_trial = initial.study.ask()
    initial.study.tell(
        initial_trial,
        state=optuna.trial.TrialState.PRUNED,
    )
    initial_identity = initial.study_identity
    initial_path = initial.storage_path
    runner._release_study(initial)

    recovery = runner._create_study("alpha")
    try:
        assert recovery.study_identity != initial_identity
        payload = json.loads(
            recovery.study.user_attrs["semantic_payload"]
        )
        assert payload["recovery_generation"] == 1
        assert initial_path.is_file()
        assert recovery.storage_path.is_file()
    finally:
        runner._release_study(recovery)


def test_nonmultitask_hpo_failures_are_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed HPO scope does not prevent independent selection."""
    runner = CampaignRunner.__new__(CampaignRunner)
    runner.base_dict = {"multitask": False}
    runner.hpo_config = make_hpo_config()
    runner.config_class = BrainOmniConfig
    runner.selection_provenance = {}
    runner._active_study_runtime = None

    monkeypatch.setattr(
        orchestrator_module,
        "validate_search_space",
        lambda *args: None,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_study_scope_configs",
        lambda *args: {"bad": {}, "good": {}},
    )

    def run_scope(scope: str, scope_config: dict[str, Any]) -> dict[str, Any]:
        del scope_config
        if scope == "bad":
            raise RuntimeError("intentional HPO failure")
        runner.selection_provenance[scope] = {"study_identity": "good"}
        return {"selected": scope}

    runner._run_hpo_scope = run_scope

    selected = runner._run_hpo()

    assert selected == {"good": {"selected": "good"}}
    assert runner.hpo_outcomes[0]["status"] == "failed"
    assert runner.hpo_outcomes[1] == {
        "scope": "good",
        "status": "selected",
        "study_identity": "good",
    }


def test_campaign_finishes_selected_scopes_after_hpo_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An HPO scope failure yields partial status after other scopes run."""
    runner = CampaignRunner.__new__(CampaignRunner)
    runner.hpo_config = make_hpo_config()
    runner.base_dict = {
        "model_type": "brainomni",
        "seeds": [42],
    }
    runner.campaign_hash = "campaign"
    runner.campaign_aliases = frozenset()
    runner.distributed_initialized = False
    runner.paths = CampaignPaths(
        log_root=tmp_path / "log",
        checkpoint_root=tmp_path / "ckpt",
    )
    runner.hpo_outcomes = []
    calls: list[str] = []
    statuses: list[dict[str, Any]] = []
    runner._configure_campaign_logging = lambda: None
    runner._save_campaign_config = lambda: None
    runner._initialize_distributed = lambda seed: calls.append("init")

    def run_hpo() -> dict[str, dict[str, str]]:
        calls.append("hpo")
        runner.hpo_outcomes = [
            {"scope": "bad", "status": "failed", "fingerprint": "x"},
            {"scope": "good", "status": "selected"},
        ]
        return {"good": {"selected": "good"}}

    def run_seeds(
        selected: dict[str, dict[str, str]],
    ) -> tuple[dict[str, Any], bool]:
        calls.append("seeds")
        assert selected == {"good": {"selected": "good"}}
        return {
            "failed": [],
            "succeeded": [42],
            "complete": True,
        }, True

    def update_status(state: str, **kwargs: Any) -> None:
        statuses.append({"state": state, **kwargs})

    runner._run_hpo = run_hpo
    runner._run_seeds = run_seeds
    runner._update_invocation_status = update_status
    monkeypatch.setattr(orchestrator_module, "get_is_master", lambda: False)

    with pytest.raises(CampaignExecutionError):
        runner.run()

    assert calls == ["init", "hpo", "seeds"]
    assert statuses[-1]["state"] == "partial"
    assert statuses[-1]["invocation"]["hpo_failed"][0]["scope"] == "bad"


def test_campaign_runs_hpo_once_before_all_evaluation_seeds() -> None:
    """Evaluation seeds consume one selected result without repeating HPO."""
    runner = CampaignRunner.__new__(CampaignRunner)
    runner.hpo_config = make_hpo_config()
    runner.base_dict = {
        "model_type": "brainomni",
        "seeds": [42, 43, 44],
    }
    runner.campaign_hash = "campaign"
    runner.distributed_initialized = False
    calls = []
    runner._save_campaign_config = lambda: None
    runner._initialize_distributed = lambda seed: calls.append(("init", seed))
    runner._run_hpo = lambda: calls.append(("hpo", 0)) or {"winner": {}}

    def run_seeds(selected):
        calls.append(("seeds", list(runner.base_dict["seeds"]), selected))
        return ({"failed": []}, False)

    runner._run_seeds = run_seeds
    runner.run()

    assert calls == [
        ("init", 0),
        ("hpo", 0),
        ("seeds", [42, 43, 44], {"winner": {}}),
    ]


def test_deterministic_model_ignores_hpo_with_one_seed() -> None:
    """Deterministic baselines warn and skip HPO for their sole seed."""
    runner = CampaignRunner.__new__(CampaignRunner)
    runner.hpo_config = make_hpo_config()
    runner.base_dict = {
        "model_type": "minirocket",
        "seeds": [42],
    }
    runner.campaign_hash = "campaign"
    runner.distributed_initialized = False
    calls = []
    runner._save_campaign_config = lambda: None
    runner._initialize_distributed = lambda seed: calls.append(("init", seed))

    def reject_hpo():
        pytest.fail("Deterministic model must not run HPO.")

    def fixed_scopes():
        calls.append(("fixed",))
        return {"fixed": {}}

    def run_seeds(selected):
        calls.append(("seeds", list(runner.base_dict["seeds"]), selected))
        return ({"failed": []}, False)

    runner._run_hpo = reject_hpo
    runner._fixed_scopes = fixed_scopes
    runner._run_seeds = run_seeds
    with pytest.warns(UserWarning, match="ignoring"):
        runner.run()

    assert calls == [
        ("init", 42),
        ("fixed",),
        ("seeds", [42], {"fixed": {}}),
    ]

def test_nonmultitask_failures_are_isolated_by_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop only the failing dataset after two matching seed errors."""
    runner = CampaignRunner.__new__(CampaignRunner)
    runner.base_dict = {
        "multitask": False,
        "seeds": [42, 43, 44],
    }
    runner.paths = CampaignPaths(
        log_root=tmp_path / "log",
        checkpoint_root=tmp_path / "ckpt",
    )
    runner.final_dataset_attempts = []
    calls: list[tuple[str, int]] = []

    def run_seed(seed: int, selected: dict[str, dict]) -> bool:
        """Fail alpha repeatedly while beta completes every seed."""
        dataset_name = next(iter(selected))
        calls.append((dataset_name, seed))
        if dataset_name == "alpha":
            raise RuntimeError("dataset setup failed")
        return True

    runner._run_seed = run_seed
    messages: list[str] = []
    monkeypatch.setattr(
        orchestrator_module.logger,
        "info",
        lambda message, *args: messages.append(message % args),
    )
    monkeypatch.setenv("EEGFM_ERROR_LOG_PATH", "/tmp/baseline.err")

    invocation, eligible = runner._run_seeds({
        "alpha": {},
        "beta": {},
    })

    assert calls == [
        ("alpha", 42),
        ("alpha", 43),
        ("beta", 42),
        ("beta", 43),
        ("beta", 44),
    ]
    assert eligible is False
    assert invocation["dataset_outcomes"] == [
        {
            "dataset": "alpha",
            "status": "failed",
            "attempted": [42, 43],
            "succeeded": [],
            "skipped": [],
            "failed": invocation["failed"],
            "unattempted": [44],
        },
        {
            "dataset": "beta",
            "status": "succeeded",
            "attempted": [42, 43, 44],
            "succeeded": [42, 43, 44],
            "skipped": [],
            "failed": [],
            "unattempted": [],
        },
    ]
    assert (
        "Dataset execution summary: succeeded=1, skipped=0, failed=1."
        in messages
    )
    assert any(
        message.startswith("Failed dataset alpha:")
        for message in messages
    )
    assert any("/tmp/baseline.err" in message for message in messages)


def _progressive_trial(number: int) -> dict[str, Any]:
    """Return one trial with finite objective and epoch history."""
    return {
        "number": number,
        "objective": float(number),
        "objective_history": [
            {"epoch": epoch, "value": float(number) + epoch / 100.0}
            for epoch in range(5)
        ],
    }


def test_progressive_resume_waits_for_complete_block() -> None:
    """A partial study is not assessed until its next block boundary."""
    args = ProgressiveHpoArgs(
        initial_trials=10,
        increment_trials=10,
    )
    trials = [_progressive_trial(number) for number in range(30)]

    assert progressive_assessment_block(trials[:9], 9, args) is None
    first = progressive_assessment_block(trials[:10], 10, args)
    assert first is not None
    assert [trial["number"] for trial in first] == list(range(10))

    assert progressive_assessment_block(trials[:19], 19, args) is None
    second = progressive_assessment_block(trials[:20], 20, args)
    assert second is not None
    assert [trial["number"] for trial in second] == list(range(10, 20))

    assert progressive_assessment_block(trials[:26], 26, args) is None
    third = progressive_assessment_block(trials, 30, args)
    assert third is not None
    assert [trial["number"] for trial in third] == list(range(20, 30))


def test_progressive_resume_detects_missing_historical_evidence() -> None:
    """Legacy trials without epoch objectives cannot trigger early stop."""
    args = ProgressiveHpoArgs()
    complete = [_progressive_trial(number) for number in range(9)]
    historical = [
        {
            "number": number,
            "objective": float(number),
        }
        for number in range(9)
    ]

    assert has_complete_progressive_evidence(complete, args)
    assert not has_complete_progressive_evidence(historical, args)


def test_progressive_block_rejects_inconsistent_completed_count() -> None:
    """A mismatched completed-trial count fails instead of truncating."""
    with pytest.raises(ValueError, match="does not match collected trials"):
        progressive_assessment_block(
            [_progressive_trial(0)],
            2,
            ProgressiveHpoArgs(),
        )


def test_progressive_hpo_flat_outcome_stops_early() -> None:
    """Flat studies stop when between-trial variance is below threshold."""
    # Create trials with very similar objectives (flat)
    trials = [
        {
            "objective": 1.0,
            "objective_history": [
                {"epoch": 0, "value": 1.2},
                {"epoch": 1, "value": 1.1},
                {"epoch": 2, "value": 1.05},
                {"epoch": 3, "value": 1.02},
                {"epoch": 4, "value": 1.0},
            ],
        },
        {
            "objective": 1.001,
            "objective_history": [
                {"epoch": 0, "value": 1.15},
                {"epoch": 1, "value": 1.08},
                {"epoch": 2, "value": 1.03},
                {"epoch": 3, "value": 1.01},
                {"epoch": 4, "value": 1.001},
            ],
        },
    ]

    assessment = assess_progressive_study(
        trials,
        "minimize",
        ProgressiveHpoArgs(),
    )

    assert assessment.outcome == "flat"
    assert assessment.should_expand is False
    assert assessment.completed_trials == 2


def test_progressive_hpo_clear_winner_with_gap() -> None:
    """A clear gap continues when the incumbent remains unstable."""
    # Create trials with clear winner but not necessarily stable
    trials = [
        {
            "objective": 0.5,  # Clear winner
            "objective_history": [
                {"epoch": 0, "value": 0.8},
                {"epoch": 1, "value": 0.6},
                {"epoch": 2, "value": 0.501},
                {"epoch": 3, "value": 0.5005},
                {"epoch": 4, "value": 0.5},
            ],
        },
        {
            "objective": 1.5,  # Runner-up
            "objective_history": [
                {"epoch": 0, "value": 2.0},
                {"epoch": 1, "value": 1.8},
                {"epoch": 2, "value": 1.6},
                {"epoch": 3, "value": 1.55},
                {"epoch": 4, "value": 1.5},
            ],
        },
        {
            "objective": 2.0,
            "objective_history": [
                {"epoch": 0, "value": 2.5},
                {"epoch": 1, "value": 2.3},
                {"epoch": 2, "value": 2.1},
                {"epoch": 3, "value": 2.05},
                {"epoch": 4, "value": 2.0},
            ],
        },
    ]

    assessment = assess_progressive_study(
        trials,
        "minimize",
        ProgressiveHpoArgs(),
    )

    # Should have good between-trial variance and clear winner gap
    assert assessment.winner_gap is not None
    assert assessment.winner_gap > 0.5
    assert assessment.between_trial_sd is not None
    assert assessment.between_trial_sd > 0.01


def test_progressive_hpo_unresolved_outcome_continues() -> None:
    """Unresolved variation without a clear winner continues."""
    # Create trials with high variance but no clear stable winner
    trials = [
        {
            "objective": 0.5,
            "objective_history": [
                {"epoch": 0, "value": 1.0},
                {"epoch": 1, "value": 0.8},
                {"epoch": 2, "value": 0.6},
                {"epoch": 3, "value": 0.55},
                {"epoch": 4, "value": 0.5},
            ],
        },
        {
            "objective": 0.7,
            "objective_history": [
                {"epoch": 0, "value": 1.2},
                {"epoch": 1, "value": 0.95},
                {"epoch": 2, "value": 0.75},
                {"epoch": 3, "value": 0.72},
                {"epoch": 4, "value": 0.7},
            ],
        },
        {
            "objective": 1.5,
            "objective_history": [
                {"epoch": 0, "value": 2.0},
                {"epoch": 1, "value": 1.8},
                {"epoch": 2, "value": 1.6},
                {"epoch": 3, "value": 1.55},
                {"epoch": 4, "value": 1.5},
            ],
        },
    ]

    assessment = assess_progressive_study(
        trials,
        "minimize",
        ProgressiveHpoArgs(minimum_resolution=0.01),
    )

    assert assessment.outcome == "responsive_unresolved"
    assert assessment.should_expand is True


def test_auto_filter_encoder_lr_scale_for_frozen_encoder() -> None:
    """Frozen encoder searches automatically filter encoder_lr_scale."""
    from baseline.hpo.search import validate_search_space

    base = BrainOmniConfig(fs=TEST_FS, seeds=[0]).model_dump(mode="json")
    base["training"]["freeze_encoder"] = True

    hpo = HpoConfig.model_validate({
        "enabled": True,
        "n_trials": 5,
        "search_space": {
            "training.max_lr": {
                "distribution": "float",
                "low": 1e-4,
                "high": 1e-3,
            },
            "training.encoder_lr_scale": {
                "distribution": "float",
                "low": 0.1,
                "high": 1.0,
            },
        },
    })

    # Should auto-filter encoder_lr_scale
    validate_search_space(base, hpo, BrainOmniConfig)

    assert "training.encoder_lr_scale" not in hpo.search_space
    assert "training.max_lr" in hpo.search_space


def test_unfrozen_encoder_keeps_encoder_lr_scale() -> None:
    """Unfrozen encoder searches keep encoder_lr_scale in search space."""
    from baseline.hpo.search import validate_search_space

    base = BrainOmniConfig(fs=TEST_FS, seeds=[0]).model_dump(mode="json")
    base["training"]["freeze_encoder"] = False

    hpo = HpoConfig.model_validate({
        "enabled": True,
        "n_trials": 5,
        "search_space": {
            "training.max_lr": {
                "distribution": "float",
                "low": 1e-4,
                "high": 1e-3,
            },
            "training.encoder_lr_scale": {
                "distribution": "float",
                "low": 0.1,
                "high": 1.0,
            },
        },
    })

    # Should keep encoder_lr_scale
    validate_search_space(base, hpo, BrainOmniConfig)

    assert "training.encoder_lr_scale" in hpo.search_space
    assert "training.max_lr" in hpo.search_space
