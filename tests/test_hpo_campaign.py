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

import baseline.hpo.orchestrator as orchestrator_module
from baseline.brainomni.brainomni_config import BrainOmniConfig
from baseline.hpo.artifacts import (
    CampaignPaths,
    collect_test_rows,
    failure_fingerprint,
    get_campaign_hash,
    summarize_test_rows,
)
from baseline.hpo.config import HpoConfig
from baseline.hpo.orchestrator import (
    CampaignRunner,
    _hpo_scope_root,
    _study_progress,
    _study_scope_configs,
)
from baseline.hpo.search import (
    reduce_objective,
    sample_config,
    validate_search_space,
)
from baseline.registry import register_builtin_models
from baseline_main import _load_configs, _normalize_legacy_seed
from common.utils import setup_yaml


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
    assert BrainOmniConfig().seeds == [42]
    assert BrainOmniConfig().seed == 42
    with pytest.raises(RuntimeError, match="exactly one effective seed"):
        _ = BrainOmniConfig(seeds=[42, 43]).seed


@pytest.mark.parametrize("seeds", ([True], ["42"], 42))
def test_public_seeds_reject_coerced_types(seeds) -> None:
    """Boolean, string, and scalar seed inputs are not silently coerced."""
    with pytest.raises(ValidationError, match="seeds must"):
        BrainOmniConfig(seeds=seeds)


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
    base = BrainOmniConfig(seeds=[42]).model_dump(mode="json")
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
    base = BrainOmniConfig(seeds=[0]).model_dump(mode="json")
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
    base = BrainOmniConfig(seeds=[0]).model_dump(mode="json")
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
    runner.base_dict = {"seeds": [42, 43, 44, 45]}
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
    scope_config = BrainOmniConfig().model_dump(mode="json")
    runner.base_dict = scope_config
    runner.config_class = BrainOmniConfig

    runner.invocation_root = tmp_path / "log" / "invocations" / "test"
    runner.invocation_root.mkdir(parents=True)
    runner._active_study_runtime = None
    runtime = runner._create_study("alpha")
    trial = runtime.study.ask()
    trial.set_user_attr("decoded_params", {})
    runtime.study.tell(trial, 1.0)
    runner._release_study(runtime)

    def reject_training(*args, **kwargs):
        del args, kwargs
        raise AssertionError("trainer must not be constructed")

    monkeypatch.setattr(runner, "_run_adaptive_trainer", reject_training)

    selected = runner._run_hpo_scope("alpha", scope_config)

    assert selected == scope_config
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


def test_deterministic_model_ignores_hpo_but_keeps_all_seeds() -> None:
    """Deterministic baselines warn, skip HPO, and retain seed repetition."""
    runner = CampaignRunner.__new__(CampaignRunner)
    runner.hpo_config = make_hpo_config()
    runner.base_dict = {
        "model_type": "minirocket",
        "seeds": [42, 43, 44],
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
        ("seeds", [42, 43, 44], {"fixed": {}}),
    ]
