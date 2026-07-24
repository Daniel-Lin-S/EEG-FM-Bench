"""Run HPO studies and independent multi-seed baseline campaigns.

Inputs are one validated model configuration and an optional HpoConfig.
Outputs are persistent Optuna studies, current-style seed artifact trees,
completion metadata, and campaign test summaries.
"""

from __future__ import annotations

import copy
import gc
import json
import logging
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Type

import torch
import yaml
from pydantic import BaseModel

from baseline.adaptive_batching import (
    configure_cuda_allocator,
    derive_batch_candidates,
    is_cuda_oom,
    select_safe_micro_batch,
)
from baseline.abstract.factory import ModelRegistry
from baseline.hpo.artifacts import (
    CampaignPaths,
    build_campaign_paths,
    check_completion_compatibility,
    failure_fingerprint,
    get_campaign_hash,
    write_campaign_summary,
)
from baseline.hpo.config import HpoConfig
from baseline.hpo.search import (
    objective_values,
    reduce_objective,
    sample_config,
    set_dotted_value,
    validate_search_space,
)
from baseline.utils.run_artifacts import get_config_hash
from common.distributed.env import (
    clean_torch_distributed,
    get_is_master,
    get_world_size,
)
from common.log import setup_log


logger = logging.getLogger("baseline")
DETERMINISTIC_MODELS = frozenset({"catch22", "minirocket"})
PRUNED_STATUS = "pruned"
COMPLETE_STATUS = "complete"
FAILED_STATUS = "failed"
OOM_MESSAGE_PARTS = (
    "out of memory",
    "cuda error: memory allocation",
)


class CampaignExecutionError(RuntimeError):
    """Raised after one or more seed executions fail."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write one JSON mapping."""
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if existing == dict(payload):
            return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write one YAML mapping."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(dict(payload), file_obj, sort_keys=False)
    temporary.replace(path)


def _broadcast_object(value: Any) -> Any:
    """Broadcast a picklable master value to all distributed ranks."""
    if not (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        return value
    objects = [value if get_is_master() else None]
    torch.distributed.broadcast_object_list(objects, src=0)
    return objects[0]


def _scoped_config(
    config: Mapping[str, Any],
    seed: int,
    datasets_config: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Return one independent seed- and dataset-scoped config."""
    scoped = copy.deepcopy(dict(config))
    scoped["seeds"] = [seed]
    if datasets_config is not None:
        scoped["data"]["datasets"] = dict(datasets_config)
    return scoped


def _force_local_trial_logging(config: Dict[str, Any]) -> None:
    """Disable cloud trial logging and ensure a CSV validation trace."""
    logging_config = config["logging"]
    logging_config["use_cloud"] = False
    outputs = list(logging_config.get("outputs", []))
    if logging_config.get("level", "info") != "debug":
        outputs = [output for output in outputs if output != "log"]
    if "csv" not in outputs:
        outputs.append("csv")
    logging_config["outputs"] = outputs


def _study_scope_configs(
    config: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Return dataset-specific or joint HPO base configs."""
    if config["multitask"]:
        return {"multitask": copy.deepcopy(dict(config))}

    scopes: Dict[str, Dict[str, Any]] = {}
    for dataset_name, dataset_config in config["data"]["datasets"].items():
        scopes[dataset_name] = _scoped_config(
            config,
            seed=0,
            datasets_config={dataset_name: dataset_config},
        )
        scopes[dataset_name]["seeds"] = list(config["seeds"])
    return scopes


def _config_hash(config: Mapping[str, Any]) -> str:
    """Return one scoped trainer configuration hash."""
    return get_config_hash(
        copy.deepcopy(dict(config)),
        bool(config["multitask"]),
    )


def _expected_configs(
    selected_configs: Mapping[str, Mapping[str, Any]],
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    """Return selected dataset configurations for one seed."""
    expected: Dict[str, Dict[str, Any]] = {}
    for config in selected_configs.values():
        scoped = _scoped_config(config, seed)
        config_hash = _config_hash(scoped)
        for dataset_name in scoped["data"]["datasets"]:
            previous = expected.get(dataset_name)
            if previous is not None and _config_hash(previous) != config_hash:
                raise ValueError(
                    f"Dataset '{dataset_name}' has conflicting selected "
                    "configurations."
                )
            expected[dataset_name] = copy.deepcopy(scoped)
    if not expected:
        raise ValueError("Selected campaign configuration has no datasets.")
    return expected


def _seed_scope_is_complete(
    paths: CampaignPaths,
    campaign_hash: str,
    seed: int,
    selected_configs: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Return whether every selected dataset scope has matching completion."""
    expected = _expected_configs(selected_configs, seed)
    for dataset_name, config in expected.items():
        completion_path = (
            paths.seed_log_root(seed)
            / "datasets"
            / dataset_name
            / "completion.json"
        )
        result = check_completion_compatibility(
            completion_path,
            campaign_hash,
            seed,
            config,
        )
        if not result.compatible:
            logger.debug(
                "Seed %d dataset %s is incomplete: %s",
                seed,
                dataset_name,
                result.reason,
            )
            return False
    return True


def _is_recoverable_trial_failure(exc: Exception) -> bool:
    """Return whether one HPO trial failure may safely be continued."""
    if isinstance(exc, (ValueError, torch.cuda.OutOfMemoryError)):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return any(part in message for part in OOM_MESSAGE_PARTS)


def _release_training_state() -> None:
    """Release cyclic trainer references and cached CUDA allocations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _hpo_scope_root(root: Path, scope: str) -> Path:
    """Return the requested log or checkpoint root for one HPO scope."""
    if scope == "multitask":
        return root / "hpo" / "multitask"
    return root / "hpo" / "datasets" / scope


def _study_progress(trials: list[Any]) -> tuple[int, int]:
    """Count budgeted trials and the trailing terminal-failure streak.

    Parameters
    ----------
    trials : list[Any]
        Persisted Optuna trials in chronological order. Each element must
        expose a state whose ``name`` is an Optuna trial-state name.

    Returns
    -------
    tuple[int, int]
        Count of complete-or-pruned trials and consecutive failed trials at
        the end of the persisted history.

    Raises
    ------
    ValueError
        If a trial does not expose a recognized state name.
    """
    budgeted_names = frozenset({"COMPLETE", "PRUNED"})
    terminal_names = budgeted_names | {"FAIL"}
    state_names = [trial.state.name for trial in trials]
    unknown = set(state_names) - terminal_names - {"RUNNING", "WAITING"}
    if unknown:
        raise ValueError(
            "Expected recognized Optuna trial states, but got "
            f"{sorted(unknown)}."
        )
    budgeted = sum(name in budgeted_names for name in state_names)
    consecutive_failures = 0
    for state_name in reversed(state_names):
        if state_name != "FAIL":
            break
        consecutive_failures += 1
    return budgeted, consecutive_failures


def _synchronize_micro_batch(micro_batch: int) -> int:
    """Select the smallest safe micro-batch across distributed ranks.

    Parameters
    ----------
    micro_batch : int
        Largest candidate predicted safe on the current rank.

    Returns
    -------
    int
        Minimum candidate reported by any initialized distributed rank.
    """
    if micro_batch <= 0:
        raise ValueError(
            f"Expected a positive micro-batch, but got {micro_batch}."
        )
    if not (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        return micro_batch
    device = torch.device("cuda", torch.cuda.current_device())
    value = torch.tensor(micro_batch, device=device, dtype=torch.int64)
    torch.distributed.all_reduce(
        value,
        op=torch.distributed.ReduceOp.MIN,
    )
    return int(value.item())


class CampaignRunner:
    """Coordinate HPO, seed runs, failure policy, and summaries."""

    def __init__(
        self,
        config: BaseModel,
        hpo_config: HpoConfig,
        config_class: Type[BaseModel],
    ):
        self.config = config
        self.config_class = config_class
        self.base_dict = config.model_dump(mode="json")
        self.hpo_config = hpo_config
        self.campaign_hash = get_campaign_hash(
            self.base_dict,
            hpo_config.model_dump(mode="json"),
        )
        self.paths = build_campaign_paths(
            run_dir=self.base_dict["logging"]["run_dir"],
            model_type=self.base_dict["model_type"],
            experiment_name=(
                self.base_dict["logging"]["experiment_name"]
            ),
            campaign_hash=self.campaign_hash,
        )
        self.distributed_initialized = False
        self.adaptive_batch_cache: Dict[str, Dict[str, Any]] = {}
        self.last_adaptive_memory_profile: Dict[str, Any] = {}

    def _configure_campaign_logging(self) -> None:
        """Configure campaign console logging and Optuna verbosity."""
        logging_config = self.base_dict.get("logging", {})
        level = str(logging_config.get("level", "info")).upper()
        setup_log(
            start_time=None,
            name="baseline",
            level=level,
        )
        try:
            import optuna
        except ImportError:
            return
        optuna_level = (
            optuna.logging.DEBUG
            if level == "DEBUG"
            else optuna.logging.WARNING
        )
        optuna.logging.set_verbosity(optuna_level)

    def _adaptive_signature(self, config: BaseModel) -> str:
        """Return an invocation-local memory-profile identity."""
        payload = config.model_dump(mode="json")
        training = payload["training"]
        signature = {
            "model_type": payload["model_type"],
            "model": payload["model"],
            "data": payload["data"],
            "fs": payload["fs"],
            "use_amp": training["use_amp"],
            "freeze_encoder": training["freeze_encoder"],
            "lora": training["lora"],
            "world_size": get_world_size(),
        }
        if torch.cuda.is_available():
            signature["gpu"] = torch.cuda.get_device_name(
                torch.cuda.current_device()
            )
        return json.dumps(signature, sort_keys=True)

    def _measure_adaptive_memory(
        self,
        config: BaseModel,
        global_batch_size: int,
        world_size: int,
    ) -> tuple[str, Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """Return a current or newly calibrated CUDA memory model."""
        batching = config.training.adaptive_batching
        signature = self._adaptive_signature(config)
        cache = getattr(self, "adaptive_batch_cache", {})
        cached = cache.get(signature)
        if cached is not None:
            return signature, cache, dict(cached)

        calibration_retried = False
        while True:
            calibration_config = config.model_copy(deep=True)
            calibration = ModelRegistry.create_trainer(
                calibration_config
            )
            calibration.configure_runtime_batching(
                global_batch_size,
                1,
                world_size,
            )
            try:
                memory_model = calibration.calibrate_cuda_memory()
            except BaseException as exc:
                self.last_adaptive_memory_profile = dict(
                    calibration.adaptive_batch_profile
                )
                if not is_cuda_oom(exc):
                    raise
                if calibration_retried:
                    raise torch.cuda.OutOfMemoryError(
                        "The model's fixed CUDA state cannot fit while "
                        "preserving the configured memory reserve after "
                        "two micro-batch-one calibration attempts."
                    ) from exc
                calibration_retried = True
                wait_seconds = batching.contention_wait_seconds
                logger.warning(
                    "Micro-batch-one calibration cannot fit; releasing "
                    "CUDA state and retrying in %d seconds.",
                    wait_seconds,
                )
                calibration = None
                _release_training_state()
                if wait_seconds:
                    time.sleep(wait_seconds)
                continue
            finally:
                calibration = None
                _release_training_state()
            cache[signature] = memory_model
            self.adaptive_batch_cache = cache
            return signature, cache, memory_model

    def _predicted_candidates(
        self,
        candidates: list[int],
        memory_model: Mapping[str, Any],
    ) -> tuple[list[int], int]:
        """Return candidates starting at the DDP-wide predicted maximum."""
        eligible = candidates
        oom_cap = memory_model.get("oom_cap")
        if oom_cap is not None:
            eligible = [
                candidate
                for candidate in candidates
                if candidate <= int(oom_cap)
            ]
        selected, _ = select_safe_micro_batch(
            eligible,
            int(memory_model["estimated_fixed_bytes"]),
            int(memory_model["calibration_peak_reserved_bytes"]),
            int(memory_model["calibration_batch_size"]),
            int(memory_model["process_limit_bytes"]),
        )
        selected = _synchronize_micro_batch(selected)
        _, predicted_peak = select_safe_micro_batch(
            [selected],
            int(memory_model["estimated_fixed_bytes"]),
            int(memory_model["calibration_peak_reserved_bytes"]),
            int(memory_model["calibration_batch_size"]),
            int(memory_model["process_limit_bytes"]),
        )
        return candidates[candidates.index(selected):], predicted_peak

    def _run_adaptive_trainer(
        self,
        config: BaseModel,
        prepare: Callable[[Any], None],
    ) -> Dict[str, Any]:
        """Run one neural config with internal exact-divisor OOM backoff."""
        batching = config.training.adaptive_batching
        world_size = get_world_size()
        global_batch_size = config.data.batch_size
        candidates = derive_batch_candidates(
            global_batch_size,
            world_size,
            batching.enabled,
        )
        self.last_adaptive_memory_profile = {}
        signature = self._adaptive_signature(config)
        cache = getattr(self, "adaptive_batch_cache", {})
        memory_profile: Dict[str, Any] = {}
        if batching.enabled and torch.cuda.is_available():
            signature, cache, memory_profile = (
                self._measure_adaptive_memory(
                    config,
                    global_batch_size,
                    world_size,
                )
            )
            device = torch.device(
                "cuda",
                torch.cuda.current_device(),
            )
            current_limit = configure_cuda_allocator(
                device,
                batching.memory_reserve_fraction,
            )
            if current_limit is None:
                raise RuntimeError(
                    "CUDA allocator configuration returned no limit."
                )
            memory_profile = dict(memory_profile)
            memory_profile.update(current_limit.as_dict())
            candidates, predicted_peak = self._predicted_candidates(
                candidates,
                memory_profile,
            )
            memory_profile["estimated_selected_peak_bytes"] = (
                predicted_peak
            )
            self.last_adaptive_memory_profile = dict(
                memory_profile
            )
        batch_one_retried = False
        last_error: Optional[BaseException] = None
        for candidate_index, micro_batch in enumerate(candidates):
            while True:
                attempt_config = config.model_copy(deep=True)
                trainer = ModelRegistry.create_trainer(attempt_config)
                trainer.configure_runtime_batching(
                    global_batch_size,
                    micro_batch,
                    world_size,
                )
                trainer.adaptive_batch_profile["adaptive_retry"] = (
                    candidate_index
                )
                trainer.adaptive_batch_profile.update(memory_profile)
                try:
                    prepare(trainer)
                    result = trainer.run()
                except BaseException as exc:
                    last_error = exc
                    self.last_adaptive_memory_profile = dict(
                        trainer.adaptive_batch_profile
                    )
                    if not is_cuda_oom(exc):
                        raise
                    logger.warning(
                        "CUDA OOM at micro-batch %d; recalibrating within "
                        "the same run attempt.",
                        micro_batch,
                    )
                    if micro_batch != 1:
                        next_micro_batch = candidates[
                            candidate_index + 1
                        ]
                        if memory_profile:
                            memory_profile["oom_cap"] = next_micro_batch
                            cache[signature] = dict(memory_profile)
                            self.adaptive_batch_cache = cache
                        break
                    if batch_one_retried:
                        raise torch.cuda.OutOfMemoryError(
                            "The model's fixed CUDA state cannot fit while "
                            "preserving the configured memory reserve after "
                            "two micro-batch-one runtime attempts."
                        ) from exc
                    batch_one_retried = True
                    wait_seconds = batching.contention_wait_seconds
                    if wait_seconds:
                        logger.warning(
                            "Micro-batch one cannot fit; waiting %d seconds "
                            "before one clean retry.",
                            wait_seconds,
                        )
                        trainer = None
                        _release_training_state()
                        time.sleep(wait_seconds)
                    continue
                finally:
                    trainer = None
                    _release_training_state()
                return result
        if last_error is None:
            raise RuntimeError("Adaptive batching produced no candidates.")
        raise last_error

    def _initialize_distributed(self, seed: int) -> None:
        """Initialize one process group reused by all neural executions."""
        if self.base_dict["model_type"] in DETERMINISTIC_MODELS:
            return
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            self.distributed_initialized = True
            return
        config_dict = _scoped_config(self.base_dict, seed)
        bootstrap_config = self.config_class.model_validate(config_dict)
        trainer = ModelRegistry.create_trainer(bootstrap_config)
        trainer.setup_distributed()
        self.distributed_initialized = True
        del trainer

    def _save_campaign_config(self) -> None:
        """Persist the source campaign configuration."""
        if not get_is_master():
            return
        self.paths.log_root.mkdir(parents=True, exist_ok=True)
        self.paths.checkpoint_root.mkdir(parents=True, exist_ok=True)
        payload = copy.deepcopy(self.base_dict)
        payload["hpo"] = self.hpo_config.model_dump(mode="json")
        path = self.paths.log_root / "campaign.yaml"
        _atomic_yaml(path, payload)
        logger.info(
            "Campaign log root: %s",
            self.paths.log_root.resolve(),
        )
        logger.info(
            "Campaign checkpoint root: %s",
            self.paths.checkpoint_root.resolve(),
        )
        logger.info(
            "Campaign layout: campaign.yaml, hpo/, logs/seed_<seed>/, "
            "and summary/ under %s.",
            self.paths.log_root.resolve(),
        )

    def _create_study(self, scope: str):
        """Create or resume one rank-zero Optuna study."""
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "HPO requires Optuna. Install project requirements or run "
                "'pip install optuna'."
            ) from exc

        scope_root = _hpo_scope_root(self.paths.log_root, scope)
        scope_root.mkdir(parents=True, exist_ok=True)
        storage_path = (scope_root / "study.sqlite3").resolve()
        storage = f"sqlite:///{storage_path}"
        sampler = optuna.samplers.TPESampler(
            seed=self.hpo_config.sampler.seed,
            n_startup_trials=(
                self.hpo_config.sampler.n_startup_trials
            ),
        )
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=(
                self.hpo_config.pruner.n_startup_trials
            ),
            n_warmup_steps=self.hpo_config.pruner.n_warmup_epochs,
            interval_steps=self.hpo_config.pruner.interval_epochs,
        )
        study_name = f"{self.campaign_hash}-{scope}"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            direction=self.hpo_config.objective.direction,
            load_if_exists=True,
        )
        for trial in study.get_trials(deepcopy=False):
            if trial.state == optuna.trial.TrialState.RUNNING:
                study.tell(
                    trial.number,
                    state=optuna.trial.TrialState.FAIL,
                )
        return study

    def _write_trial_status(
        self,
        trial_root: Path,
        status: str,
        decoded_params: Mapping[str, Any],
        objective: Optional[float] = None,
        error: Optional[str] = None,
        objective_history: Optional[list[Mapping[str, Any]]] = None,
        memory_information: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Persist one trial's decoded parameters and terminal state."""
        if not get_is_master():
            return
        payload: Dict[str, Any] = {
            "status": status,
            "parameters": dict(decoded_params),
        }
        if objective is not None:
            payload["objective"] = objective
        if error is not None:
            payload["error"] = error
        if objective_history is not None:
            payload["objective_history"] = list(objective_history)
        if memory_information is not None:
            payload["memory_information"] = dict(memory_information)
        _atomic_json(trial_root / "trial.json", payload)

    def _run_hpo_scope(
        self,
        scope: str,
        scope_config: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Run or resume one HPO scope and return its selected config."""
        import optuna
        self._configure_campaign_logging()

        study = self._create_study(scope) if get_is_master() else None
        if get_is_master():
            persisted_trials = study.get_trials(deepcopy=False)
            budgeted, consecutive_failures = _study_progress(
                persisted_trials
            )
            study_status = "resumed" if persisted_trials else "new"
        else:
            budgeted = 0
            consecutive_failures = 0
            study_status = "new"
        progress = _broadcast_object({
            "budgeted": budgeted,
            "consecutive_failures": consecutive_failures,
            "study_status": study_status,
        })
        budgeted = int(progress["budgeted"])
        consecutive_failures = int(progress["consecutive_failures"])
        study_status = str(progress["study_status"])
        if get_is_master():
            study_path = (
                _hpo_scope_root(self.paths.log_root, scope)
                / "study.sqlite3"
            ).resolve()
            logger.info(
                "HPO scope %s: %s study, budget %d/%d at %s.",
                scope,
                study_status,
                budgeted,
                self.hpo_config.n_trials,
                study_path,
            )
        failure_limit = self.hpo_config.max_consecutive_failed_trials
        if consecutive_failures >= failure_limit:
            raise RuntimeError(
                f"HPO scope '{scope}' already has {consecutive_failures} "
                "consecutive failed trials. Correct the configuration or "
                "search space before resuming."
            )

        while budgeted < self.hpo_config.n_trials:
            if get_is_master():
                trial = study.ask()
                sampled, decoded = sample_config(
                    scope_config,
                    self.hpo_config,
                    trial,
                )
                payload = {
                    "number": trial.number,
                    "config": sampled,
                    "decoded": decoded,
                }
            else:
                trial = None
                payload = None
            payload = _broadcast_object(payload)
            trial_number = int(payload["number"])
            sampled = dict(payload["config"])
            decoded = dict(payload["decoded"])
            sampled["seeds"] = [self.hpo_config.seed]
            _force_local_trial_logging(sampled)

            trial_root = (
                _hpo_scope_root(self.paths.log_root, scope)
                / "trials"
                / f"trial_{trial_number:05d}"
            )
            if get_is_master():
                _atomic_yaml(trial_root / "resolved_config.yaml", sampled)
            checkpoint_root = (
                _hpo_scope_root(self.paths.checkpoint_root, scope)
                / f"trial_{trial_number:05d}"
            )
            best_value: Optional[float] = None
            objective_history: list[Dict[str, Any]] = []
            trial_budgeted = False
            self.last_adaptive_memory_profile = {}

            def validation_callback(
                epoch: int,
                metrics: Mapping[str, Mapping[str, float]],
                train_sizes: Mapping[str, int],
            ) -> bool:
                nonlocal best_value
                values = objective_values(
                    metrics,
                    self.hpo_config.objective.metric,
                )
                value = reduce_objective(
                    values,
                    self.hpo_config.objective.multitask_reduction,
                    train_sizes,
                )
                if not math.isfinite(value):
                    raise ValueError(
                        f"Trial objective is not finite: {value}."
                    )
                if best_value is None:
                    best_value = value
                elif self.hpo_config.objective.direction == "minimize":
                    best_value = min(best_value, value)
                else:
                    best_value = max(best_value, value)
                objective_history.append({
                    "epoch": epoch,
                    "value": value,
                })
                trial.report(value, step=epoch)
                return trial.should_prune()

            if self.base_dict["logging"].get("level") != "debug":
                logger.setLevel(logging.WARNING)

            try:
                trial_config = self.config_class.model_validate(sampled)
                if not trial_config.validate_config():
                    raise ValueError(
                        f"Invalid sampled configuration for scope {scope}."
                    )

                def prepare_trial(attempt_trainer: Any) -> None:
                    """Attach this Optuna trial's managed-run context."""
                    attempt_trainer.configure_managed_run(
                        log_dir=trial_root,
                        checkpoint_dir=checkpoint_root,
                        campaign_hash=self.campaign_hash,
                        run_mode="hpo",
                        validation_callback=(
                            validation_callback
                            if get_is_master()
                            else None
                        ),
                        external_distributed=True,
                    )

                result = self._run_adaptive_trainer(
                    trial_config,
                    prepare_trial,
                )
                pruned = bool(result.get("pruned"))
                if get_is_master():
                    if best_value is None:
                        raise ValueError(
                            "Trial completed without a validation objective."
                        )
                    trial.set_user_attr("decoded_params", decoded)
                    if pruned:
                        study.tell(
                            trial,
                            state=optuna.trial.TrialState.PRUNED,
                        )
                        self._write_trial_status(
                            trial_root,
                            PRUNED_STATUS,
                            decoded,
                            objective=best_value,
                            objective_history=objective_history,
                        )
                    else:
                        study.tell(trial, best_value)
                        self._write_trial_status(
                            trial_root,
                            COMPLETE_STATUS,
                            decoded,
                            objective=best_value,
                            objective_history=objective_history,
                        )
                trial_budgeted = True
            except Exception as exc:
                if not _is_recoverable_trial_failure(exc):
                    raise
                if get_is_master():
                    fingerprint = failure_fingerprint(exc)
                    memory_information = getattr(
                        self,
                        "last_adaptive_memory_profile",
                        {},
                    )
                    trial.set_user_attr("error_fingerprint", fingerprint)
                    trial.set_user_attr(
                        "memory_information",
                        memory_information,
                    )
                    study.tell(
                        trial,
                        state=optuna.trial.TrialState.FAIL,
                    )
                    self._write_trial_status(
                        trial_root,
                        FAILED_STATUS,
                        decoded,
                        error=fingerprint,
                        objective_history=objective_history,
                        memory_information=memory_information,
                    )
                logger.warning(
                    "HPO trial %d failed: %s",
                    trial_number,
                    exc,
                )
            finally:
                _release_training_state()
            if trial_budgeted:
                budgeted += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= failure_limit:
                    raise RuntimeError(
                        f"HPO scope '{scope}' reached "
                        f"{consecutive_failures} consecutive failed trials. "
                        "Correct the configuration or search space."
                    )

        self._configure_campaign_logging()
        if get_is_master():
            try:
                best_trial = study.best_trial
            except ValueError as exc:
                raise RuntimeError(
                    f"HPO scope '{scope}' has no completed trial."
                ) from exc
            decoded = best_trial.user_attrs.get("decoded_params")
            if not isinstance(decoded, dict):
                distribution_by_path = self.hpo_config.search_space
                decoded = {}
                for path, encoded in best_trial.params.items():
                    distribution = distribution_by_path[path]
                    if distribution.distribution == "categorical":
                        index = int(str(encoded).split("_")[-1])
                        decoded[path] = copy.deepcopy(
                            distribution.choices[index]
                        )
                    else:
                        decoded[path] = encoded
            best_payload = {
                "trial_number": best_trial.number,
                "objective": best_trial.value,
                "parameters": decoded,
            }
            _atomic_json(
                _hpo_scope_root(self.paths.log_root, scope) / "best.json",
                best_payload,
            )
        else:
            best_payload = None
        best_payload = _broadcast_object(best_payload)
        if get_is_master():
            logger.info(
                "HPO scope %s selected trial %d with objective %.8g; "
                "best parameters: %s. Best artifact: %s.",
                scope,
                best_payload["trial_number"],
                best_payload["objective"],
                json.dumps(best_payload["parameters"], sort_keys=True),
                (_hpo_scope_root(self.paths.log_root, scope)
                 / "best.json").resolve(),
            )

        selected = copy.deepcopy(dict(scope_config))
        for path, value in best_payload["parameters"].items():
            set_dotted_value(selected, path, value)
        validated = self.config_class.model_validate(selected)
        if not validated.validate_config():
            raise RuntimeError(
                f"Selected HPO configuration for '{scope}' is invalid."
            )
        return selected

    def _run_hpo(self) -> Dict[str, Dict[str, Any]]:
        """Run all configured studies and return selected scope configs."""
        validate_search_space(
            self.base_dict,
            self.hpo_config,
            self.config_class,
        )
        selected: Dict[str, Dict[str, Any]] = {}
        for scope, scope_config in _study_scope_configs(
            self.base_dict
        ).items():
            selected[scope] = self._run_hpo_scope(
                scope,
                scope_config,
            )
        return selected

    def _fixed_scopes(self) -> Dict[str, Dict[str, Any]]:
        """Return one fixed campaign scope when HPO is disabled."""
        scope = "multitask" if self.base_dict["multitask"] else "fixed"
        return {scope: copy.deepcopy(self.base_dict)}

    def _start_seed_cloud(self, seed: int) -> Dict[str, Any]:
        """Start one cloud run shared by all dataset runs for a seed."""
        logging_config = self.config.logging
        context: Dict[str, Any] = {}
        if not logging_config.use_cloud or not get_is_master():
            return context

        backend = logging_config.cloud_backend.lower()
        run_id = f"{self.campaign_hash}-seed-{seed}"
        if backend in {"wandb", "both"}:
            import wandb

            wandb.init(
                project=(
                    logging_config.project
                    or self.base_dict["model_type"]
                ),
                entity=logging_config.entity,
                id=run_id,
                name=f"seed_{seed}",
                group=self.campaign_hash,
                resume="allow",
                mode=(
                    "offline"
                    if logging_config.offline
                    else "online"
                ),
                tags=list(logging_config.tags) + [f"seed_{seed}"],
                config=self.base_dict,
                dir=str(self.paths.log_root.resolve()),
            )
            context["wandb"] = True

        if backend in {"comet", "both"}:
            import comet_ml

            api_key = (
                logging_config.api_key
                or os.environ.get("COMET_API_KEY")
            )
            if not api_key:
                warnings.warn(
                    "Comet API key is missing; the per-seed Comet run "
                    "was not created.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                key_path = (
                    self.paths.seed_log_root(seed)
                    / "cloud"
                    / "comet_experiment_key.txt"
                )
                kwargs = {
                    "api_key": api_key,
                    "project_name": (
                        logging_config.project
                        or self.base_dict["model_type"]
                    ),
                }
                if logging_config.entity:
                    kwargs["workspace"] = logging_config.entity
                if key_path.is_file():
                    experiment = comet_ml.ExistingExperiment(
                        previous_experiment=(
                            key_path.read_text(encoding="utf-8").strip()
                        ),
                        **kwargs,
                    )
                else:
                    experiment = comet_ml.Experiment(**kwargs)
                    key_path.parent.mkdir(parents=True, exist_ok=True)
                    key_path.write_text(
                        experiment.get_key(),
                        encoding="utf-8",
                    )
                experiment.set_name(f"seed_{seed}")
                experiment.add_tags(
                    list(logging_config.tags) + [f"seed_{seed}"]
                )
                experiment.log_parameters(self.base_dict)
                context["comet"] = experiment
        return context


    def _finish_seed_cloud(self, context: Mapping[str, Any]) -> None:
        """Finish a campaign-owned per-seed cloud run."""
        if context.get("wandb"):
            import wandb

            wandb.finish()
        comet_experiment = context.get("comet")
        if comet_experiment is not None:
            comet_experiment.end()

    def _run_seed(
        self,
        seed: int,
        selected_configs: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Execute all selected configurations for one independent seed."""
        seed_log_root = self.paths.seed_log_root(seed)
        seed_checkpoint_root = self.paths.seed_checkpoint_root(seed)
        cloud_context = self._start_seed_cloud(seed)
        try:
            for selected in selected_configs.values():
                trainer = None
                try:
                    scoped = _scoped_config(selected, seed)
                    final_config = self.config_class.model_validate(scoped)
                    if not final_config.validate_config():
                        raise ValueError(
                            "Invalid final configuration for seed "
                            f"{seed}."
                        )
                    is_neural = (
                        self.base_dict["model_type"]
                        not in DETERMINISTIC_MODELS
                    )
                    if is_neural:
                        def prepare_seed(
                            attempt_trainer: Any,
                        ) -> None:
                            """Attach this seed's managed-run context."""
                            attempt_trainer.comet_experiment = (
                                cloud_context.get("comet")
                            )
                            attempt_trainer.configure_managed_run(
                                log_dir=seed_log_root,
                                checkpoint_dir=seed_checkpoint_root,
                                campaign_hash=self.campaign_hash,
                                run_mode="final",
                                external_distributed=True,
                                external_cloud=bool(cloud_context),
                            )

                        self._run_adaptive_trainer(
                            final_config,
                            prepare_seed,
                        )
                    else:
                        trainer = ModelRegistry.create_trainer(final_config)
                        trainer.comet_experiment = cloud_context.get("comet")
                        trainer.log_dir_override = seed_log_root.resolve()
                        trainer.ckpt_dir_override = (
                            seed_checkpoint_root.resolve()
                        )
                        trainer.campaign_hash = self.campaign_hash
                        trainer.external_cloud = bool(cloud_context)
                        trainer.run()
                finally:
                    trainer = None
                    _release_training_state()
        finally:
            if get_is_master():
                self._finish_seed_cloud(cloud_context)

    def _run_seeds(
        self,
        selected_configs: Mapping[str, Mapping[str, Any]],
    ) -> tuple[Dict[str, Any], bool]:
        """Run configured seeds with consecutive-error stopping."""
        seeds = list(self.base_dict["seeds"])
        attempted: list[int] = []
        succeeded: list[int] = []
        failed: list[Dict[str, Any]] = []
        skipped: list[int] = []
        unattempted: list[int] = []
        previous_failure: Optional[str] = None
        first_attempt_succeeded: Optional[bool] = None

        pending: list[int] = []
        for seed in seeds:
            if _seed_scope_is_complete(
                self.paths,
                self.campaign_hash,
                seed,
                selected_configs,
            ):
                skipped.append(seed)
                logger.info(
                    "Seed %d is already complete and compatible; skipping.",
                    seed,
                )
                continue
            pending.append(seed)

        for index, seed in enumerate(pending):
            attempted.append(seed)
            logger.info(
                "Starting seed %d; artifacts: %s.",
                seed,
                self.paths.seed_log_root(seed).resolve(),
            )
            try:
                self._run_seed(seed, selected_configs)
            except Exception as exc:
                fingerprint = failure_fingerprint(exc)
                failed.append({
                    "seed": seed,
                    "fingerprint": fingerprint,
                })
                if first_attempt_succeeded is None:
                    first_attempt_succeeded = False
                repeated = fingerprint == previous_failure
                previous_failure = fingerprint
                logger.exception(
                    "Seed %d failed with fingerprint %s",
                    seed,
                    fingerprint,
                )
                if repeated:
                    unattempted.extend(pending[index + 1:])
                    break
            else:
                succeeded.append(seed)
                logger.info(
                    "Seed %d completed; artifacts: %s.",
                    seed,
                    self.paths.seed_log_root(seed).resolve(),
                )
                if first_attempt_succeeded is None:
                    first_attempt_succeeded = True
                previous_failure = None

        invocation = {
            "attempted": attempted,
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "unattempted": unattempted,
            "complete": not failed,
        }
        if not failed:
            summary_eligible = True
        else:
            summary_eligible = bool(
                first_attempt_succeeded and succeeded
            )
        return invocation, summary_eligible

    def _log_campaign_summary(
        self,
        summary: Mapping[str, Any],
    ) -> None:
        """Log compact final dataset metric summaries at INFO."""
        logger.info(
            "Campaign summary: %s",
            (self.paths.summary_root / "summary.json").resolve(),
        )
        for row in summary["test_summary"]:
            message = (
                "%s test %s: count=%d, mean=%.6g, median=%.6g"
            )
            arguments: list[Any] = [
                row["dataset"],
                row["metric"],
                row["count"],
                row["mean"],
                row["median"],
            ]
            if "std" in row:
                message += ", std=%.6g"
                arguments.append(row["std"])
            logger.info(message, *arguments)


    def _update_cloud_summary(
        self,
        summary: Mapping[str, Any],
    ) -> None:
        """Update one stable cloud summary run after eligible execution."""
        logging_config = self.config.logging
        if not logging_config.use_cloud or not get_is_master():
            return

        metrics: Dict[str, float] = {}
        for row in summary["test_summary"]:
            prefix = f"{row['dataset']}/test/{row['metric']}"
            for statistic in ("mean", "median", "std"):
                if statistic in row:
                    metrics[f"{prefix}/{statistic}"] = row[statistic]
        metrics["summary/completed_seed_count"] = len({
            row["seed"]
            for row in summary["test_runs"]
        })

        backend = logging_config.cloud_backend.lower()
        run_id = f"{self.campaign_hash}-summary"
        if backend in {"wandb", "both"}:
            import wandb

            run = wandb.init(
                project=(
                    logging_config.project
                    or self.base_dict["model_type"]
                ),
                entity=logging_config.entity,
                id=run_id,
                name=run_id,
                group=self.campaign_hash,
                resume="allow",
                mode=(
                    "offline"
                    if logging_config.offline
                    else "online"
                ),
                tags=list(logging_config.tags) + ["summary"],
            )
            run.log(metrics)
            run.summary["invocation"] = summary["invocation"]
            run.finish()

        if backend in {"comet", "both"}:
            import comet_ml

            api_key = (
                logging_config.api_key
                or os.environ.get("COMET_API_KEY")
            )
            if not api_key:
                warnings.warn(
                    "Comet API key is missing; the stable summary run "
                    "was not updated.",
                    UserWarning,
                    stacklevel=2,
                )
                return
            key_path = (
                self.paths.summary_root
                / "comet_experiment_key.txt"
            )
            kwargs = {
                "api_key": api_key,
                "project_name": (
                    logging_config.project
                    or self.base_dict["model_type"]
                ),
            }
            if logging_config.entity:
                kwargs["workspace"] = logging_config.entity
            if key_path.is_file():
                experiment = comet_ml.ExistingExperiment(
                    previous_experiment=(
                        key_path.read_text(encoding="utf-8").strip()
                    ),
                    **kwargs,
                )
            else:
                experiment = comet_ml.Experiment(**kwargs)
                key_path.parent.mkdir(parents=True, exist_ok=True)
                key_path.write_text(
                    experiment.get_key(),
                    encoding="utf-8",
                )
            experiment.set_name(run_id)
            experiment.add_tags(
                list(logging_config.tags) + ["summary"]
            )
            experiment.log_metrics(metrics)
            experiment.log_other(
                "invocation",
                json.dumps(summary["invocation"], sort_keys=True),
            )
            experiment.end()

    def run(self) -> Dict[str, Any]:
        """Run HPO if enabled, then execute independent configured seeds."""
        self._configure_campaign_logging()
        self._save_campaign_config()
        hpo_enabled = self.hpo_config.enabled
        if (
            hpo_enabled
            and self.base_dict["model_type"] in DETERMINISTIC_MODELS
        ):
            warnings.warn(
                f"{self.base_dict['model_type']} is deterministic; ignoring "
                "the configured HPO section.",
                UserWarning,
                stacklevel=2,
            )
            hpo_enabled = False

        initialization_seed = (
            self.hpo_config.seed
            if hpo_enabled
            else self.base_dict["seeds"][0]
        )
        self._initialize_distributed(initialization_seed)
        try:
            selected = (
                self._run_hpo()
                if hpo_enabled
                else self._fixed_scopes()
            )
            invocation, summary_eligible = self._run_seeds(selected)
            self._configure_campaign_logging()
            summary = None
            if summary_eligible and get_is_master():
                compatible_configs: Dict[
                    tuple[int, str],
                    Dict[str, Any],
                ] = {}
                logs_root = self.paths.log_root / "logs"
                discovered_seeds = set(self.base_dict["seeds"])
                if logs_root.is_dir():
                    for seed_root in logs_root.glob("seed_*"):
                        try:
                            discovered_seeds.add(
                                int(seed_root.name.removeprefix("seed_"))
                            )
                        except ValueError:
                            continue
                for seed in discovered_seeds:
                    expected = _expected_configs(selected, seed)
                    for dataset_name, config in expected.items():
                        compatible_configs[(seed, dataset_name)] = config
                summary = write_campaign_summary(
                    self.paths,
                    self.campaign_hash,
                    invocation,
                    compatible_configs,
                )
                if summary is not None:
                    self._log_campaign_summary(summary)
                    self._update_cloud_summary(summary)
            if invocation["failed"]:
                raise CampaignExecutionError(
                    "One or more seed executions failed. See campaign "
                    f"artifacts at {self.paths.log_root.resolve()}."
                )
            return {
                "campaign_hash": self.campaign_hash,
                "selected_configs": selected,
                "invocation": invocation,
                "summary": summary,
            }
        finally:
            if self.distributed_initialized:
                clean_torch_distributed()
