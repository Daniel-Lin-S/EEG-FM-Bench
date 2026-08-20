"""
Abstract trainer base class for baseline models.
"""
import csv
import json
import shutil
import time
from contextlib import nullcontext
import datetime
import os
from enum import Enum
import math
import logging
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import comet_ml
import datasets
import pandas as pd
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score, cohen_kappa_score, f1_score
from torch import Tensor
from torch.utils.data import DataLoader

from baseline.abstract.adapter import AbstractDataLoaderFactory
from baseline.abstract.config import AbstractConfig, BaseLoggingArgs
from baseline.adaptive_batching import (
    MEASURED_MEMORY_MODEL_VERSION,
    configure_cuda_allocator,
    estimate_fixed_training_bytes,
    profile_payload,
    resolve_training_memory_mode,
)
from baseline.hpo.artifacts import check_completion_compatibility
from baseline.utils.identity import IDENTITY_VERSION, short_identity
from baseline.utils.lora import (
    inject_lora, get_lora_state_dict, load_lora_state_dict, get_model_lora_targets
)
from baseline.utils.common import seed_torch
from baseline.utils.run_artifacts import get_config_hash, save_resolved_config
from common.log import setup_log
from data.processor.wrapper import get_dataset_n_class, get_dataset_category, get_dataset_shape_info
from common.distributed.env import get_is_master, get_global_rank, get_local_rank, get_world_size, get_master_addr, \
    get_master_port, clean_torch_distributed
from common.distributed.loader import DistributedGroupBatchSampler
COMPLETION_CONFIG_HASH_VERSION = IDENTITY_VERSION


logger = logging.getLogger("baseline")


class HpoStopReason(str, Enum):
    """Terminal reason requested by an HPO validation callback."""

    NONE = "none"
    OPTUNA_PRUNED = "optuna_pruned"
    PATIENCE = "patience"


class ChainableSequentialLR(torch.optim.lr_scheduler.SequentialLR):
    """A ``SequentialLR`` that never passes an epoch to child schedulers.

    PyTorch's ``SequentialLR`` calls ``scheduler.step(0)`` at each milestone.
    That deprecated epoch argument emits a warning on recent PyTorch releases.
    Calling ``step()`` starts the next scheduler at its initial chainable state
    and gives the same warmup-to-cosine transition used by this benchmark.
    """

    def step(self):
        self.last_epoch += 1
        # ``SequentialLR`` selects schedulers with ``bisect_right``. With the
        # two schedulers used here, this crosses the warmup milestone without
        # taking its deprecated ``step(0)`` path.
        scheduler_index = 0 if self.last_epoch < self._milestones[0] else 1
        scheduler = self._schedulers[scheduler_index]
        if scheduler_index == 1 and self.last_epoch == self._milestones[0]:
            # Match ``SequentialLR.step(0)`` by restoring the next scheduler
            # base rates ourselves, then use its chainable ``step()`` call.
            for param_group, base_lr in zip(self.optimizer.param_groups, scheduler.base_lrs):
                param_group["lr"] = base_lr
        scheduler.step()
        self._last_lr = scheduler.get_last_lr()


METRIC_PRECISION_DICT = {
    "lr": "6e",
    "header_lr": "6e",
    "encoder_lr": "6e",
    "gram": "2f",
    "accuracy": "3f",
    "acc": "3f",
    "f1": "3f",
    "pr": "3f",
    "recall": "3f",
    "cohen_kappa": "3f",
    "auroc": "3f",
    "auc_pr": "3f",
    "balanced_accuracy": "3f",
    "balanced_acc": "3f",
    "f1_weighted": "3f",
    "loss": "4f",
}


def format_console_log_dict(log_data: dict, prefix: str = 'train') -> str:
    """
    Format log dictionary with proper precision.

    Args:
        log_data: Dictionary of log metrics
        prefix: Prefix to remove from keys (e.g., 'train/')

    Returns:
        Formatted log string
    """
    prefix = f"{prefix}/"
    log_data = {key[len(prefix):] if key.startswith(prefix) else key: value for key, value in log_data.items()}
    formatted_log = ", ".join([
        f"{key}: {value:.{METRIC_PRECISION_DICT.get(key, '5e')}}" if isinstance(value, float)
        else f"{key}: {value}"
        for key, value in log_data.items()
    ])
    formatted_log = f"{prefix[:-1]} {formatted_log}"
    return formatted_log


class AbstractTrainer(ABC):
    """Abstract base trainer for all baseline models."""

    def __init__(self, cfg: AbstractConfig):
        self.cfg = cfg
        self.model_type = cfg.model_type
        self.multitask = cfg.multitask

        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.loss_fn = None

        self.epoch = 0
        self.current_step = 0

        self.world_size = 1
        self.rank = 0
        self.local_rank = 0

        # Dataset information
        self.ds_conf = cfg.data.datasets
        self.num_ds = len(self.ds_conf)

        self.ds_info = {}
        self.montage_info = {}
        self.dataloader_factory: Optional[AbstractDataLoaderFactory] = None

        self.start_time = datetime.datetime.now()
        self.comet_experiment = None
        self.tensorboard_writer: Optional[Any] = None

        self.ckpt_dir: str = ""
        self.log_dir: str = ""
        self.execution_id: str = ""
        self.tensorboard_writers: Dict[str, Any] = {}
        self.csv_file: Optional[Any] = None
        self.csv_writer: Optional[csv.DictWriter] = None
        self.current_dataset: Optional[str] = None
        self.final_checkpoint_paths: Dict[str, Path] = {}
        self.final_test_metrics: Dict[str, Dict[str, float]] = {}
        self.final_validation_metrics: Dict[str, Dict[str, float]] = {}
        self.latest_eval_counts: Dict[str, int] = {}
        self._loader_build_seconds: Dict[str, float] = {}
        self._evaluation_timings: Dict[str, Dict[str, float]] = {}
        self.run_mode = "legacy"
        self.hpo_logging_mode = "full"  # Options: "full", "reduced"
        self.campaign_hash: Optional[str] = None
        self.selection_provenance: Optional[Mapping[str, Any]] = None
        self.campaign_invocation_id: Optional[str] = None
        self.checkpoint_cleanup_failures: List[str] = []
        self.campaign_aliases: frozenset[str] = frozenset()
        self.log_dir_override: Optional[Path] = None
        self.ckpt_dir_override: Optional[Path] = None
        self.validation_callback: Optional[
            Callable[..., Union[HpoStopReason, str, bool]]
        ] = None
        self.external_distributed = False
        self.external_cloud = False
        self.training_result: Dict[str, Any] = {}
        self.runtime_world_size = 1
        self.requested_global_batch_size = cfg.data.batch_size
        self.micro_batch_size = cfg.data.batch_size
        self.accumulation_steps = 1
        self.adaptive_batch_profile: Dict[str, Any] = {}
        self._runtime_batch_configured = False


        # LoRA tracking
        self.lora_modules: List[str] = []

        # Lazy-created pretrain reconstruction head (registered on model)
        self._pretrain_recon_head = None

    def setup_distributed(self):
        """Setup distributed training environment."""
        rank = get_global_rank()
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            self.rank = rank
            self.local_rank = get_local_rank()
            self.world_size = get_world_size()
            self.device = torch.device(f"cuda:{self.local_rank}")
            return

        local_rank = get_local_rank()
        world_size = get_world_size()
        master_addr = get_master_addr()
        master_port = get_master_port(
            job_id=int(os.environ.get("SLURM_JOB_ID", -1)),
            port=self.cfg.master_port
        )

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = str(local_rank)

        assert 0 <= local_rank < 8
        torch.cuda.set_device(local_rank)

        torch.distributed.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )

        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank

    def setup_device(self, device: Optional[str] = None):
        """Setup non-distributed device for analysis or single-GPU runs."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0

    def maybe_wrap_ddp(self, model: nn.Module, find_unused_parameters: bool = True) -> nn.Module:
        """Wrap model with DDP if distributed is initialized, otherwise return as-is."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank], find_unused_parameters=find_unused_parameters
            )
        return model

    def encode_str(self, s: str, max_length=512):
        """Encode string to tensor for distributed broadcasting."""
        encoded = s.encode()[:max_length]
        encoded += b'\0' * (max_length - len(encoded))
        return torch.ByteTensor(list(encoded)).to(self.device)

    def broadcast_str(self, s, max_length=512, rank=0):
        """Broadcast string across distributed processes."""
        if rank == 0:
            tensor = self.encode_str(s, max_length)
        else:
            tensor = torch.zeros(max_length, dtype=torch.uint8, device=self.device)
        torch.distributed.broadcast(tensor, src=0)

        bytes_list = tensor.cpu().numpy().tobytes()
        string = bytes_list.split(b'\0')[0].decode()
        return string

    def configure_managed_run(
        self,
        log_dir: Path,
        checkpoint_dir: Path,
        campaign_hash: str,
        run_mode: str,
        campaign_aliases: Optional[Iterable[str]] = None,
        validation_callback: Optional[
            Callable[..., Union[HpoStopReason, str, bool]]
        ] = None,
        selection_provenance: Optional[Mapping[str, Any]] = None,
        invocation_id: Optional[str] = None,
        external_distributed: bool = False,
        external_cloud: bool = False,
        hpo_logging_mode: str = "full",
    ) -> None:
        """Configure one campaign-controlled trainer execution."""
        if run_mode not in {"hpo", "final", "recovery"}:
            raise ValueError(f"Unsupported managed run mode: {run_mode}.")
        if hpo_logging_mode not in {"full", "reduced"}:
            raise ValueError(f"Unsupported HPO logging mode: {hpo_logging_mode}.")
        self.log_dir_override = log_dir.resolve()
        self.ckpt_dir_override = checkpoint_dir.resolve()
        self.campaign_hash = campaign_hash
        self.campaign_aliases = frozenset(
            campaign_aliases or ()
        )
        self.run_mode = run_mode
        self.hpo_logging_mode = hpo_logging_mode
        self.validation_callback = validation_callback
        self.external_distributed = external_distributed
        self.selection_provenance = (
            dict(selection_provenance) if selection_provenance else None
        )
        self.campaign_invocation_id = invocation_id
        self.external_cloud = external_cloud

    def configure_runtime_batching(
        self,
        global_batch_size: int,
        micro_batch_size: int,
        world_size: int,
    ) -> None:
        """Configure one exact global-to-micro batch decomposition.

        Parameters
        ----------
        global_batch_size : int
            Requested samples across all ranks per optimizer update.
        micro_batch_size : int
            Samples loaded by each rank for one forward/backward pass.
        world_size : int
            Number of ranks participating in gradient synchronization.
        """
        if world_size <= 0:
            raise ValueError(
                f"Expected a positive world size, but got {world_size}."
            )
        denominator = micro_batch_size * world_size
        if micro_batch_size <= 0 or global_batch_size % denominator != 0:
            raise ValueError(
                "The global batch must be divisible by micro_batch_size "
                f"* world_size, but got {global_batch_size}, "
                f"{micro_batch_size}, and {world_size}."
            )
        self.requested_global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.accumulation_steps = global_batch_size // denominator
        if self.dataloader_factory is not None:
            self.dataloader_factory.batch_size = micro_batch_size
        self.runtime_world_size = world_size
        self._runtime_batch_configured = True
        self.adaptive_batch_profile.update({
            "requested_global_batch_size": global_batch_size,
            "world_size": world_size,
            "micro_batch_size": micro_batch_size,
            "accumulation_steps": self.accumulation_steps,
        })

    def _ensure_runtime_batching(self) -> None:
        """Use accumulation one when no campaign override was supplied."""
        if self._runtime_batch_configured:
            return
        if self.requested_global_batch_size % self.world_size != 0:
            raise ValueError(
                "data.batch_size must be divisible by the distributed "
                f"world size, but got {self.requested_global_batch_size} "
                f"and {self.world_size}."
            )
        micro_batch = self.requested_global_batch_size // self.world_size
        self.configure_runtime_batching(
            self.requested_global_batch_size,
            micro_batch,
            self.world_size,
        )

    def _configure_cuda_memory_limit(self) -> None:
        """Apply and record this run's adaptive CUDA allocator ceiling."""
        args = self.cfg.training.adaptive_batching
        if not args.enabled or self.device.type != "cuda":
            return
        limit = configure_cuda_allocator(
            self.device,
            args.memory_reserve_fraction,
        )
        if limit is not None:
            self.adaptive_batch_profile.update(limit.as_dict())

    def _record_fixed_memory_estimate(self) -> None:
        """Record analytical model, gradient, and optimizer memory."""
        if self.model is None or self.optimizer is None:
            raise RuntimeError(
                "Model and optimizer are required for memory estimation."
            )
        fixed = estimate_fixed_training_bytes(
            self.model,
            self.optimizer,
        )
        self.adaptive_batch_profile.update({
            "memory_model_version": MEASURED_MEMORY_MODEL_VERSION,
            "training_memory_mode": resolve_training_memory_mode(
                self.cfg.training.freeze_encoder,
                self.cfg.training.lora.use_lora,
            ),
        })
        self.adaptive_batch_profile = profile_payload(
            self.adaptive_batch_profile,
            fixed,
        )
        self._write_adaptive_batch_profile()

    def _write_adaptive_batch_profile(self) -> None:
        """Persist the current adaptive batching diagnostics."""
        if not get_is_master() or not self.log_dir:
            return
        payload = dict(self.adaptive_batch_profile)
        if self.device.type == "cuda" and torch.cuda.is_available():
            payload["measured_peak_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(self.device)
            )
            payload["measured_peak_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(self.device)
            )
        path = Path(self.log_dir) / "adaptive_batching.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _move_batch_to_device(
        self,
        batch: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Move tensor values in one loader batch to the training device.

        Parameters
        ----------
        batch : Mapping[str, Any]
            Loader batch containing tensor and metadata values.

        Returns
        -------
        dict[str, Any]
            New batch mapping whose tensor values reside on ``self.device``.
        """
        non_blocking = bool(
            self.cfg.data.pin_memory
            and getattr(self.device, "type", None) == "cuda"
        )
        return {
            key: value.to(self.device, non_blocking=non_blocking)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }

    def calibrate_cuda_memory(self) -> Dict[str, Any]:
        """Measure one disposable micro-batch update on the current rank.

        Returns
        -------
        dict[str, Any]
            Analytical fixed-memory values, allocator limits, and measured
            CUDA peaks for the calibration update.

        Raises
        ------
        RuntimeError
            If calibration is requested without CUDA or training data.
        """
        seed_torch(self.cfg.seed)
        self.setup_distributed()
        self._ensure_runtime_batching()
        self._configure_cuda_memory_limit()
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA memory calibration requires an available CUDA device."
            )

        if self.multitask:
            self.collect_dataset_info(mixed=True)
            model = self.setup_model()
            train_loader, _ = self.create_dataloader(
                datasets.Split.TRAIN
            )
        else:
            if not self.ds_conf:
                raise RuntimeError(
                    "CUDA calibration requires at least one dataset."
                )
            dataset_name, dataset_config = next(iter(self.ds_conf.items()))
            self.current_dataset = dataset_name
            self.collect_dataset_info(
                mixed=False,
                ds_name=dataset_name,
            )
            model = self.setup_model()
            train_loader, _ = self.create_single_dataloader(
                dataset_name,
                dataset_config,
                datasets.Split.TRAIN,
            )
        if not isinstance(train_loader, DataLoader):
            raise TypeError(
                "CUDA calibration expected train_loader to be a DataLoader."
            )
        self.setup_optimizer_and_scheduler(model, train_loader)
        try:
            raw_batch = next(iter(train_loader))
        except StopIteration as exc:
            raise RuntimeError(
                "CUDA calibration cannot use an empty training loader."
            ) from exc
        batch = self._move_batch_to_device(raw_batch)
        labels = batch.get("label")
        if not isinstance(labels, torch.Tensor) or labels.shape[0] <= 0:
            raise RuntimeError(
                "CUDA calibration requires a non-empty tensor label batch."
            )
        sample_count = int(labels.shape[0])
        self.model.train()
        self.optimizer.zero_grad()
        _, loss = self.train_step(batch, labels)
        if not torch.isfinite(loss):
            raise ValueError(
                "CUDA calibration produced a non-finite training loss: "
                f"{loss.detach().item()}."
            )
        self.scaler.scale(loss * sample_count).backward()
        self._finish_accumulated_update(sample_count)
        torch.cuda.synchronize(self.device)
        self.adaptive_batch_profile.update({
            "calibration_batch_size": sample_count,
            "calibration_peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(self.device)
            ),
            "calibration_peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(self.device)
            ),
        })
        return dict(self.adaptive_batch_profile)

    def get_train_io_path(
        self,
        args: BaseLoggingArgs,
    ) -> tuple[str, str]:
        """Create or select log and checkpoint roots for this execution."""
        if not get_is_master():
            return "", ""

        config = self.cfg.model_dump(mode="json")
        self.execution_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.execution_id = f"{self.execution_id}-{os.getpid()}"
        if self.log_dir_override is not None:
            if self.ckpt_dir_override is None:
                raise RuntimeError(
                    "Managed log override requires a checkpoint override."
                )
            log_path = self.log_dir_override
            ckpt_path = self.ckpt_dir_override
        else:
            config_hash = get_config_hash(config, self.multitask)
            experiment_name = (
                f"{args.experiment_name}-{short_identity(config_hash)}"
            )
            if self.multitask:
                experiment_name = (
                    f"{experiment_name}-{self.execution_id}"
                )
            run_dir = Path(args.run_dir)
            log_path = (
                run_dir / "log" / "baseline" / self.model_type
                / experiment_name
            )
            ckpt_path = (
                run_dir / "ckpt" / "baseline" / self.model_type
                / experiment_name
            )
        log_path.mkdir(parents=True, exist_ok=True)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        save_resolved_config(
            config,
            log_path / "configs" / f"{self.execution_id}.yaml",
        )
        return str(log_path.resolve()), str(ckpt_path.resolve())

    def _has_output(self, output_name: str) -> bool:
        """Return whether one local trace type is enabled."""
        return output_name in self.cfg.logging.outputs

    def _completion_path(self, ds_name: str) -> Path:
        """Return completion metadata location for one dataset."""
        return Path(self.log_dir, 'datasets', ds_name, 'completion.json')

    def _resolved_config_hash(self) -> str:
        """Return the identity of this seed-scoped resolved config."""
        config = self.cfg.model_dump(mode="json")
        return get_config_hash(config, self.multitask)

    def _dataset_is_complete(
        self,
        ds_name: str,
        ds_config: str,
    ) -> bool:
        """Return whether matching final artifacts already exist."""
        completion_path = self._completion_path(ds_name)
        if not completion_path.is_file():
            return False
        try:
            completion = json.loads(
                completion_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            logger.warning(
                "Ignoring invalid completion metadata: %s",
                completion_path,
            )
            return False
        metrics = completion.get("test_metrics")
        numeric_metrics = [] if not isinstance(metrics, dict) else [
            float(value)
            for value in metrics.values()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
        ]
        compatible = (
            completion.get("dataset_config") == ds_config
            and completion.get("status") == "completed"
            and completion.get("config_hash") == self._resolved_config_hash()
            and bool(numeric_metrics)
            and all(math.isfinite(value) for value in numeric_metrics)
        )
        if self.campaign_hash is None:
            return compatible
        return check_completion_compatibility(
            completion_path,
            self.campaign_hash,
            self.cfg.seed,
            self.cfg.model_dump(mode="json"),
            campaign_aliases=self.campaign_aliases,
        ).compatible

    def _reset_dataset_outputs(self, ds_name: str) -> None:
        """Remove stale traces before retrying a dataset."""
        if not get_is_master():
            return
        self._completion_path(ds_name).unlink(missing_ok=True)
        csv_path = Path(self.log_dir, 'csv', f'{ds_name}.csv')
        csv_path.unlink(missing_ok=True)
        tensorboard_path = Path(self.log_dir, 'tensorboard', ds_name)
        if tensorboard_path.exists():
            shutil.rmtree(tensorboard_path)
        checkpoint_dir = self._checkpoint_scope_directory(ds_name)
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

    def _reset_unified_outputs(self) -> None:
        """Remove incomplete shared-run artifacts before epoch-zero restart."""
        if not get_is_master():
            return
        had_artifacts = False
        for dataset_name in self.ds_conf:
            completion_path = self._completion_path(dataset_name)
            had_artifacts = had_artifacts or completion_path.exists()
            completion_path.unlink(missing_ok=True)
            tensorboard_path = Path(
                self.log_dir,
                "tensorboard",
                dataset_name,
            )
            if tensorboard_path.exists():
                had_artifacts = True
                shutil.rmtree(tensorboard_path)
        csv_path = Path(self.log_dir, "csv", "training.csv")
        had_artifacts = had_artifacts or csv_path.exists()
        csv_path.unlink(missing_ok=True)
        checkpoint_dir = self._checkpoint_scope_directory(None)
        if checkpoint_dir.exists():
            had_artifacts = True
            shutil.rmtree(checkpoint_dir)
        if had_artifacts:
            logger.warning(
                "Reset incomplete multitask artifacts before restarting at "
                "epoch 0 under %s.",
                Path(self.log_dir).resolve(),
            )

    def _checkpoint_scope_directory(
        self,
        ds_name: Optional[str],
    ) -> Path:
        """Return the checkpoint directory for one training scope.

        Parameters
        ----------
        ds_name : str or None
            Dataset name for separate training, or ``None`` for multitask.

        Returns
        -------
        pathlib.Path
            Directory containing temporary and retained scope checkpoints.
        """
        if ds_name is None:
            return Path(self.ckpt_dir, "unified")
        return Path(self.ckpt_dir, "seperated", ds_name)

    def _cleanup_checkpoint_artifacts(
        self,
        ds_name: Optional[str],
        best_checkpoint: Path,
    ) -> list[str]:
        """Remove temporary checkpoints and return cleanup failures.

        Parameters
        ----------
        ds_name : str or None
            Dataset name for separate training, or ``None`` for multitask.
        best_checkpoint : pathlib.Path
            Validation-best checkpoint used for final evaluation.

        Returns
        -------
        list[str]
            Absolute paths that could not be removed.
        """
        if not get_is_master():
            return []
        scope_dir = self._checkpoint_scope_directory(ds_name)
        if not scope_dir.exists():
            return []
        retained = {best_checkpoint.resolve()}
        targets = [scope_dir]
        if self.cfg.logging.save_checkpoints:
            targets = [
                path
                for path in scope_dir.iterdir()
                if path.resolve() not in retained
            ]
        failures: list[str] = []
        for target in targets:
            try:
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink(missing_ok=True)
            except OSError as exc:
                resolved = str(target.resolve())
                failures.append(resolved)
                logger.warning(
                    "Checkpoint cleanup failed at %s: %s",
                    resolved,
                    exc,
                )
        self.checkpoint_cleanup_failures.extend(failures)
        return failures

    def _open_csv_writer(
        self,
        ds_name: str,
        append: bool = False,
    ) -> None:
        """Open one metric CSV trace for replacement or recovery.

        Parameters
        ----------
        ds_name : str
            Dataset name or the shared ``training`` trace name.
        append : bool, optional, default=False
            Whether to preserve an existing trace and append new events.
        """
        if not self._has_output('csv') or not get_is_master():
            return
        self._close_csv_writer()
        csv_path = Path(self.log_dir, 'csv', f'{ds_name}.csv')
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        mode = 'a' if append and csv_path.is_file() else 'w'
        write_header = mode == 'w' or csv_path.stat().st_size == 0
        self.csv_file = csv_path.open(mode, newline='', encoding='utf-8')
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=['timestamp', 'dataset', 'split', 'epoch', 'step',
                        'metric', 'value'],
        )
        if write_header:
            self.csv_writer.writeheader()

    def _close_csv_writer(self) -> None:
        """Flush and close the active metric CSV writer."""
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def _write_csv_metrics(self, log_data: dict, step: int) -> None:
        """Write numeric metrics from one trainer event to CSV."""
        if self.csv_writer is None:
            return
        for key, value in log_data.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            parts = key.split('/')
            dataset = self.current_dataset or ''
            if len(parts) >= 3 and parts[0] in self.ds_conf:
                dataset, split = parts[:2]
                metric = '/'.join(parts[2:])
            elif len(parts) >= 2:
                split, metric = parts[0], '/'.join(parts[1:])
            else:
                split, metric = '', key
            self.csv_writer.writerow({
                'timestamp': datetime.datetime.now().isoformat(),
                'dataset': dataset,
                'split': split,
                'epoch': self.epoch,
                'step': step,
                'metric': metric,
                'value': value,
            })
        self.csv_file.flush()

    def get_data_diagnostics(
        self,
        ds_name: str,
    ) -> Mapping[str, Any]:
        """Return data-pipeline diagnostics for one dataset completion.

        Parameters
        ----------
        ds_name : str
            Completed dataset name.

        Returns
        -------
        Mapping[str, Any]
            Provider-owned data diagnostics.
        """
        if self.dataloader_factory is None:
            return {}
        return self.dataloader_factory.get_data_diagnostics(ds_name)

    def get_model_diagnostics(
        self,
        ds_name: str,
    ) -> Mapping[str, Any]:
        """Return model diagnostics for one dataset completion.

        Parameters
        ----------
        ds_name : str
            Completed dataset name.

        Returns
        -------
        Mapping[str, Any]
            Provider-owned model diagnostics. The default is empty.
        """
        del ds_name
        return {}

    def get_training_diagnostics(
        self,
        ds_name: str,
    ) -> Mapping[str, Any]:
        """Return training-scheme diagnostics for one completion.

        Parameters
        ----------
        ds_name : str
            Completed dataset name.

        Returns
        -------
        Mapping[str, Any]
            Provider-owned training diagnostics. The default is empty.
        """
        del ds_name
        return {}

    def _record_loader_build_seconds(
        self,
        split: datasets.NamedSplit,
        elapsed_seconds: float,
    ) -> None:
        """Accumulate one successful data-loader construction duration."""
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
            raise ValueError(
                "Loader build duration must be finite and non-negative."
            )
        split_name = str(split)
        self._loader_build_seconds[split_name] = (
            self._loader_build_seconds.get(split_name, 0.0)
            + elapsed_seconds
        )

    def _record_evaluation_seconds(
        self,
        prefix: str,
        elapsed_seconds: float,
    ) -> None:
        """Accumulate one successful complete evaluation-pass duration."""
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
            raise ValueError(
                "Evaluation duration must be finite and non-negative."
            )
        split_name = "validation" if prefix == "eval" else prefix
        timing = self._evaluation_timings.setdefault(
            split_name,
            {"passes": 0.0, "total_seconds": 0.0, "latest_seconds": 0.0},
        )
        timing["passes"] += 1.0
        timing["total_seconds"] += elapsed_seconds
        timing["latest_seconds"] = elapsed_seconds

    def get_performance_diagnostics(self) -> Dict[str, Any]:
        """Return finite loader and end-to-end evaluation timing summaries."""
        evaluations: Dict[str, Dict[str, Union[int, float]]] = {}
        for split_name, timing in sorted(self._evaluation_timings.items()):
            passes = int(timing["passes"])
            total_seconds = float(timing["total_seconds"])
            if passes <= 0 or not math.isfinite(total_seconds):
                raise ValueError(
                    "Evaluation timing must have positive passes and finite "
                    "total seconds."
                )
            evaluations[split_name] = {
                "passes": passes,
                "total_seconds": total_seconds,
                "mean_seconds": total_seconds / passes,
                "latest_seconds": float(timing["latest_seconds"]),
            }
        loader_seconds = {
            split_name: float(seconds)
            for split_name, seconds in sorted(
                self._loader_build_seconds.items()
            )
        }
        return {
            "scope": "unified" if self.multitask else "dataset",
            "loader_build_seconds": loader_seconds,
            "evaluation": evaluations,
        }

    def _build_completion_diagnostics(
        self,
        ds_name: str,
    ) -> Dict[str, Any]:
        """Build validated typed diagnostics for one completion artifact.

        Parameters
        ----------
        ds_name : str
            Completed dataset name.

        Returns
        -------
        Dict[str, Any]
            Non-empty data, model, and training namespaces.
        """
        providers = {
            "data": self.get_data_diagnostics,
            "model": self.get_model_diagnostics,
            "training": self.get_training_diagnostics,
        }
        diagnostics: Dict[str, Any] = {}
        for namespace, provider in providers.items():
            payload = provider(ds_name)
            if not isinstance(payload, Mapping):
                raise TypeError(
                    f"{namespace} diagnostics for {ds_name} must be a "
                    f"mapping, but got {type(payload).__name__}."
                )
            if payload:
                diagnostics[namespace] = dict(payload)
        diagnostics["performance"] = self.get_performance_diagnostics()
        try:
            json.dumps(diagnostics, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Completion diagnostics for {ds_name} must contain only "
                "finite JSON-serializable values."
            ) from exc
        return diagnostics

    def _write_completion(self, ds_name: str, ds_config: str) -> None:
        """Atomically persist final metadata after successful training."""
        checkpoint_path = self.final_checkpoint_paths.get(ds_name)
        metrics = self.final_test_metrics.get(ds_name)
        validation_metrics = self.final_validation_metrics.get(ds_name)
        if (
            checkpoint_path is None
            or metrics is None
            or validation_metrics is None
        ):
            raise RuntimeError(
                f'Cannot mark {ds_name} complete without final checkpoint and '
                "test and validation metrics."
            )
        completion_path = self._completion_path(ds_name)
        retain_checkpoint = self.cfg.logging.save_checkpoints
        stored_checkpoint = (
            str(checkpoint_path.resolve()) if retain_checkpoint else None
        )
        completion_path.parent.mkdir(parents=True, exist_ok=True)
        content = {
            'status': 'completed',
            'campaign_hash': self.campaign_hash,
            'campaign_identity_version': IDENTITY_VERSION,
            'config_hash': self._resolved_config_hash(),
            'seed': self.cfg.seed,
            'config_hash_version': COMPLETION_CONFIG_HASH_VERSION,
            'dataset_config': ds_config,
            'execution_id': self.execution_id,
            'invocation_id': self.campaign_invocation_id,
            'has_checkpoint': retain_checkpoint,
            'checkpoint_path': stored_checkpoint,
            'checkpoint_retention_requested': retain_checkpoint,
            'selection_provenance': self.selection_provenance,
            'validation_metrics': validation_metrics,
            'test_metrics': metrics,
            'completed_at': datetime.datetime.now().isoformat(),
            'batching': {
                'requested_global_batch_size': (
                    self.requested_global_batch_size
                ),
                'world_size': self.runtime_world_size,
                'micro_batch_size': self.micro_batch_size,
                'accumulation_steps': self.accumulation_steps,
            },
        }
        diagnostics = self._build_completion_diagnostics(ds_name)
        if diagnostics:
            content["diagnostics"] = diagnostics
        temporary_path = completion_path.with_suffix('.tmp')
        temporary_path.write_text(json.dumps(content, indent=2),
                                  encoding='utf-8')
        temporary_path.replace(completion_path)

    def setup_logging(self):
        log_dir, ckpt_dir = self.get_train_io_path(self.cfg.logging)
        # Broadcast paths in distributed environment
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            ckpt_dir = self.broadcast_str(ckpt_dir, max_length=512, rank=self.rank)
            log_dir = self.broadcast_str(log_dir, max_length=512, rank=self.rank)

        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir

        if get_is_master():
            file_path = None
            if self._has_output('log'):
                file_path = os.path.join(
                    log_dir,
                    'logs',
                    f'{self.execution_id}.log',
                )
            log_level = self.cfg.logging.level
            if self.run_mode == "hpo" and log_level != "debug":
                log_level = "warning"
            setup_log(
                file_path=file_path,
                start_time=self.start_time.timestamp(),
                name="baseline",
                level=log_level.upper(),
            )
            logger.debug(
                "Log dir: %s, checkpoint dir: %s",
                self.log_dir,
                self.ckpt_dir,
            )

        logger.debug(
            "Starting %s training with %d dataset(s): %s",
            self.cfg.model_type,
            self.num_ds,
            list(self.ds_conf),
        )

    def init_cloud_logging(self):
        """Initialize cloud logging (wandb, comet, etc.)."""
        if self.external_cloud:
            return
        if not self.cfg.logging.use_cloud:
            return

        if get_is_master():
            # Initialize logging based on backend configuration
            backend = self.cfg.logging.cloud_backend.lower()

            if backend in ['wandb', 'both']:
                self._init_wandb()

            if backend in ['comet', 'both']:
                self._init_comet()

    def init_tensorboard_logging(self):
        """Initialize TensorBoard writers selected by ``logging.outputs``."""
        if not self._has_output('tensorboard') or not get_is_master():
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "TensorBoard logging requires the 'tensorboard' package. "
                "Install project requirements before selecting tensorboard."
            ) from exc
        if self.multitask:
            tensorboard_dir = Path(self.log_dir, 'tensorboard')
            self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)
            logger.debug('TensorBoard logging enabled: %s', tensorboard_dir)

    def _open_dataset_tensorboard(self, ds_name: str) -> None:
        """Open an overwriteable TensorBoard writer for one dataset."""
        if not self._has_output('tensorboard') or not get_is_master():
            return
        from torch.utils.tensorboard import SummaryWriter

        if ds_name in self.tensorboard_writers:
            return
        tensorboard_dir = Path(self.log_dir, 'tensorboard', ds_name)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        self.tensorboard_writers[ds_name] = writer
        if not self.multitask:
            self.tensorboard_writer = writer
        logger.debug('TensorBoard logging enabled: %s', tensorboard_dir)

    def finish_tensorboard_logging(self):
        """Flush and close all local TensorBoard writers."""
        writers = list(self.tensorboard_writers.values())
        if self.tensorboard_writer is not None:
            writers.append(self.tensorboard_writer)
        seen_writers = set()
        for writer in writers:
            if id(writer) not in seen_writers:
                writer.close()
                seen_writers.add(id(writer))
        self.tensorboard_writers = {}
        self.tensorboard_writer = None

    def _log_to_tensorboard(
        self,
        log_data: dict,
        step: int,
        ds_name: Optional[str] = None,
    ) -> None:
        """Write numeric metrics to one TensorBoard trace."""
        writer = self.tensorboard_writer
        if ds_name is not None:
            writer = self.tensorboard_writers.get(ds_name)
        if writer is None:
            return
        for key, value in log_data.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                writer.add_scalar(key, value, global_step=step)

    def _init_wandb(self):
        """Initialize wandb logging with unified naming."""
        try:
            # Create wandb metrics list
            wandb_metrics = []
            if self.multitask:
                wandb_metrics = ["train/step"]

            for ds_name in self.ds_conf.keys():
                if not self.multitask:
                    wandb_metrics.append(f"{ds_name}/train/step")
                wandb_metrics.extend([
                    f"{ds_name}/eval/epoch",
                    f"{ds_name}/test/epoch"
                ])

            wandb_dir = os.path.join(self.cfg.logging.run_dir, 'log', 'baseline', 'wandb')

            if self.cfg.logging.project is None:
                logger.warning("Project name not set, using experiment_name as fallback")

            # Use unified run name from log directory
            run_name = os.path.basename(self.log_dir)

            # Setup wandb configuration with unified parameters
            wandb_config = {
                'dir': wandb_dir,
                'project': self.cfg.logging.project or self.cfg.logging.experiment_name,
                'name': run_name,
                'config': self.cfg.model_dump(),
                'tags': self.cfg.logging.tags,
                'mode': 'offline' if self.cfg.logging.offline else 'online',
            }

            # Add optional parameters if specified
            if self.cfg.logging.entity:
                wandb_config['entity'] = self.cfg.logging.entity

            # Set API key if specified
            if self.cfg.logging.api_key:
                os.environ['WANDB_API_KEY'] = self.cfg.logging.api_key

            wandb.init(**wandb_config)

            # Define step metrics
            if self.multitask:
                wandb.define_metric("train/step")

            for metric in wandb_metrics:
                idx = metric.rfind('/')
                if idx == -1:
                    raise ValueError('No prefix to set metric')
                wandb.define_metric(metric)
                group = metric[:idx]
                wandb.define_metric(f'{group}/*', step_metric=metric)

            logger.debug("Wandb logging enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")

    def _init_comet(self):
        try:
            # Setup comet configuration with unified parameters
            comet_config = {}

            # Set API key (from config or environment)
            api_key = self.cfg.logging.api_key or os.getenv('COMET_API_KEY')
            if not api_key:
                logger.warning("Comet API key not found, skipping comet logging")
                return

            comet_config['api_key'] = api_key
            comet_config['project_name'] = self.cfg.logging.project or self.cfg.logging.experiment_name

            if self.cfg.logging.entity:
                comet_config['workspace'] = self.cfg.logging.entity

            comet_config['experiment_name'] = (
                f"{self.model_type}_{'uni' if self.cfg.multitask else 'sep'}"
                f"_{datetime.datetime.now().strftime('%m%d_%H%M%S')}"
            )

            # Initialize comet experiment
            self.comet_experiment = comet_ml.Experiment(**comet_config)

            # Log configuration
            self.comet_experiment.log_parameters(self.cfg.model_dump())
            self.comet_experiment.add_tags(self.cfg.logging.tags)

            logger.debug("Comet.ml logging enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize comet.ml: {e}")
            self.comet_experiment = None

    def finish_cloud_logging(self):
        """Finish cloud logging."""
        if self.external_cloud:
            return
        if not get_is_master():
            return

        backend = self.cfg.logging.cloud_backend.lower()

        if backend in ['wandb', 'both']:
            self._finish_wandb()

        if backend in ['comet', 'both']:
            self._finish_comet()

    def _finish_wandb(self):
        """Finish wandb logging."""
        try:
            wandb.finish()
            logger.debug("Wandb logging finished")
        except Exception as e:
            logger.warning(f"Error finishing wandb: {e}")

    def _finish_comet(self):
        """Finish comet.ml logging."""
        try:
            self.comet_experiment.end()
            logger.debug("Comet.ml logging finished")
        except Exception as e:
            logger.warning(f"Error finishing comet.ml: {e}")

    def _create_ft_cloud_log_data(self, log_data: dict, prefix: str, ds_metric: dict):
        # eval epoch metrics
        cloud_data = deepcopy(log_data)

        # Add raw confusion matrix data for cloud logging backends
        for ds_name in ds_metric.keys():
            matrix = ds_metric[ds_name]['cm'].cpu().numpy()
            labels = self.ds_info[ds_name]['category']

            # Store raw matrix and labels for both wandb and comet to handle
            cloud_data.update({f"{ds_name}/{prefix}/cm_matrix": matrix})
            cloud_data.update({f"{ds_name}/{prefix}/cm_labels": labels})

        return cloud_data

    def _log_to_cloud(self, log_data: dict):
        """Log data to configured cloud services."""
        backend = self.cfg.logging.cloud_backend.lower()

        if backend in ['wandb', 'both']:
            self._log_to_wandb(log_data)

        if backend in ['comet', 'both']:
            self._log_to_comet(log_data)

    def _log_to_wandb(self, log_data: dict):
        """Log data to wandb."""
        try:
            # Separate confusion matrix data from regular metrics
            wandb_data = {}
            cm_data = {}

            for key, value in log_data.items():
                if 'cm_matrix' in key or 'cm_labels' in key:
                    cm_data[key] = value
                else:
                    wandb_data[key] = value

            # Create wandb tables for confusion matrices
            for key, matrix in cm_data.items():
                if key.endswith('cm_matrix'):
                    base_key = key.replace('cm_matrix', '')
                    labels_key = base_key + 'cm_labels'
                    if labels_key in cm_data:
                        labels = cm_data[labels_key]
                        # Create wandb table
                        df = pd.DataFrame(matrix, columns=labels)
                        confusion_table = wandb.Table(dataframe=df)
                        wandb_data[f"{base_key}/cm"] = confusion_table

            # Log all data to wandb
            wandb.log(wandb_data)
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {e}")

    def _log_to_comet(self, log_data: dict):
        """Log data to comet.ml."""
        if self.comet_experiment is None:
            return

        try:
            # Separate confusion matrix data from regular metrics
            metrics = {}
            cm_data = {}

            for key, value in log_data.items():
                if 'cm_matrix' in key or 'cm_labels' in key:
                    cm_data[key] = value
                else:
                    metrics[key] = value

            # Log regular metrics
            if metrics:
                self.comet_experiment.log_metrics(metrics)

            # Log confusion matrices
            for key, matrix in cm_data.items():
                if key.endswith('cm_matrix'):
                    base_key = key.replace('cm_matrix', '')
                    labels_key = base_key + 'cm_labels'
                    if labels_key in cm_data:
                        labels = cm_data[labels_key]
                        self.comet_experiment.log_confusion_matrix(
                            matrix=matrix,
                            labels=labels,
                            title=f"Confusion Matrix - {base_key.replace('/', '_')}"
                        )
        except Exception as e:
            logger.warning(f"Failed to log to comet.ml: {e}")

    def _calculate_metrics_for_dataset(
            self,
            labels: torch.Tensor,
            logits: torch.Tensor,
            ds_name: str,
            prefix: str,
            loss: float,
    ) -> Dict[str, float]:
        label_np = labels.numpy()
        pred_np = torch.argmax(logits, dim=-1).numpy()

        n_class = self.ds_info[ds_name]['n_class']

        metrics = {
            f'{ds_name}/{prefix}/epoch': self.epoch,
            f'{ds_name}/{prefix}/loss': loss,
        }

        # Basic accuracy
        # noinspection PyUnresolvedReferences
        accuracy = (pred_np == label_np).mean()
        metrics[f'{ds_name}/{prefix}/acc'] = float(accuracy)

        # Balanced accuracy
        balanced_acc = balanced_accuracy_score(label_np, pred_np)
        metrics[f'{ds_name}/{prefix}/balanced_acc'] = float(balanced_acc)

        if n_class == 2:
            # Binary classification metrics
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()

            try:
                auroc = roc_auc_score(label_np, probs)
                metrics[f'{ds_name}/{prefix}/auroc'] = float(auroc)
            except ValueError as e:
                logger.warning(f'Error calculating AUROC for {ds_name} {prefix}: {e}')
                metrics[f'{ds_name}/{prefix}/auroc'] = 0.0

            try:
                auc_pr = average_precision_score(label_np, probs)
                metrics[f'{ds_name}/{prefix}/auc_pr'] = float(auc_pr)
            except ValueError as e:
                logger.warning(f'Error calculating AUC-PR for {ds_name} {prefix}: {e}')
                metrics[f'{ds_name}/{prefix}/auc_pr'] = 0.0
        else:
            # Multi-class classification metrics
            cohen_kappa = cohen_kappa_score(label_np, pred_np)
            metrics[f'{ds_name}/{prefix}/cohen_kappa'] = float(cohen_kappa)

            f1_weighted = f1_score(label_np, pred_np, average='weighted')
            metrics[f'{ds_name}/{prefix}/f1'] = float(f1_weighted)

        return metrics

    def collect_dataset_info(self, mixed: bool, ds_name: str = ''):
        """Collect information about datasets for model setup."""
        training_mode = (
            "multitask" if self.multitask else "per dataset"
        )
        logger.debug(
            "Collecting dataset information for %s ...",
            training_mode,
        )

        if mixed:
            self.ds_info = {}
            for dataset_name, dataset_config in self.ds_conf.items():
                self.ds_info[dataset_name] = {
                    'config': dataset_config,
                    'n_class': get_dataset_n_class(dataset_name, dataset_config),
                    'category': get_dataset_category(dataset_name, dataset_config),
                    'shape_info': get_dataset_shape_info(dataset_name, dataset_config, self.cfg.fs),
                }
                logger.debug(
                    "Dataset %s - %s for mixed set",
                    dataset_name,
                    dataset_config,
                )

        else:
            ds_conf = self.ds_conf[ds_name]
            self.ds_info = {
                ds_name: {
                    'config': ds_conf,
                    'n_class': get_dataset_n_class(ds_name, ds_conf),
                    'category': get_dataset_category(ds_name, ds_conf),
                    'shape_info': get_dataset_shape_info(ds_name, ds_conf, self.cfg.fs),
                }}
            logger.debug(f"Dataset {ds_name} - {ds_conf} only")

    def _gather_tensor(self, tensor: Tensor, max_length: int) -> Optional[list[Tensor]]:
        is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        if not is_dist:
            return [tensor]

        exist_mask = torch.tensor([tensor.shape[0]], dtype=torch.int32, device=self.device)
        mask_gather_list = [torch.zeros_like(exist_mask) for _ in range(self.world_size)] \
            if get_is_master() else None
        torch.distributed.gather(exist_mask, gather_list=mask_gather_list, dst=0)

        tensor_pad = torch.zeros([max_length, *(tensor.shape[1:])], dtype=tensor.dtype, device=tensor.device)
        tensor_pad[:tensor.shape[0]] = tensor
        gather_list = [torch.zeros_like(tensor_pad) for _ in range(self.world_size)] \
            if get_is_master() else None
        torch.distributed.gather(tensor_pad, gather_list=gather_list, dst=0)

        if get_is_master():
            for i in range(len(gather_list)):
                gather_list[i] = gather_list[i][:mask_gather_list[i]]

        return gather_list

    def _gather_result(self, logits: Tensor, targets: Tensor) -> tuple[Optional[Tensor], Optional[Tensor]]:
        logits_list = self._gather_tensor(logits, self.micro_batch_size)
        target_list = self._gather_tensor(targets, self.micro_batch_size)

        if get_is_master():
            all_logits = torch.cat(logits_list, dim=0)
            all_target = torch.cat(target_list, dim=0)
            return all_logits.cpu(), all_target.cpu()
        return None, None

    @staticmethod
    def _calc_confusion_matrix(pred: Tensor, target: Tensor, n_class: int) -> Tensor:
        pred, target = pred.long(), target.long()

        linear_indices = target * n_class + pred
        conf_matrix_flat = torch.bincount(linear_indices, minlength=n_class * n_class)
        conf_matrix = conf_matrix_flat.reshape(n_class, n_class)

        return conf_matrix

    def _clip_grad_norm_(self, already_unscaled: bool = False):
        if not already_unscaled:
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.max_grad_norm)
        return grad_norm.detach().cpu().item()

    def create_dataloader(self, split: datasets.NamedSplit = datasets.Split.TRAIN):
        logger.debug("Creating main training dataloader...")
        start_time = time.perf_counter()
        mixed = (split == datasets.Split.TRAIN and self.cfg.multitask)

        dataloaders, samplers = self.dataloader_factory.create_dataloader(
            datasets_config=self.ds_conf,
            mixed=mixed,
            fs=self.cfg.fs,
            num_replicas=self.world_size,
            rank=self.local_rank,
            split=split,
        )
        self._record_loader_build_seconds(
            split,
            time.perf_counter() - start_time,
        )

        return dataloaders, samplers

    def create_single_dataloader(self, ds_name: str, ds_config: str, split: datasets.NamedSplit = datasets.Split.TRAIN):
        logger.debug("Creating single main training dataloader...")
        start_time = time.perf_counter()

        dataloader, sampler = self.dataloader_factory.create_dataloader(
            datasets_config={ds_name: ds_config},
            mixed=False,
            fs=self.cfg.fs,
            num_replicas=self.world_size,
            rank=self.local_rank,
            split=split,
        )

        dataloader = dataloader[0]
        sampler = sampler[0]
        self._record_loader_build_seconds(
            split,
            time.perf_counter() - start_time,
        )

        return dataloader, sampler

    @staticmethod
    def _label_sets_by_dataset(
        dataloaders: Union[list[DataLoader], DataLoader],
    ) -> Dict[str, set[int]]:
        """Read each dataset label space without iterating worker processes."""
        if isinstance(dataloaders, DataLoader):
            dataloaders = [dataloaders]

        label_sets: Dict[str, set[int]] = {}
        for dataloader in dataloaders:
            adapter = dataloader.dataset
            dataset = getattr(adapter, "dataset", adapter)
            if "label" not in dataset.column_names:
                continue

            for montage, label in zip(dataset["montage"], dataset["label"]):
                dataset_name = str(montage).split("/", maxsplit=1)[0]
                label_sets.setdefault(dataset_name, set()).add(int(label))

        return label_sets

    def warn_on_split_label_mismatch(
        self,
        train_loaders: Union[list[DataLoader], DataLoader],
        validation_loaders: Union[list[DataLoader], DataLoader],
        test_loaders: Union[list[DataLoader], DataLoader],
    ) -> None:
        """Warn before training when a dataset split has a different label space."""
        train_labels = self._label_sets_by_dataset(train_loaders)
        validation_labels = self._label_sets_by_dataset(validation_loaders)
        test_labels = self._label_sets_by_dataset(test_loaders)

        for dataset_name in sorted(train_labels | validation_labels | test_labels):
            train_set = sorted(train_labels.get(dataset_name, set()))
            validation_set = sorted(validation_labels.get(dataset_name, set()))
            test_set = sorted(test_labels.get(dataset_name, set()))
            if train_set == validation_set == test_set:
                continue

            warnings.warn(
                "Label-space mismatch before baseline training: "
                f"dataset={dataset_name}; fold=N/A (fixed dataset split); "
                f"training labels={train_set}; validation labels={validation_set}; "
                f"test labels={test_set}. This can cause evaluation predictions "
                "to contain classes absent from a split",
                UserWarning,
                stacklevel=2,
            )

    @abstractmethod
    def setup_model(self):
        """Setup model architecture."""
        pass

    def get_lora_target_modules(self) -> List[str]:
        """
        Get LoRA target modules for this model.

        Can be overridden by subclasses to provide model-specific targets.
        By default, uses the configuration or model-type specific defaults.
        """
        lora_cfg = self.cfg.training.lora

        # If explicit target modules specified (not just ["default"])
        if lora_cfg.lora_target_modules != ["default"]:
            return lora_cfg.lora_target_modules

        # Otherwise, use model-type specific defaults
        return get_model_lora_targets(self.model_type, lora_cfg.lora_target_type)

    def apply_lora(self, model: nn.Module) -> nn.Module:
        """
        Apply LoRA to the model if enabled in configuration.

        Args:
            model: The model to apply LoRA to

        Returns:
            Model with LoRA layers injected (or original model if LoRA disabled)
        """
        lora_cfg = self.cfg.training.lora

        if not lora_cfg.use_lora:
            return model

        logger.debug(
            "Applying LoRA with r=%d, alpha=%d, scope=%s",
            lora_cfg.lora_r,
            lora_cfg.lora_alpha,
            lora_cfg.lora_scope,
        )

        target_modules = self.get_lora_target_modules()
        logger.debug(f"LoRA target modules: {target_modules}")

        model, injected_modules = inject_lora(
            model=model,
            target_modules=target_modules,
            r=lora_cfg.lora_r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            exclude_modules=lora_cfg.lora_exclude_modules,
            scope=lora_cfg.lora_scope,
            verbose=get_is_master(),
        )

        self.lora_modules = injected_modules

        return model

    def setup_optim_params(self, model):
        """
        Setup optimizer parameters with support for LoRA.

        When LoRA is enabled:
        - Only LoRA parameters and classifier/head parameters are trainable
        - Base encoder parameters are frozen

        When LoRA is disabled:
        - Uses original freeze_encoder logic
        """
        lora_cfg = self.cfg.training.lora

        head_params = []
        encoder_params = []
        lora_params = []

        for name, param in model.named_parameters():
            # Check if this is a LoRA parameter
            if "lora_A" in name or "lora_B" in name:
                lora_params.append(param)
            elif 'classifier' in name or 'conv_router' in name:
                head_params.append(param)
            else:
                encoder_params.append(param)

        params = [{'params': head_params, 'lr': self.cfg.training.max_lr}]

        if lora_cfg.use_lora:
            # LoRA mode: train LoRA params + head, freeze encoder
            lora_lr = self.cfg.training.max_lr * lora_cfg.lora_lr_scale
            params.append({'params': lora_params, 'lr': lora_lr})

            # Freeze non-LoRA encoder parameters
            for param in encoder_params:
                param.requires_grad = False

            lora_param_count = sum(p.numel() for p in lora_params)
            head_param_count = sum(p.numel() for p in head_params)
            frozen_param_count = sum(p.numel() for p in encoder_params)

            logger.debug(f"LoRA training mode:")
            logger.debug(
                "  - LoRA params: %s (lr=%.2e)",
                f"{lora_param_count:,}",
                lora_lr,
            )
            logger.debug(
                "  - Head params: %s (lr=%.2e)",
                f"{head_param_count:,}",
                self.cfg.training.max_lr,
            )
            logger.debug(f"  - Frozen encoder params: {frozen_param_count:,}")
        else:
            # Original logic
            if not self.cfg.training.freeze_encoder:
                encoder_lr = self.cfg.training.max_lr * self.cfg.training.encoder_lr_scale
                params.append({'params': encoder_params, 'lr': encoder_lr})
            else:
                # Freeze encoder parameters
                for param in encoder_params:
                    param.requires_grad = False
                logger.debug("Encoder parameters frozen")

        return params

    def setup_optimizer_and_scheduler(self, model, train_loader: DataLoader):
        params = self.setup_optim_params(model)

        optimizer = torch.optim.AdamW(
            params,
            weight_decay=self.cfg.training.weight_decay
        )

        # Gradient scaler for mixed precision
        scaler = torch.amp.GradScaler(enabled=self.cfg.training.use_amp)

        # Learning rate scheduler
        updates_per_epoch = math.ceil(
            len(train_loader) / self.accumulation_steps
        )
        warmup_steps = updates_per_epoch * self.cfg.training.warmup_epochs
        total_steps = updates_per_epoch * self.cfg.training.max_epochs

        if self.cfg.training.lr_schedule == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[p['lr'] for p in params],
                total_steps=total_steps,
                pct_start=self.cfg.training.pct_start
            )
        elif self.cfg.training.lr_schedule == 'cosine':  # warm cosine annealing
            warm_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.cfg.training.warmup_scale,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.cfg.training.min_lr
            )
            scheduler = ChainableSequentialLR(
                optimizer,
                schedulers=[warm_scheduler, cos_scheduler],
                milestones=[warmup_steps]
            )
        else:
            raise NotImplementedError('Unknown learning rate schedule')

        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self._record_fixed_memory_estimate()

    def debug_params_grad(self):
        for name, param in self.model.named_parameters():
            if get_is_master() and param.grad is not None:
                logger.debug(
                    f"{name} "
                    f"Range: [{param.grad.min():.8f}, {param.grad.max():.8f}], "
                    f"Scale: {param.grad.abs().mean():.8f}")

    def get_current_lr(self):
        """Get current learning rates for all parameter groups."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

    # ===========================================
    # Analysis Mode Training Interface
    # ===========================================

    def finetune_one_batch(
        self,
        batch: dict,
        pre_step_hook: Optional[callable] = None,
        post_step_hook: Optional[callable] = None,
    ) -> tuple[float, float, float]:
        """Train on a single batch (used for analysis loops)."""
        self.model.train()
        self.optimizer.zero_grad()

        batch = self._move_batch_to_device(batch)
        labels = batch['label']

        logits, loss = self.train_step(batch, labels)

        if torch.isnan(loss):
            raise ValueError("NaN loss detected during analysis step")

        self.scaler.scale(loss).backward()

        # Unscale grads before analysis hook to avoid scaled gradients
        self.scaler.unscale_(self.optimizer)

        if pre_step_hook is not None:
            pre_step_hook(self.model, self.current_step, batch)

        grad_norm = self._clip_grad_norm_(already_unscaled=True)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == labels).float().mean().item()

        loss_val = loss.detach().item()

        if post_step_hook is not None:
            post_step_hook(self.model, self.current_step, loss_val, grad_norm)

        self.current_step += 1
        self.scheduler.step()

        return loss_val, grad_norm, acc

    def create_masked_batch(
        self,
        batch: dict,
        mask_ratio: float = 0.5,
        mask_strategy: str = "random_mixed",
        temporal_ratio: float = 0.5,
    ) -> Tuple[dict, torch.Tensor, torch.Tensor]:
        """Create masked batch for pretraining objective.

        This creates a masked version of the input data for reconstruction-based
        pretraining. The masking is done on patches (after the data is reshaped
        into patches).

        Args:
            batch: Input batch with 'data' key of shape [B, C, T]
            mask_ratio: Fraction of patches to mask (0.0 - 1.0)
            mask_strategy: Masking strategy:
                - "random": Random patch masking
                - "temporal": Mask entire time steps across all channels
                - "channel": Mask entire channels across all time steps
                - "random_mixed": Mix of temporal and channel masking
            temporal_ratio: For "random_mixed", ratio of temporal vs channel masking

        Returns:
            (masked_batch, mask, original_patches):
                - masked_batch: Batch with masked data
                - mask: Boolean mask [B, C, n_patches] where True = masked
                - original_patches: Original patches [B, C, n_patches, patch_size]
        """
        data = batch['data']  # [B, C, T]
        batch_size, n_channels, n_timepoints = data.shape

        # Infer patch size from model (most models use power of 2)
        patch_size = getattr(self.model, 'patch_size', None)
        if patch_size is None:
            # Try to get from encoder
            encoder = getattr(self.model, 'encoder', None)
            if encoder is not None:
                patch_size = getattr(encoder, 'patch_size', 200)
            else:
                patch_size = 200  # Default

        n_patches = n_timepoints // patch_size

        # Reshape to patches: [B, C, n_patches, patch_size]
        data_trimmed = data[:, :, :n_patches * patch_size]
        original_patches = data_trimmed.view(batch_size, n_channels, n_patches, patch_size)

        # Create mask based on strategy
        device = data.device

        if mask_strategy == "random":
            # Random patch-wise masking
            mask = torch.rand(batch_size, n_channels, n_patches, device=device) < mask_ratio

        elif mask_strategy == "temporal":
            # Mask entire time steps (same mask across channels)
            temporal_mask = torch.rand(batch_size, 1, n_patches, device=device) < mask_ratio
            mask = temporal_mask.expand(-1, n_channels, -1)

        elif mask_strategy == "channel":
            # Mask entire channels (same mask across time)
            channel_mask = torch.rand(batch_size, n_channels, 1, device=device) < mask_ratio
            mask = channel_mask.expand(-1, -1, n_patches)

        elif mask_strategy == "random_mixed":
            # Mix of temporal and channel masking
            n_temporal = int(mask_ratio * temporal_ratio * n_patches)
            n_channel = int(mask_ratio * (1 - temporal_ratio) * n_channels)

            mask = torch.zeros(batch_size, n_channels, n_patches, dtype=torch.bool, device=device)

            for b in range(batch_size):
                # Temporal masking (random time steps)
                if n_temporal > 0:
                    t_indices = torch.randperm(n_patches, device=device)[:n_temporal]
                    mask[b, :, t_indices] = True

                # Channel masking (random channels)
                if n_channel > 0:
                    c_indices = torch.randperm(n_channels, device=device)[:n_channel]
                    mask[b, c_indices, :] = True
        else:
            raise ValueError(f"Unknown mask strategy: {mask_strategy}")

        # Apply mask (zero out masked patches)
        masked_patches = original_patches.clone()
        mask_expanded = mask.unsqueeze(-1).expand_as(masked_patches)
        masked_patches[mask_expanded] = 0.0

        # Reshape back to [B, C, T]
        masked_data = masked_patches.view(batch_size, n_channels, n_patches * patch_size)

        # Create masked batch
        masked_batch = batch.copy()
        masked_batch['data'] = masked_data

        return masked_batch, mask, original_patches

    def pretrain_step_for_analysis(
        self,
        batch: dict,
        mask_ratio: float = 0.5,
        mask_strategy: str = "random_mixed",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pretrain step with MSE reconstruction objective.

        This performs a single pretraining step:
        1. Mask input patches
        2. Forward through encoder
        3. Reconstruct masked patches (using simple linear head)
        4. Compute MSE loss on masked positions only

        Args:
            batch: Input batch
            mask_ratio: Fraction of patches to mask
            mask_strategy: Masking strategy

        Returns:
            (loss, logits, mask): Reconstruction loss, predicted patches, mask
        """
        # Create masked batch
        masked_batch, mask, original_patches = self.create_masked_batch(
            batch, mask_ratio, mask_strategy
        )

        with torch.amp.autocast('cuda', enabled=self.cfg.training.use_amp, dtype=torch.bfloat16):
            # Get encoder output
            encoder = getattr(self.model, 'encoder', self.model)

            # Forward through encoder
            # Most encoders expect [B, C, n_patches, patch_size]
            data = masked_batch['data']
            batch_size, n_channels, n_timepoints = data.shape

            patch_size = getattr(encoder, 'patch_size', 200)
            n_patches = n_timepoints // patch_size

            # Reshape for encoder
            data_patches = data[:, :, :n_patches * patch_size].view(
                batch_size, n_channels, n_patches, patch_size
            )

            # Get features from encoder
            # Output shape varies by model, typically [B, C, n_patches, D] or [B, T, D]
            features = encoder(data_patches)

            # Handle different output shapes
            if features.dim() == 3:
                # [B, T, D] - typical transformer output
                # Reshape to [B, C, n_patches, D] if possible
                if features.shape[1] == n_channels * n_patches:
                    features = features.view(batch_size, n_channels, n_patches, -1)
                else:
                    # Use as-is, project to reconstruction
                    embed_dim = features.shape[-1]
                    if self._pretrain_recon_head is None:
                        head = torch.nn.Linear(embed_dim, patch_size)
                        head = head.to(features.device).to(features.dtype)
                        target_model = getattr(self.model, "module", self.model)
                        target_model._pretrain_recon_head = head
                        self._pretrain_recon_head = head
                        if self.optimizer is not None:
                            self.optimizer.add_param_group({
                                "params": self._pretrain_recon_head.parameters(),
                                "lr": self.cfg.training.max_lr,
                            })
                    reconstructed = self._pretrain_recon_head(features)
                    # This path needs special handling - skip for now
                    raise NotImplementedError("3D output reconstruction not fully implemented")

            if features.dim() == 4:
                # [B, C, n_patches, D]
                embed_dim = features.shape[-1]

                # Create reconstruction head if not exists (register on model)
                if self._pretrain_recon_head is None:
                    head = torch.nn.Linear(embed_dim, patch_size)
                    head = head.to(features.device).to(features.dtype)
                    # Register on underlying model for checkpointing
                    target_model = getattr(self.model, "module", self.model)
                    target_model._pretrain_recon_head = head
                    self._pretrain_recon_head = head
                    # Ensure optimizer updates this head if optimizer already built
                    if self.optimizer is not None:
                        self.optimizer.add_param_group({
                            "params": self._pretrain_recon_head.parameters(),
                            "lr": self.cfg.training.max_lr,
                        })

                # Reconstruct: [B, C, n_patches, patch_size]
                reconstructed = self._pretrain_recon_head(features)
            else:
                raise ValueError(f"Unexpected feature shape: {features.shape}")

        # Compute MSE loss on masked positions only
        # mask: [B, C, n_patches], original_patches: [B, C, n_patches, patch_size]
        mask_expanded = mask.unsqueeze(-1).expand_as(original_patches)

        # Only compute loss on masked patches
        pred_masked = reconstructed[mask_expanded]
        target_masked = original_patches[mask_expanded]

        if pred_masked.numel() == 0:
            # No masked patches (edge case)
            loss = torch.tensor(0.0, device=reconstructed.device, requires_grad=True)
        else:
            loss = torch.nn.functional.mse_loss(pred_masked.float(), target_masked.float())

        return loss, reconstructed, mask

    def pretrain_one_batch_for_analysis(
        self,
        batch: dict,
        mask_ratio: float = 0.5,
        mask_strategy: str = "random_mixed",
        pre_step_hook: Optional[callable] = None,
    ) -> Tuple[float, float]:
        """Pretrain on a single batch (used for analysis loops).

        Args:
            batch: Input batch
            mask_ratio: Fraction of patches to mask
            mask_strategy: Masking strategy
            pre_step_hook: Callable(model, step, batch) called before optimizer.step()

        Returns:
            (loss, grad_norm): Loss value and gradient norm
        """
        self.model.train()
        self.optimizer.zero_grad()

        batch = self._move_batch_to_device(batch)

        loss, reconstructed, mask = self.pretrain_step_for_analysis(batch, mask_ratio, mask_strategy)

        if torch.isnan(loss):
            raise ValueError("NaN loss detected during pretrain step")

        self.scaler.scale(loss).backward()

        # Unscale grads before analysis hook
        self.scaler.unscale_(self.optimizer)

        if pre_step_hook is not None:
            pre_step_hook(self.model, self.current_step, batch)

        grad_norm = self._clip_grad_norm_(already_unscaled=True)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        loss_val = loss.detach().item()

        self.current_step += 1
        self.scheduler.step()

        return loss_val, grad_norm

    def setup_analysis_mode(self):
        """Configure trainer for gradient/feature analysis mode.

        This sets up the trainer in a special mode optimized for analysis:
        1. Optionally disables cloud logging (wandb/comet)
        2. Sets up analysis-specific output directory
        3. Returns hooks for gradient capture

        """
        self.cfg.logging.use_cloud = False
        logger.debug("Analysis mode: cloud logging disabled")

    # ===========================================
    # Fine-tuning Training Interface
    # ===========================================

    def train_step(self, batch, labels):
        with torch.amp.autocast('cuda', enabled=self.cfg.training.use_amp, dtype=torch.bfloat16):
            logits = self.model(batch)

        loss = self.loss_fn(logits, labels)
        return logits, loss

    def _finish_accumulated_update(self, local_samples: int) -> float:
        """Normalize accumulated gradients and perform one optimizer update."""
        sample_tensor = torch.tensor(
            float(local_samples),
            dtype=torch.float64,
            device=self.device,
        )
        is_distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if is_distributed:
            torch.distributed.all_reduce(
                sample_tensor,
                op=torch.distributed.ReduceOp.SUM,
            )
        global_samples = float(sample_tensor.item())
        if global_samples <= 0:
            raise ValueError(
                "Expected a positive accumulated sample count, but got "
                f"{global_samples}."
            )
        gradient_denominator = global_samples / self.world_size
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(gradient_denominator)
        self.scaler.unscale_(self.optimizer)
        grad_norm = self._clip_grad_norm_(already_unscaled=True)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return grad_norm

    def _log_accumulated_update(
        self,
        loss_sum: Tensor,
        correct_sum: Tensor,
        local_samples: int,
        grad_norm: float,
        dataset_name: str,
    ) -> None:
        """Log sample-weighted statistics for one optimizer update."""
        if self.current_step % self.cfg.logging.log_step_interval != 0:
            return
        sample_tensor = torch.tensor(
            float(local_samples),
            dtype=torch.float64,
            device=self.device,
        )
        is_distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if is_distributed:
            for tensor in (loss_sum, correct_sum, sample_tensor):
                torch.distributed.all_reduce(
                    tensor,
                    op=torch.distributed.ReduceOp.SUM,
                )
        sample_count = float(sample_tensor.item())
        if sample_count <= 0:
            raise ValueError("Cannot log an empty optimizer update.")
        if not get_is_master():
            return
        log_data = {
            "train/epoch": self.epoch,
            "train/step": self.current_step,
            "train/loss_ce": float(loss_sum.item() / sample_count),
            "train/acc": float(correct_sum.item() / sample_count),
            "train/grad_norm": grad_norm,
            "train/header_lr": self.get_current_lr()[0],
        }
        if not self.cfg.training.freeze_encoder:
            log_data["train/encoder_lr"] = self.get_current_lr()[-1]
        if not self.multitask:
            log_data = {
                f"{dataset_name}/{key}": value
                for key, value in log_data.items()
            }
        if self.cfg.logging.use_cloud:
            self._log_to_cloud(log_data)

        # Suppress per-step logging in reduced HPO mode
        if not (self.run_mode == "hpo" and self.hpo_logging_mode == "reduced"):
            self._log_to_tensorboard(log_data, self.current_step)
            self._write_csv_metrics(log_data, self.current_step)

        logger.debug(format_console_log_dict(log_data, prefix="train"))

    def _run_accumulated_epoch(
        self,
        train_loader: DataLoader,
        train_sampler: DistributedGroupBatchSampler,
        update_hook: Callable[[], None],
    ) -> None:
        """Train an epoch with sample-normalized gradient accumulation."""
        train_sampler.set_epoch(self.epoch)
        self.optimizer.zero_grad()
        local_samples = 0
        loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        correct_sum = torch.zeros(
            (),
            dtype=torch.float64,
            device=self.device,
        )
        dataset_name = "multitask"
        total_micro_batches = len(train_loader)
        for micro_step, batch in enumerate(train_loader):
            batch = self._move_batch_to_device(batch)
            labels = batch["label"]
            dataset_name = batch["montage"][0].split("/")[0]
            sample_count = int(labels.shape[0])
            if sample_count <= 0:
                raise ValueError("Training produced an empty micro-batch.")
            update_boundary = (
                (micro_step + 1) % self.accumulation_steps == 0
                or micro_step + 1 == total_micro_batches
            )
            sync_context = nullcontext()
            if not update_boundary and hasattr(self.model, "no_sync"):
                sync_context = self.model.no_sync()
            with sync_context:
                logits, loss = self.train_step(batch, labels)
                if not torch.isfinite(loss):
                    raise ValueError(
                        "Training loss is not finite at optimizer step "
                        f"{self.current_step}: {loss.detach().item()}."
                    )
                self.scaler.scale(loss * sample_count).backward()
            predictions = torch.argmax(logits.detach(), dim=-1)
            loss_sum += loss.detach().double() * sample_count
            correct_sum += (
                predictions == labels
            ).sum().detach().double()
            local_samples += sample_count
            if not update_boundary:
                continue
            grad_norm = self._finish_accumulated_update(local_samples)
            self._log_accumulated_update(
                loss_sum,
                correct_sum,
                local_samples,
                grad_norm,
                dataset_name,
            )
            self.current_step += 1
            update_hook()
            self.optimizer.zero_grad()
            local_samples = 0
            loss_sum.zero_()
            correct_sum.zero_()

    def train_epoch(
        self,
        train_loader: DataLoader,
        train_sampler: DistributedGroupBatchSampler,
    ) -> None:
        """Train one epoch using the configured adaptive micro-batch."""
        self.model.train()
        if self.cfg.training.freeze_encoder:
            encoder = (
                self.model.module.encoder
                if hasattr(self.model, "module")
                else self.model.encoder
            )
            encoder.eval()
        self._run_accumulated_epoch(
            train_loader,
            train_sampler,
            self.scheduler.step,
        )

    def eval_step(self, batch, labels):
        with torch.amp.autocast('cuda', enabled=self.cfg.training.use_amp, dtype=torch.bfloat16):
            logits = self.model(batch)

        loss = self.loss_fn(logits, labels)
        return logits, loss

    def eval_epoch(self, dataloaders: list[DataLoader], prefix: str):
        """Evaluate one epoch and return metrics."""
        start_time = time.perf_counter()
        is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        if get_is_master():
            logger.debug("Starting %s evaluation.", prefix)

        self.model.eval()

        overall_metrics = {}
        metric_results: Dict[str, Dict[str, float]] = {}
        self.latest_eval_counts = {}
        for ds_name in self.ds_info.keys():
            n_class = self.ds_info[ds_name]['n_class']
            overall_metrics[ds_name] = {
                'loss_sum': torch.zeros([1], dtype=torch.float64, device=self.device),
                'cm': torch.zeros((n_class, n_class), dtype=torch.int64, device=self.device),
                'cnt': torch.zeros(1, dtype=torch.int64, device=self.device),
                'logits': [],
                'labels': [],
            }

        with torch.no_grad():
            for dataloader in dataloaders:
                for batch in dataloader:
                    batch = self._move_batch_to_device(batch)
                    labels = batch['label']
                    ds_name = batch['montage'][0].split('/')[0]
                    n_class = self.ds_info[ds_name]['n_class']

                    # Forward pass with mixed precision
                    logits, loss = self.train_step(batch, labels)

                    logits = logits.float()
                    pred = torch.argmax(logits, dim=1).detach()
                    cm = self._calc_confusion_matrix(pred, labels.detach(), n_class)

                    batch_size = labels.shape[0]
                    overall_metrics[ds_name]['loss_sum'] += loss.detach() * batch_size
                    overall_metrics[ds_name]['cnt'] += batch_size
                    overall_metrics[ds_name]['cm'] += cm.detach()

                    logits_across, labels_across = self._gather_result(logits.detach(), labels.detach())
                    if get_is_master():
                        overall_metrics[ds_name]['logits'].append(logits_across.cpu())
                        overall_metrics[ds_name]['labels'].append(labels_across.cpu())

                if is_dist:
                    torch.distributed.barrier()

            log_dict = {}
            for ds_name in self.ds_info.keys():
                if is_dist:
                    torch.distributed.all_reduce(overall_metrics[ds_name]['loss_sum'], op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(overall_metrics[ds_name]['cnt'], op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(overall_metrics[ds_name]['cm'], op=torch.distributed.ReduceOp.SUM)

                count = int(overall_metrics[ds_name]['cnt'].item())
                if count <= 0:
                    raise ValueError(
                        f"Expected evaluation examples for {ds_name}, but "
                        f"got {count}."
                    )
                self.latest_eval_counts[ds_name] = count
                overall_metrics[ds_name]['loss'] = overall_metrics[ds_name]['loss_sum'] / overall_metrics[ds_name][
                    'cnt'].float()

                # Calculate metrics on aggregated data (only master process in distributed mode)
                if get_is_master():
                    labels_all = torch.concat(overall_metrics[ds_name]['labels'], dim=0)
                    logits_all = torch.concat(overall_metrics[ds_name]['logits'], dim=0)
                    loss_metric = overall_metrics[ds_name]['loss'].detach().cpu().item()
                    metrics = self._calculate_metrics_for_dataset(
                        labels=labels_all,
                        logits=logits_all,
                        ds_name=ds_name,
                        prefix=prefix,
                        loss=loss_metric
                    )

                    log_dict = log_dict | metrics
                    metric_results[ds_name] = metrics
                    log_console = format_console_log_dict(metrics, prefix=f"{ds_name}/{prefix}")
                    logger.debug(log_console)

            if get_is_master() and self.cfg.logging.use_cloud:
                log_cloud = self._create_ft_cloud_log_data(log_dict, prefix, overall_metrics)
                self._log_to_cloud(log_cloud)
            if get_is_master():
                if self.multitask:
                    for ds_name, metrics in metric_results.items():
                        self._open_dataset_tensorboard(ds_name)
                        self._log_to_tensorboard(
                            metrics,
                            self.current_step,
                            ds_name=ds_name,
                        )
                        self._write_csv_metrics(metrics, self.current_step)
                else:
                    self._log_to_tensorboard(log_dict, self.current_step)
                    self._write_csv_metrics(log_dict, self.current_step)

            if is_dist:
                torch.distributed.barrier()
            self._record_evaluation_seconds(
                prefix,
                time.perf_counter() - start_time,
            )

            return metric_results

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        pass

    def load_lora_checkpoint(self, lora_checkpoint_path: str):
        """
        Load LoRA weights from a checkpoint file.

        Args:
            lora_checkpoint_path: Path to the LoRA checkpoint file
        """
        if not self.cfg.training.lora.use_lora:
            logger.warning("LoRA is not enabled, skipping LoRA checkpoint loading")
            return

        logger.debug(f"Loading LoRA checkpoint from {lora_checkpoint_path}")
        lora_state_dict = torch.load(lora_checkpoint_path, map_location=self.device, weights_only=True)

        missing_keys, unexpected_keys = load_lora_state_dict(
            self.model, lora_state_dict, strict=False
        )

        if missing_keys:
            logger.warning(f"Missing LoRA keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected LoRA keys: {unexpected_keys}")

        logger.debug("LoRA checkpoint loaded successfully")

    def save_checkpoint(self, ds_name: Optional[str] = None, is_milestone: bool = False, **kwargs):
        """Save checkpoint with unified path management."""
        if not get_is_master():
            return

        if ds_name is None:
            ds_name = 'unified'
            checkpoint_dir = Path(self.ckpt_dir, ds_name)
        else:
            checkpoint_dir = Path(self.ckpt_dir, 'seperated', ds_name)

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': self.epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.cfg.model_dump(mode='json'),
            'dataset_name': ds_name,
        }

        # Save checkpoint
        suffix = kwargs.get('suffix')
        if suffix is None:
            suffix = (
                'last' if is_milestone else f'epoch_{self.epoch}'
            )
        checkpoint_path = checkpoint_dir / f'{self.model_type}_{ds_name}_{suffix}.pt'
        torch.save(checkpoint, checkpoint_path)

        logger.debug("Checkpoint saved for %s: %s", ds_name, checkpoint_path)

        # Save LoRA weights separately if LoRA is enabled
        if self.cfg.training.lora.use_lora:
            self.save_lora_checkpoint(checkpoint_dir, ds_name, suffix)
        return checkpoint_path

    def save_lora_checkpoint(self, checkpoint_dir: Path, ds_name: str, suffix: str):
        """
        Save LoRA weights separately from the main checkpoint.

        Args:
            checkpoint_dir: Directory to save the checkpoint
            ds_name: Dataset name
            suffix: Checkpoint suffix (e.g., 'last', 'epoch_10')
        """
        if not get_is_master():
            return

        lora_state_dict = get_lora_state_dict(self.model)

        if not lora_state_dict:
            logger.warning("No LoRA parameters found to save")
            return

        lora_checkpoint_path = checkpoint_dir / f'{self.model_type}_{ds_name}_{suffix}_lora.pt'
        torch.save(lora_state_dict, lora_checkpoint_path)

        lora_param_count = sum(v.numel() for v in lora_state_dict.values())
        logger.debug(
            "LoRA checkpoint saved: %s (%s params)",
            lora_checkpoint_path,
            f"{lora_param_count:,}",
        )


    def _training_checkpoint_path(
        self,
        ds_name: Optional[str],
        suffix: str,
    ) -> Path:
        """Return a repository-generated fine-tuning checkpoint path."""
        checkpoint_name = "unified" if ds_name is None else ds_name
        if ds_name is None:
            checkpoint_dir = Path(self.ckpt_dir, checkpoint_name)
        else:
            checkpoint_dir = Path(
                self.ckpt_dir,
                "seperated",
                checkpoint_name,
            )
        filename = (
            f"{self.model_type}_{checkpoint_name}_{suffix}.pt"
        )
        return checkpoint_dir / filename

    def load_training_checkpoint(self, checkpoint_path: Path) -> None:
        """Load a checkpoint produced by save_checkpoint into this model."""
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Training checkpoint does not exist: "
                f"{checkpoint_path.resolve()}."
            )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(
                f"Checkpoint at {checkpoint_path.resolve()} has no "
                "model_state_dict mapping."
            )
        self.model.load_state_dict(state_dict)
        epoch = checkpoint.get("epoch")
        if isinstance(epoch, int) and not isinstance(epoch, bool):
            if epoch < 0:
                raise ValueError(
                    "Training checkpoint epoch cannot be negative."
                )
            self.epoch = epoch

    def _broadcast_bool(self, value: bool) -> bool:
        """Broadcast one master-process decision to every training rank."""
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            return value
        tensor = torch.tensor(
            [int(value) if get_is_master() else 0],
            dtype=torch.uint8,
            device=self.device,
        )
        torch.distributed.broadcast(tensor, src=0)
        return bool(tensor.item())

    @staticmethod
    def _training_sizes(
        train_loader: DataLoader,
    ) -> Dict[str, int]:
        """Count unsharded training examples by dataset."""
        adapter = train_loader.dataset
        dataset = getattr(adapter, "dataset", adapter)
        if "montage" not in dataset.column_names:
            raise ValueError(
                "Training dataset has no montage column for dataset counts."
            )
        sizes: Dict[str, int] = {}
        for montage in dataset["montage"]:
            dataset_name = str(montage).split("/", maxsplit=1)[0]
            sizes[dataset_name] = sizes.get(dataset_name, 0) + 1
        if not sizes:
            raise ValueError("Training dataset contains no examples.")
        return sizes

    def _validation_score(
        self,
        metrics_by_dataset: Dict[str, Dict[str, float]],
    ) -> float:
        """Return the macro validation score used for final early stopping."""
        metric_name = self.cfg.training.early_stopping.metric
        values: List[float] = []
        for dataset_name, metrics in metrics_by_dataset.items():
            key = f"{dataset_name}/eval/{metric_name}"
            if key not in metrics:
                raise ValueError(
                    f"Expected early-stopping metric '{key}', but it was "
                    "not emitted."
                )
            value = float(metrics[key])
            if not torch.isfinite(torch.tensor(value)):
                raise ValueError(
                    f"Early-stopping metric '{key}' is not finite: {value}."
                )
            values.append(value)
        if not values:
            raise ValueError("Validation produced no early-stopping metrics.")
        return sum(values) / len(values)

    def _is_improvement(
        self,
        score: float,
        best_score: Optional[float],
    ) -> bool:
        """Return whether score improves by the configured minimum delta."""
        if best_score is None:
            return True
        args = self.cfg.training.early_stopping
        if args.direction == "minimize":
            return score < best_score - args.min_delta
        return score > best_score + args.min_delta

    def _callback_stop_reason(
        self,
        metrics: Dict[str, Dict[str, float]],
        train_sizes: Dict[str, int],
    ) -> HpoStopReason:
        """Run the HPO callback and synchronize its explicit stop reason."""
        payload: Optional[Dict[str, Any]] = None
        if get_is_master():
            try:
                reason = HpoStopReason.NONE
                if self.validation_callback is not None:
                    response = self.validation_callback(
                        self.epoch,
                        metrics,
                        train_sizes,
                    )
                    if isinstance(response, bool):
                        reason = (
                            HpoStopReason.OPTUNA_PRUNED
                            if response
                            else HpoStopReason.NONE
                        )
                    else:
                        reason = HpoStopReason(response)
                payload = {
                    "stop_reason": reason.value,
                    "error": None,
                }
            except Exception as exc:
                payload = {
                    "stop_reason": HpoStopReason.NONE.value,
                    "error": (
                        f"{type(exc).__module__}."
                        f"{type(exc).__name__}: {exc}"
                    ),
                }

        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            objects = [payload]
            torch.distributed.broadcast_object_list(objects, src=0)
            payload = objects[0]
        if payload is None:
            raise RuntimeError(
                "Validation callback did not produce a synchronized result."
            )
        if payload["error"] is not None:
            raise ValueError(
                "Validation callback failed: "
                f"{payload['error']}."
            )
        return HpoStopReason(str(payload["stop_reason"]))

    def _managed_epoch_loop(
        self,
        train_loader: DataLoader,
        train_sampler: DistributedGroupBatchSampler,
        valid_loaders: list[DataLoader],
        checkpoint_dataset: Optional[str],
    ) -> tuple[
        Dict[str, Dict[str, float]],
        Optional[Path],
        HpoStopReason,
    ]:
        """Train one model and return best validation state."""
        train_sizes = self._training_sizes(train_loader)
        best_score: Optional[float] = None
        best_metrics: Dict[str, Dict[str, float]] = {}
        best_checkpoint: Optional[Path] = None
        epochs_without_improvement = 0
        stop_reason = HpoStopReason.NONE

        for epoch in range(self.cfg.training.max_epochs):
            self.epoch = epoch
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                torch.distributed.barrier()

            self.train_epoch(train_loader, train_sampler)
            validation_metrics = self.eval_epoch(valid_loaders, "eval")
            if get_is_master() and not validation_metrics:
                raise RuntimeError(
                    "Validation returned no metrics on the master process."
                )

            if self.run_mode == "hpo":
                stop_reason = self._callback_stop_reason(
                    validation_metrics,
                    train_sizes,
                )
                if stop_reason is not HpoStopReason.NONE:
                    break
                best_metrics = validation_metrics
                continue

            score = (
                self._validation_score(validation_metrics)
                if get_is_master()
                else 0.0
            )
            improved = self._broadcast_bool(
                get_is_master()
                and self._is_improvement(score, best_score)
            )
            if improved:
                if get_is_master():
                    best_score = score
                    best_metrics = validation_metrics
                    self.save_checkpoint(
                        ds_name=checkpoint_dataset,
                        suffix="best",
                    )
                best_checkpoint = self._training_checkpoint_path(
                    checkpoint_dataset,
                    "best",
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (
                (epoch + 1) % self.cfg.logging.ckpt_interval == 0
                and self.run_mode == "legacy"
            ):
                self.save_checkpoint(ds_name=checkpoint_dataset)

            stopping = self.cfg.training.early_stopping
            should_stop = (
                stopping.enabled
                and epochs_without_improvement >= stopping.patience
            )
            if self._broadcast_bool(should_stop):
                logger.debug(
                    "Early stopping at epoch %d after %d epochs without "
                    "improvement.",
                    epoch,
                    epochs_without_improvement,
                )
                break

        if self.run_mode != "hpo":
            if best_checkpoint is None:
                raise RuntimeError(
                    "Training finished without a validation-best checkpoint."
                )
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                torch.distributed.barrier()
            self.load_training_checkpoint(best_checkpoint)
        return best_metrics, best_checkpoint, stop_reason

    def run(self) -> Dict[str, Any]:
        """Execute one seed-scoped neural training run."""
        seed_torch(self.cfg.seed)
        self.setup_distributed()
        self._ensure_runtime_batching()
        self._configure_cuda_memory_limit()
        self.setup_logging()
        self.init_tensorboard_logging()
        self.init_cloud_logging()

        logger.info(
            "Starting %s training for seed %d with %d dataset(s): %s",
            self.cfg.model_type,
            self.cfg.seed,
            self.num_ds,
            list(self.cfg.data.datasets),
        )

        try:
            if self.cfg.multitask:
                result = self.run_unified_training()
            else:
                result = self.run_separate_training()
            result["performance"] = self.get_performance_diagnostics()
            self.training_result = result
            return result
        finally:
            self._write_adaptive_batch_profile()
            self._close_csv_writer()
            self.finish_cloud_logging()
            self.finish_tensorboard_logging()
            if not self.external_distributed:
                clean_torch_distributed(self.local_rank)

    def recover_multitask_datasets(
        self,
        dataset_names: Iterable[str],
        checkpoint_path: Path,
    ) -> Dict[str, Any]:
        """Evaluate missing multitask datasets from one retained best state.

        Parameters
        ----------
        dataset_names : Iterable[str]
            Missing datasets whose existing compatible peers stay untouched.
        checkpoint_path : pathlib.Path
            Retained shared validation-best training checkpoint.

        Returns
        -------
        dict[str, Any]
            Recovered validation/test metrics and cleanup diagnostics.
        """
        missing = list(dataset_names)
        if not self.multitask or self.run_mode != "recovery":
            raise ValueError(
                "Shared evaluation recovery requires multitask mode and "
                "run_mode='recovery'."
            )
        unknown = [name for name in missing if name not in self.ds_conf]
        if not missing or unknown:
            raise ValueError(
                "Expected known missing multitask datasets, but got "
                f"{missing}; unknown={unknown}."
            )

        seed_torch(self.cfg.seed)
        self.setup_distributed()
        self._ensure_runtime_batching()
        self._configure_cuda_memory_limit()
        self.setup_logging()
        self.init_tensorboard_logging()
        self.init_cloud_logging()
        try:
            self._open_csv_writer("training", append=True)
            self.collect_dataset_info(mixed=True)
            self.setup_model()
            self.load_training_checkpoint(checkpoint_path)
            validation_loaders: list[DataLoader] = []
            test_loaders: list[DataLoader] = []
            for dataset_name in missing:
                dataset_config = self.ds_conf[dataset_name]
                validation_loader, _ = self.create_single_dataloader(
                    dataset_name,
                    dataset_config,
                    datasets.Split.VALIDATION,
                )
                test_loader, _ = self.create_single_dataloader(
                    dataset_name,
                    dataset_config,
                    datasets.Split.TEST,
                )
                if not isinstance(validation_loader, DataLoader):
                    raise TypeError(
                        "Recovery validation loader must be a DataLoader."
                    )
                if not isinstance(test_loader, DataLoader):
                    raise TypeError(
                        "Recovery test loader must be a DataLoader."
                    )
                validation_loaders.append(validation_loader)
                test_loaders.append(test_loader)
            validation_metrics = self.eval_epoch(
                validation_loaders,
                "eval",
            )
            test_metrics = self.eval_epoch(test_loaders, "test")
            if get_is_master():
                for dataset_name in missing:
                    dataset_config = self.ds_conf[dataset_name]
                    if (
                        dataset_name not in validation_metrics
                        or dataset_name not in test_metrics
                    ):
                        raise RuntimeError(
                            "Evaluation recovery returned no metrics for "
                            f"dataset '{dataset_name}'."
                        )
                    self.final_validation_metrics[dataset_name] = (
                        validation_metrics[dataset_name]
                    )
                    self.final_test_metrics[dataset_name] = (
                        test_metrics[dataset_name]
                    )
                    self.final_checkpoint_paths[dataset_name] = (
                        checkpoint_path
                    )
                    self._write_completion(
                        dataset_name,
                        dataset_config,
                    )
                self._cleanup_checkpoint_artifacts(
                    None,
                    checkpoint_path,
                )
            return {
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "checkpoint_cleanup_failures": list(
                    self.checkpoint_cleanup_failures
                ),
            }
        finally:
            self._write_adaptive_batch_profile()
            self._close_csv_writer()
            self.finish_cloud_logging()
            self.finish_tensorboard_logging()
            if not self.external_distributed:
                clean_torch_distributed(self.local_rank)

    def run_unified_training(self) -> Dict[str, Any]:
        """Train one shared model for all configured datasets."""
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.barrier()

        if self.run_mode != "hpo":
            self._reset_unified_outputs()
        self._open_csv_writer("training")
        self.collect_dataset_info(mixed=True)
        model = self.setup_model()

        train_loader, train_sampler = self.create_dataloader(
            datasets.Split.TRAIN
        )
        valid_loaders, _ = self.create_dataloader(
            datasets.Split.VALIDATION
        )
        test_loaders = None
        if self.run_mode != "hpo":
            test_loaders, _ = self.create_dataloader(datasets.Split.TEST)
            self.warn_on_split_label_mismatch(
                train_loader,
                valid_loaders,
                test_loaders,
            )

        if not isinstance(train_loader, DataLoader):
            raise TypeError("train_loader must be a DataLoader.")
        if not isinstance(
            train_sampler,
            DistributedGroupBatchSampler,
        ):
            raise TypeError(
                "train_sampler must be a DistributedGroupBatchSampler."
            )

        self.setup_optimizer_and_scheduler(model, train_loader)
        validation_metrics, checkpoint_path, stop_reason = (
            self._managed_epoch_loop(
                train_loader,
                train_sampler,
                valid_loaders,
                None,
            )
        )
        result: Dict[str, Any] = {
            "validation_metrics": validation_metrics,
            "hpo_stop_reason": stop_reason.value,
            "pruned": stop_reason is HpoStopReason.OPTUNA_PRUNED,
        }
        if self.run_mode == "hpo":
            return result

        if test_loaders is None or checkpoint_path is None:
            raise RuntimeError("Final evaluation loaders were not created.")
        test_metrics = self.eval_epoch(test_loaders, "test")
        if get_is_master():
            for dataset_name, dataset_config in self.ds_conf.items():
                self.final_validation_metrics[dataset_name] = (
                    validation_metrics[dataset_name]
                )
                self.final_test_metrics[dataset_name] = (
                    test_metrics[dataset_name]
                )
                self.final_checkpoint_paths[dataset_name] = checkpoint_path
                self._write_completion(dataset_name, dataset_config)
            self._cleanup_checkpoint_artifacts(
                None, checkpoint_path
            )
        result.update({
            "test_metrics": test_metrics,
            "checkpoint_path": (
                str(checkpoint_path.resolve())
                if self.cfg.logging.save_checkpoints
                else None
            ),
            "checkpoint_cleanup_failures": list(
                self.checkpoint_cleanup_failures
            ),
        })
        return result

    def run_separate_training(self) -> Dict[str, Any]:
        """Train one independently configured model per dataset."""
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.barrier()

        all_validation: Dict[str, Dict[str, float]] = {}
        all_test: Dict[str, Dict[str, float]] = {}
        hpo_stop_reason = HpoStopReason.NONE
        for dataset_name, dataset_config in self.ds_conf.items():
            if (
                self.run_mode != "hpo"
                and self._dataset_is_complete(
                    dataset_name,
                    dataset_config,
                )
            ):
                logger.info(
                    "Skipping completed dataset: %s",
                    dataset_name,
                )
                continue

            if get_is_master():
                self._reset_dataset_outputs(dataset_name)
                self._open_csv_writer(dataset_name)
                self._open_dataset_tensorboard(dataset_name)
            self.current_dataset = dataset_name
            self.collect_dataset_info(
                mixed=False,
                ds_name=dataset_name,
            )
            model = self.setup_model()

            train_loader, train_sampler = self.create_single_dataloader(
                dataset_name,
                dataset_config,
                datasets.Split.TRAIN,
            )
            valid_loader, _ = self.create_single_dataloader(
                dataset_name,
                dataset_config,
                datasets.Split.VALIDATION,
            )
            test_loader = None
            if self.run_mode != "hpo":
                test_loader, _ = self.create_single_dataloader(
                    dataset_name,
                    dataset_config,
                    datasets.Split.TEST,
                )
                self.warn_on_split_label_mismatch(
                    train_loader,
                    valid_loader,
                    test_loader,
                )

            if not isinstance(train_loader, DataLoader):
                raise TypeError("train_loader must be a DataLoader.")
            if not isinstance(
                train_sampler,
                DistributedGroupBatchSampler,
            ):
                raise TypeError(
                    "train_sampler must be a "
                    "DistributedGroupBatchSampler."
                )
            if not isinstance(valid_loader, DataLoader):
                raise TypeError("valid_loader must be a DataLoader.")

            self.setup_optimizer_and_scheduler(model, train_loader)
            validation_metrics, checkpoint_path, dataset_stop_reason = (
                self._managed_epoch_loop(
                    train_loader,
                    train_sampler,
                    [valid_loader],
                    dataset_name,
                )
            )
            all_validation.update(validation_metrics)
            hpo_stop_reason = dataset_stop_reason
            if self.run_mode == "hpo":
                if dataset_stop_reason is not HpoStopReason.NONE:
                    break
                continue

            if test_loader is None or checkpoint_path is None:
                raise RuntimeError(
                    f"Final evaluation loader for {dataset_name} was "
                    "not created."
                )
            test_metrics = self.eval_epoch([test_loader], "test")
            all_test.update(test_metrics)
            if get_is_master():
                self.final_validation_metrics[dataset_name] = (
                    validation_metrics[dataset_name]
                )
                self.final_test_metrics[dataset_name] = (
                    test_metrics[dataset_name]
                )
                self.final_checkpoint_paths[dataset_name] = checkpoint_path
                self._write_completion(
                    dataset_name,
                    dataset_config,
                )
                self._cleanup_checkpoint_artifacts(
                    dataset_name,
                    checkpoint_path,
                )
                self._close_csv_writer()

            self.epoch = 0
            self.current_step = 0

        return {
            "checkpoint_cleanup_failures": list(
                self.checkpoint_cleanup_failures
            ),
            "validation_metrics": all_validation,
            "test_metrics": all_test,
            "hpo_stop_reason": hpo_stop_reason.value,
            "pruned": hpo_stop_reason is HpoStopReason.OPTUNA_PRUNED,
        }
