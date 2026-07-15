"""BrainOmni configuration definitions."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import Field

from baseline.abstract.config import (
	AbstractConfig,
	BaseDataArgs,
	BaseLoggingArgs,
	BaseModelArgs,
	BaseTrainingArgs,
)


class BrainOmniDataArgs(BaseDataArgs):
	"""BrainOmni data configuration."""

	datasets: Dict[str, str] = Field(default_factory=lambda: {})
	batch_size: int = 32
	num_workers: int = 2


class BrainOmniModelArgs(BaseModelArgs):
	"""BrainOmni model configuration."""

	# Paths
	pretrained_path: Optional[str] = None
	# Deprecated compatibility field. EEG-FM-Bench now uses vendored runtime.
	repo_path: Optional[str] = None

	# Adapter behavior
	position_montage: str = "auto"
	normalize_input: bool = True
	normalize_position: bool = True
	allow_missing_positions: bool = False
	signal_normalize_eps: float = 1e-5
	position_normalize_eps: float = 1e-8

	# Checkpoint loading behavior
	freeze_tokenizer: bool = False
	strict_load: bool = False

	# Fallback architecture configuration (used when pretrained_path is None)
	window_length: int = 512
	n_filters: int = 32
	ratios: List[int] = Field(default_factory=lambda: [8, 4, 2])
	kernel_size: int = 5
	last_kernel_size: int = 5
	n_dim: int = 256
	n_head: int = 4
	n_neuro: int = 16
	dropout: float = 0.0
	codebook_dim: int = 256
	codebook_size: int = 512
	num_quantizers: int = 4
	rotation_trick: bool = True
	quantize_optimize_method: str = "ema"

	overlap_ratio: float = 0.25
	lm_dim: int = 256
	lm_head: int = 8
	lm_depth: int = 12
	lm_dropout: float = 0.1
	mask_ratio: float = 0.5
	num_quantizers_used: int = 4


class BrainOmniTrainingArgs(BaseTrainingArgs):
	"""BrainOmni training configuration."""

	max_epochs: int = 20

	weight_decay: float = 0.01
	max_grad_norm: float = 1.0

	lr_schedule: str = "cosine"
	max_lr: float = 5e-4
	encoder_lr_scale: float = 0.5
	warmup_epochs: int = 2
	warmup_scale: float = 1e-1
	pct_start: float = 0.2
	min_lr: float = 5e-5

	use_amp: bool = True
	freeze_encoder: bool = False


class BrainOmniLoggingArgs(BaseLoggingArgs):
	"""BrainOmni logging configuration."""

	experiment_name: str = "brainomni"
	run_dir: str = "assets/run"

	use_cloud: bool = True
	cloud_backend: str = "wandb"
	project: Optional[str] = "brainomni"
	entity: Optional[str] = None

	api_key: Optional[str] = None
	offline: bool = False
	tags: List[str] = Field(default_factory=lambda: [])

	log_step_interval: int = 1
	ckpt_interval: int = 20


class BrainOmniConfig(AbstractConfig):
	"""BrainOmni top-level configuration."""

	model_type: str = "brainomni"
	fs: int = 256

	data: BrainOmniDataArgs = Field(default_factory=BrainOmniDataArgs)
	model: BrainOmniModelArgs = Field(default_factory=BrainOmniModelArgs)
	training: BrainOmniTrainingArgs = Field(default_factory=BrainOmniTrainingArgs)
	logging: BrainOmniLoggingArgs = Field(default_factory=BrainOmniLoggingArgs)

	def validate_config(self) -> bool:
		"""Validate BrainOmni-specific configuration.

		Returns
		-------
		bool
			``True`` if configuration is valid.
		"""
		if not self.model.position_montage:
			return False
		if self.model.signal_normalize_eps <= 0.0:
			return False
		if self.model.position_normalize_eps <= 0.0:
			return False

		if self.model.window_length <= 0:
			return False
		if self.model.n_filters <= 0:
			return False
		if self.model.n_dim <= 0 or self.model.lm_dim <= 0:
			return False
		if self.model.n_neuro <= 0:
			return False

		if len(self.model.ratios) == 0 or any(r <= 0 for r in self.model.ratios):
			return False

		if self.model.n_dim % self.model.n_head != 0:
			return False
		if self.model.lm_dim % self.model.lm_head != 0:
			return False

		if not (0.0 <= self.model.overlap_ratio < 1.0):
			return False
		if not (0.0 <= self.model.mask_ratio <= 1.0):
			return False

		if self.model.num_quantizers <= 0:
			return False
		if self.model.num_quantizers_used <= 0:
			return False
		if self.model.num_quantizers_used > self.model.num_quantizers:
			return False

		if self.training.lr_schedule not in ["onecycle", "cosine"]:
			return False

		return True

