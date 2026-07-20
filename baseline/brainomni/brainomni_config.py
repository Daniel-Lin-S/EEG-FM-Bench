"""BrainOmni configuration definitions."""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from baseline.abstract.config import (
	AbstractConfig,
	BaseDataArgs,
	BaseLoggingArgs,
	BaseModelArgs,
	BaseTrainingArgs,
)


class BrainOmniModelArgs(BaseModelArgs):
	"""BrainOmni preprocessing and scratch-architecture settings.

	The inherited ``pretrained_path`` must name a directory containing
	``model_cfg.json`` and ``BrainOmni.pt``. Its JSON configuration determines
	the pretrained architecture; the fields below are used only when that path
	is unset.

	Parameters
	----------
	position_montage : str
		MNE standard montage used to derive six-dimensional sensor metadata;
		``"auto"`` resolves it from each benchmark montage key.
	normalize_input : bool
		Whether to remove the per-timepoint channel mean and divide each sample
		by its global population standard deviation before encoding.
	normalize_position : bool
		Whether to center and scale sensor xyz coordinates before encoding.
	allow_missing_positions : bool
		Whether channels absent from the resolved montage may use zero position
		and orientation vectors.
	signal_normalize_eps, position_normalize_eps : float
		Positive numerical floors used by signal and position normalization.
	freeze_tokenizer : bool
		Whether tokenizer parameters are excluded from gradient updates.
	strict_load : bool
		Whether checkpoint state-dict loading must match the encoder exactly.
	window_length, n_filters, ratios, kernel_size, last_kernel_size, n_dim, n_head,
	n_neuro, dropout, codebook_dim, codebook_size, num_quantizers,
	rotation_trick, quantize_optimize_method, overlap_ratio, lm_dim, lm_head,
	lm_depth, lm_dropout, mask_ratio, num_quantizers_used
		Scratch-only arguments forwarded unchanged to the vendored ``BrainOmni``
		constructor.
	"""

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
	# see docstring of BrainOmni for details of each parameter
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
	"""BrainOmni overrides of shared training defaults.

	All unlisted settings, including weight decay, clipping, AMP, encoder
	freezing, OneCycleLR fraction, and LoRA, inherit from ``BaseTrainingArgs``.

	Parameters
	----------
	max_epochs : int
		Number of complete passes over each training loader.
	lr_schedule : {"onecycle", "cosine"}
		Schedule used by the shared optimizer setup.
	max_lr : float
		Classifier learning rate and base rate for encoder parameters.
	encoder_lr_scale : float
		Multiplier applied to ``max_lr`` for trainable encoder parameters.
	warmup_epochs : int
		Number of loader-length epochs used for cosine-schedule warmup.
	warmup_scale : float
		Initial learning-rate fraction during cosine-schedule warmup.
	min_lr : float
		Final learning-rate floor for cosine annealing.
	"""

	max_epochs: int = 20

	lr_schedule: str = "cosine"
	max_lr: float = 5e-4
	encoder_lr_scale: float = 0.5
	warmup_epochs: int = 2
	warmup_scale: float = 1e-1
	min_lr: float = 5e-5


class BrainOmniLoggingArgs(BaseLoggingArgs):
	"""BrainOmni overrides of shared experiment-tracking defaults.

	All unlisted logging settings inherit from ``BaseLoggingArgs``.

	Parameters
	----------
	experiment_name : str
		Default run name used in local and cloud logs.
	use_cloud : bool
		Whether cloud experiment tracking is enabled by default.
	project : str or None
		Default cloud project name.
	ckpt_interval : int
		Number of completed epochs between checkpoint saves.
		Default: 10
	"""

	experiment_name: str = "brainomni"

	use_cloud: bool = True
	project: Optional[str] = "brainomni"

	ckpt_interval: int = 10


class BrainOmniConfig(AbstractConfig):
	"""Top-level configuration for the BrainOmni trainer.

	Shared run, dataset, sampling-frequency, and base configuration fields
	inherit from ``AbstractConfig``. BrainOmni uses the base data arguments
	directly because it does not change their defaults.

	Parameters
	----------
	model_type : str
		Registry identifier fixed to ``"brainomni"``.
	data : BaseDataArgs
		Dataset mapping and data-loader settings.
	model : BrainOmniModelArgs
		BrainOmni preprocessing, pretrained loading, and scratch architecture.
	training : BrainOmniTrainingArgs
		BrainOmni training-default overrides.
	logging : BrainOmniLoggingArgs
		BrainOmni logging-default overrides.
	"""

	model_type: str = "brainomni"

	data: BaseDataArgs = Field(default_factory=BaseDataArgs)
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

