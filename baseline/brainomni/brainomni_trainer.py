"""BrainOmni trainer implementation."""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from baseline.abstract.classifier import MultiHeadClassifier
from baseline.abstract.trainer import AbstractTrainer
from baseline.brainomni.brainomni_adapter import BrainOmniDataLoaderFactory
from baseline.brainomni.brainomni_config import BrainOmniConfig, BrainOmniModelArgs
from baseline.brainomni.model import (
	build_brainomni_from_cfg,
	load_brainomni_from_pretrained,
	load_brainomni_weights,
)


logger = logging.getLogger("baseline")


class BrainOmniUnifiedModel(nn.Module):
	"""Unified BrainOmni model wrapper for baseline training."""

	def __init__(self, encoder: nn.Module, classifier: MultiHeadClassifier, grad_cam: bool = False) -> None:
		super().__init__()
		self.encoder = encoder
		self.classifier = classifier

		self.grad_cam = grad_cam
		self.grad_cam_activation = None

	def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
		"""Run BrainOmni encoder and dataset-specific classification head.

		Parameters
		----------
		batch : Dict[str, torch.Tensor]
			Batch dictionary including ``data``, ``pos``, ``sensor_type`` and ``montage``.

		Returns
		-------
		torch.Tensor
			Classification logits with shape ``(B, n_class)``.
		"""
		x = batch["data"]
		pos = batch["pos"]
		sensor_type = batch["sensor_type"].long()
		montage = batch["montage"][0]

		features = self.encoder.encode(x=x, pos=pos, sensor_type=sensor_type)

		if features.ndim != 4:
			raise ValueError(
				"BrainOmni encoder.encode is expected to output 4D tensor [B, C, W, D], "
				f"but got shape {tuple(features.shape)}."
			)

		if self.grad_cam:
			self.grad_cam_activation = features

		# Classifier expects input shape (B, time, channels, embed_dim).
		features = features.permute(0, 2, 1, 3)

		# Classifier routes by montage/dataset key internally.
		logits = self.classifier(features, montage)

		return logits


class BrainOmniTrainer(AbstractTrainer):
	"""BrainOmni trainer implementing the abstract baseline interfaces."""

	def __init__(self, cfg: BrainOmniConfig) -> None:
		super().__init__(cfg)
		self.cfg = cfg

		self.dataloader_factory = BrainOmniDataLoaderFactory(
			batch_size=self.cfg.data.batch_size,
			num_workers=self.cfg.data.num_workers,
			pin_memory=self.cfg.data.pin_memory,
			seed=self.cfg.seed,
			normalize_input=self.cfg.model.normalize_input,
			normalize_position=self.cfg.model.normalize_position,
			signal_normalize_eps=self.cfg.model.signal_normalize_eps,
			position_normalize_eps=self.cfg.model.position_normalize_eps,
		)

		self.encoder: nn.Module | None = None
		self.classifier: MultiHeadClassifier | None = None

		self.loss_fn = nn.CrossEntropyLoss()

	def setup_model(self) -> nn.Module:
		"""Setup BrainOmni encoder and classifier model."""
		logger.info("Setting up brainomni model architecture...")
		model_cfg = self.cfg.model

		self.encoder, embed_dim = self._build_encoder(model_cfg)
		output_channels = self._get_encoder_output_channels(self.encoder)

		head_configs = {ds_name: info["n_class"] for ds_name, info in self.ds_info.items()}
		ds_shape_info = self._build_ds_shape_info(
			output_channels=output_channels,
			embed_dim=embed_dim,
		)

		self.classifier = MultiHeadClassifier(
			embed_dim=embed_dim,
			head_configs=head_configs,
			head_cfg=model_cfg.classifier_head,
			ds_shape_info=ds_shape_info,
			t_sne=model_cfg.t_sne,
		)
		logger.info("Created multi-head classifier with heads: %s", list(head_configs.keys()))

		model = BrainOmniUnifiedModel(
			encoder=self.encoder,
			classifier=self.classifier,
			grad_cam=model_cfg.grad_cam,
		)

		model = self.apply_lora(model)
		model = model.to(self.device)
		model = self.maybe_wrap_ddp(model, find_unused_parameters=True)

		self.model = model
		logger.info("BrainOmni model setup complete for datasets: %s", list(self.ds_info.keys()))
		return model

	def load_checkpoint(self, checkpoint_path: str) -> None:
		"""Load BrainOmni checkpoint into current encoder.

		Parameters
		----------
		checkpoint_path : str
			Directory containing BrainOmni checkpoint files.
		"""
		if not checkpoint_path:
			logger.warning("Empty checkpoint_path provided for BrainOmni; skipping load.")
			return

		if self.encoder is None:
			raise ValueError(
				"BrainOmni encoder is not initialized when load_checkpoint was called. "
				"Call setup_model() first."
			)

		try:
			missing_keys, unexpected_keys = load_brainomni_weights(
				model=self.encoder,
				pretrained_path=checkpoint_path,
				strict=self.cfg.model.strict_load,
				map_location=self.device,
			)
		except Exception as exc:
			logger.warning("Failed to load BrainOmni checkpoint '%s': %s", checkpoint_path, exc)
			return

		if missing_keys:
			logger.warning("BrainOmni checkpoint missing keys: %s", missing_keys)
		if unexpected_keys:
			logger.warning("BrainOmni checkpoint unexpected keys: %s", unexpected_keys)

	def _build_encoder(self, cfg: BrainOmniModelArgs) -> Tuple[nn.Module, int]:
		"""Build BrainOmni encoder from pretrained checkpoint or fallback config."""
		if cfg.pretrained_path:
			logger.info("Loading BrainOmni pretrained checkpoint from: %s", cfg.pretrained_path)
			encoder, embed_dim = load_brainomni_from_pretrained(
				pretrained_path=cfg.pretrained_path,
				strict=cfg.strict_load,
				freeze_tokenizer=cfg.freeze_tokenizer,
				map_location=self.device,
			)
			return encoder, embed_dim

		logger.info("No pretrained_path specified for BrainOmni; building from config defaults.")
		constructor_cfg = {
			"window_length": cfg.window_length,
			"n_filters": cfg.n_filters,
			"ratios": list(cfg.ratios),
			"kernel_size": cfg.kernel_size,
			"last_kernel_size": cfg.last_kernel_size,
			"n_dim": cfg.n_dim,
			"n_head": cfg.n_head,
			"n_neuro": cfg.n_neuro,
			"dropout": cfg.dropout,
			"codebook_dim": cfg.codebook_dim,
			"codebook_size": cfg.codebook_size,
			"num_quantizers": cfg.num_quantizers,
			"rotation_trick": cfg.rotation_trick,
			"quantize_optimize_method": cfg.quantize_optimize_method,
			"overlap_ratio": cfg.overlap_ratio,
			"lm_dim": cfg.lm_dim,
			"lm_head": cfg.lm_head,
			"lm_depth": cfg.lm_depth,
			"lm_dropout": cfg.lm_dropout,
			"mask_ratio": cfg.mask_ratio,
			"num_quantizers_used": cfg.num_quantizers_used,
		}

		encoder = build_brainomni_from_cfg(model_cfg=constructor_cfg)
		if cfg.freeze_tokenizer and hasattr(encoder, "tokenizer"):
			for param in encoder.tokenizer.parameters():
				param.requires_grad = False

		embed_dim = int(getattr(encoder, "lm_dim"))
		return encoder, embed_dim

	@staticmethod
	def _get_encoder_output_channels(encoder: nn.Module) -> int:
		"""Read BrainOmni output channel count from tokenizer neuro tokens.

		Parameters
		----------
		encoder : nn.Module
			BrainOmni model instance.

		Returns
		-------
		int
			Number of latent channels output by ``encode``.

		Raises
		------
		ValueError
			If expected tokenizer structure is missing.
		"""
		try:
			neuros = encoder.tokenizer.encoder.neuros
			n_channels = int(neuros.shape[0])
		except Exception as exc:
			raise ValueError(
				"Failed to infer BrainOmni latent channel count from encoder.tokenizer.encoder.neuros."
			) from exc

		if n_channels <= 0:
			raise ValueError(
				"BrainOmni latent channel count must be positive, "
				f"but got {n_channels}."
			)
		return n_channels

	def _build_ds_shape_info(self, output_channels: int, embed_dim: int) -> Dict[str, Tuple[int, int, int]]:
		"""Build montage-keyed shape info for classifier heads.

		Parameters
		----------
		output_channels : int
			BrainOmni latent channel count.
		embed_dim : int
			BrainOmni feature embedding size.

		Returns
		-------
		Dict[str, Tuple[int, int, int]]
			``montage_key -> (n_patches, n_channels, embed_dim)`` mapping.
		"""
		ds_shape_info: Dict[str, Tuple[int, int, int]] = {}
		for ds_name, info in self.ds_info.items():
			for montage_key, (n_timepoints, _) in info["shape_info"].items():
				n_tokens = self._estimate_encoded_tokens(n_timepoints)
				ds_shape_info[montage_key] = (n_tokens, output_channels, embed_dim)

			logger.info(
				"BrainOmni shape info ready for dataset '%s': %d montages.",
				ds_name,
				len(info["shape_info"]),
			)
		return ds_shape_info

	def _estimate_encoded_tokens(self, n_timepoints: int) -> int:
		"""Estimate BrainOmni token sequence length from input timepoints.

		Parameters
		----------
		n_timepoints : int
			Input waveform length in samples.

		Returns
		-------
		int
			Estimated number of temporal tokens produced by ``BrainOmni.encode``.

		Raises
		------
		ValueError
			If encoder fields are invalid.
		"""
		if self.encoder is None:
			raise ValueError("Cannot estimate BrainOmni tokens because encoder is not initialized.")

		window_length = int(getattr(self.encoder, "window_length"))
		overlap_ratio = float(getattr(self.encoder, "overlap_ratio"))

		if window_length <= 0:
			raise ValueError(f"BrainOmni window_length must be positive, got {window_length}.")
		if not (0.0 <= overlap_ratio < 1.0):
			raise ValueError(
				"BrainOmni overlap_ratio must satisfy 0 <= overlap_ratio < 1, "
				f"but got {overlap_ratio}."
			)

		stride = int(window_length * (1.0 - overlap_ratio))
		if stride <= 0:
			raise ValueError(
				"BrainOmni computed non-positive unfold stride. "
				f"window_length={window_length}, overlap_ratio={overlap_ratio}, stride={stride}."
			)

		signal_length = int(n_timepoints)
		if signal_length < window_length:
			signal_length = window_length

		if overlap_ratio > 0.0:
			right_remain = (signal_length - window_length) % stride
			if right_remain > 0:
				signal_length += (stride - right_remain)

		n_windows = 1 + (signal_length - window_length) // stride

		hop_length = self._get_encoder_hop_length(self.encoder, self.cfg.model)
		tokens_per_window = int(math.ceil(window_length / hop_length))
		n_tokens = int(n_windows * tokens_per_window)

		if n_tokens <= 0:
			raise ValueError(
				"BrainOmni estimated non-positive token count. "
				f"signal_length={signal_length}, n_windows={n_windows}, "
				f"hop_length={hop_length}, tokens_per_window={tokens_per_window}."
			)
		return n_tokens

	@staticmethod
	def _get_encoder_hop_length(encoder: nn.Module, cfg: BrainOmniModelArgs) -> int:
		"""Infer tokenizer temporal hop length from encoder internals.

		Parameters
		----------
		encoder : nn.Module
			BrainOmni model instance.
		cfg : BrainOmniModelArgs
			BrainOmni model config.

		Returns
		-------
		int
			Temporal hop length.

		Raises
		------
		ValueError
			If inferred hop length is invalid.
		"""
		hop_length: Optional[int] = None

		try:
			hop_length = int(encoder.tokenizer.encoder.seanet_encoder.hop_length)
		except Exception:
			try:
				hop_length = int(np.prod(cfg.ratios))
			except Exception as exc:
				raise ValueError("Failed to infer BrainOmni hop length from cfg.ratios.") from exc

		if hop_length <= 0:
			raise ValueError(f"BrainOmni hop length must be positive, but got {hop_length}.")
		return hop_length


def main() -> None:
	"""Entry point for standalone BrainOmni training."""
	import sys

	from omegaconf import OmegaConf

	if len(sys.argv) != 2:
		print("Usage: python brainomni_trainer.py path/to/config.yaml")
		raise SystemExit(1)

	config_path = sys.argv[1]
	file_cfg = OmegaConf.load(config_path)
	code_cfg = OmegaConf.create(BrainOmniConfig().model_dump())
	merged_config = OmegaConf.merge(code_cfg, file_cfg)
	config_dict = OmegaConf.to_container(merged_config, resolve=True)
	cfg = BrainOmniConfig.model_validate(config_dict)

	trainer = BrainOmniTrainer(cfg)
	trainer.run()


if __name__ == "__main__":
	main()

