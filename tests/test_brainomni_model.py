"""Tests for BrainOmni pretrained and scratch model construction."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from baseline.brainomni import model as brainomni_model
from baseline.brainomni.brainomni_config import BrainOmniConfig


@pytest.mark.parametrize(
	"field,value",
	[
		("signal_normalize_eps", float("nan")),
		("signal_normalize_eps", float("inf")),
		("position_normalize_eps", float("nan")),
		("position_normalize_eps", float("inf")),
	],
)
def test_config_rejects_nonfinite_normalization_epsilon(
	field: str,
	value: float,
) -> None:
	"""Normalization safeguards require finite positive epsilon values."""
	config = BrainOmniConfig()
	setattr(config.model, field, value)

	assert config.validate_config() is False


@pytest.mark.parametrize("lm_dim,lm_head", [(256, 8), (512, 16)])
def test_pretrained_loader_uses_checkpoint_model_cfg(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path,
	lm_dim: int,
	lm_head: int,
) -> None:
	"""Tiny/base-style checkpoints determine architecture from model_cfg.json."""
	model_cfg = {"lm_dim": lm_dim, "lm_head": lm_head, "custom_setting": "preserved"}
	(tmp_path / "model_cfg.json").write_text(json.dumps(model_cfg), encoding="utf-8")
	(tmp_path / "BrainOmni.pt").touch()
	seen: dict[str, object] = {}

	def build_from_cfg(model_cfg: dict[str, object]) -> SimpleNamespace:
		seen["cfg"] = model_cfg
		return SimpleNamespace(lm_dim=model_cfg["lm_dim"])

	def load_weights(**kwargs: object) -> tuple[list[str], list[str]]:
		seen["weights_path"] = kwargs["pretrained_path"]
		return [], []

	monkeypatch.setattr(brainomni_model, "build_brainomni_from_cfg", build_from_cfg)
	monkeypatch.setattr(brainomni_model, "load_brainomni_weights", load_weights)

	loaded_model, loaded_dim = brainomni_model.load_brainomni_from_pretrained(str(tmp_path))

	assert seen["cfg"] == model_cfg
	assert seen["weights_path"] == str(tmp_path)
	assert loaded_model.lm_dim == lm_dim
	assert loaded_dim == lm_dim


def test_scratch_builder_accepts_arbitrary_architecture(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Scratch construction forwards user-provided architecture unchanged."""
	model_cfg = {"lm_dim": 384, "lm_head": 12, "lm_depth": 7}
	created: dict[str, object] = {}

	class FakeBrainOmni:
		def __init__(self, **kwargs: object) -> None:
			created.update(kwargs)

	monkeypatch.setattr(brainomni_model, "import_brainomni_class", lambda: FakeBrainOmni)

	brainomni_model.build_brainomni_from_cfg(model_cfg)

	assert created == model_cfg


def test_pretrained_loader_reports_missing_architecture_file(tmp_path) -> None:
	"""Pretrained directories must include an architecture JSON file."""
	(tmp_path / "BrainOmni.pt").touch()

	with pytest.raises(FileNotFoundError, match="model_cfg.json"):
		brainomni_model.load_brainomni_model_cfg(tmp_path)
