"""Tests for explicit sampling-rate configuration contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import preproc as preproc_module
from baseline.registry import register_builtin_models
from baseline_main import _load_configs
from common.utils import setup_yaml
from data.dataset.tue.tuar import TuarBuilder


REPOSITORY_ROOT = Path(__file__).parents[1]


def test_shipped_baseline_and_preproc_yaml_define_fs() -> None:
    """Every runnable shipped YAML names a positive sampling rate."""
    config_roots = (
        REPOSITORY_ROOT / "assets" / "conf" / "baseline",
        REPOSITORY_ROOT / "assets" / "conf" / "preproc",
    )
    for config_root in config_roots:
        for config_path in sorted(config_root.rglob("*.yaml")):
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            assert isinstance(payload, dict)
            assert isinstance(payload.get("fs"), int)
            assert payload["fs"] > 0


def test_baseline_loading_rejects_missing_fs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Baseline validation receives only user YAML and CLI values."""
    config_path = tmp_path / "missing-fs.yaml"
    config_path.write_text(
        "model_type: catch22\n"
        "data:\n"
        "  datasets:\n"
        "    tuar: finetune\n",
        encoding="utf-8",
    )
    register_builtin_models()
    setup_yaml()
    monkeypatch.setattr(
        sys,
        "argv",
        ["baseline_main.py", f"conf_file={config_path}"],
    )

    with pytest.raises(ValidationError, match="fs"):
        _load_configs()


def test_preproc_loading_rejects_missing_fs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preprocessing validation has no implicit configuration defaults."""
    monkeypatch.setattr(sys, "argv", ["preproc.py"])

    with pytest.raises(ValueError, match="conf_file YAML"):
        preproc_module._load_config()


def test_preproc_loading_rejects_cli_fs_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preprocessing rates are declared only by the selected YAML."""
    config_path = tmp_path / "preproc.yaml"
    config_path.write_text("fs: 256\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["preproc.py", f"conf_file={config_path}", "fs=200"],
    )

    with pytest.raises(ValueError, match="defined by the YAML"):
        preproc_module._load_config()


def test_dataset_builder_rejects_omitted_sampling_rate() -> None:
    """Direct builder construction cannot choose an implicit rate."""
    with pytest.raises(ValueError, match="sampling rate via fs"):
        TuarBuilder("finetune")
