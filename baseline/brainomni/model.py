"""BrainOmni model construction and checkpoint loading utilities.

This module is self-contained inside EEG-FM-Bench and imports BrainOmni runtime
code only from the vendored source tree under ``baseline/brainomni/vendor``.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn


logger = logging.getLogger("baseline")

_THIS_FILE = Path(__file__).resolve()
_EEG_FM_BENCH_ROOT = _THIS_FILE.parents[2]
_CODES_ROOT = _EEG_FM_BENCH_ROOT.parent
_VENDORED_BRAINOMNI_ROOT = _THIS_FILE.parent / "vendor"

_DEEPSPEED_FALLBACK_ENABLED = False


def resolve_existing_path(path_str: str) -> Path:
    """Resolve a path string against common workspace roots.

    Parameters
    ----------
    path_str : str
        User-provided path string, absolute or relative.

    Returns
    -------
    Path
        Resolved existing path.

    Raises
    ------
    FileNotFoundError
        If no candidate path exists.
    """
    raw_path = Path(path_str).expanduser()
    candidates: list[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                (Path.cwd() / raw_path).resolve(),
                (_EEG_FM_BENCH_ROOT / raw_path).resolve(),
                (_CODES_ROOT / raw_path).resolve(),
            ]
        )

    seen: set[str] = set()
    deduped_candidates: list[Path] = []
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str not in seen:
            deduped_candidates.append(candidate)
            seen.add(candidate_str)

    for candidate in deduped_candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not resolve path "
        f"'{path_str}'. Tried: {[str(c) for c in deduped_candidates]}"
    )


def resolve_pretrained_dir(pretrained_path: str) -> Path:
    """Resolve a BrainOmni pretrained directory path.

    Parameters
    ----------
    pretrained_path : str
        Path to pretrained directory containing ``model_cfg.json`` and
        ``BrainOmni.pt``.

    Returns
    -------
    Path
        Existing pretrained directory.

    Raises
    ------
    ValueError
        If resolved path is not a directory.
    """
    pretrained_dir = resolve_existing_path(pretrained_path)
    if not pretrained_dir.is_dir():
        raise ValueError(
            "BrainOmni pretrained path must be a directory, "
            f"but got '{pretrained_dir}'."
        )
    return pretrained_dir


def load_brainomni_model_cfg(
    pretrained_dir: Path,
    config_filename: str = "model_cfg.json",
) -> Dict[str, Any]:
    """Load BrainOmni model JSON configuration.

    Parameters
    ----------
    pretrained_dir : Path
        Pretrained checkpoint directory.
    config_filename : str, optional
        Model config file name.

    Returns
    -------
    Dict[str, Any]
        BrainOmni model configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If config file does not exist.
    ValueError
        If loaded configuration is empty.
    """
    model_cfg_path = pretrained_dir / config_filename
    if not model_cfg_path.exists():
        raise FileNotFoundError(
            "BrainOmni model config file was not found at "
            f"'{model_cfg_path}'."
        )

    with open(model_cfg_path, "r", encoding="utf-8") as file_obj:
        model_cfg = json.load(file_obj)

    if not isinstance(model_cfg, dict) or len(model_cfg) == 0:
        raise ValueError(
            "BrainOmni model config should be a non-empty JSON object, "
            f"but got type '{type(model_cfg).__name__}' from '{model_cfg_path}'."
        )
    return model_cfg


def import_brainomni_class() -> type[nn.Module]:
    """Import vendored ``brainomni.model.BrainOmni`` class.

    Returns
    -------
    type[nn.Module]
        BrainOmni model class object from EEG-FM-Bench vendored runtime.

    Raises
    ------
    ImportError
        If BrainOmni class cannot be imported.
    """
    _ensure_vendored_runtime_on_sys_path()
    _ensure_deepspeed_comm_fallback()
    _purge_non_vendored_runtime_modules()

    try:
        module = importlib.import_module("brainomni.model")
    except Exception as exc:  # pragma: no cover - runtime-dependent import behavior.
        raise ImportError(
            "Failed to import vendored BrainOmni module from "
            f"'{_VENDORED_BRAINOMNI_ROOT}': {exc}"
        ) from exc

    if not hasattr(module, "BrainOmni"):
        raise ImportError(
            "Vendored module 'brainomni.model' does not expose class 'BrainOmni'. "
            f"Loaded module file: {getattr(module, '__file__', 'unknown')}"
        )
    return getattr(module, "BrainOmni")


def build_brainomni_from_cfg(model_cfg: Dict[str, Any]) -> nn.Module:
    """Build BrainOmni model from a model configuration dictionary.

    Parameters
    ----------
    model_cfg : Dict[str, Any]
        BrainOmni constructor kwargs.
    Returns
    -------
    nn.Module
        Instantiated BrainOmni model.
    """
    brainomni_cls = import_brainomni_class()
    return brainomni_cls(**model_cfg)


def load_brainomni_weights(
    model: nn.Module,
    pretrained_path: str,
    strict: bool = False,
    map_location: str | torch.device = "cpu",
    checkpoint_filename: str = "BrainOmni.pt",
) -> Tuple[list[str], list[str]]:
    """Load BrainOmni checkpoint weights into an existing model.

    Parameters
    ----------
    model : nn.Module
        Target BrainOmni model instance.
    pretrained_path : str
        Path to pretrained directory.
    strict : bool, optional
        Whether to enforce strict key matching.
    map_location : str | torch.device, optional
        Torch load device.
    checkpoint_filename : str, optional
        Checkpoint file name in ``pretrained_path``.

    Returns
    -------
    Tuple[list[str], list[str]]
        Missing keys and unexpected keys.

    Raises
    ------
    FileNotFoundError
        If checkpoint file does not exist.
    """
    pretrained_dir = resolve_pretrained_dir(pretrained_path)
    checkpoint_path = pretrained_dir / checkpoint_filename
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "BrainOmni checkpoint file was not found at "
            f"'{checkpoint_path}'."
        )

    checkpoint_obj = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state_dict = _extract_state_dict(checkpoint_obj)
    state_dict = _strip_module_prefix(state_dict)

    incompatible = model.load_state_dict(state_dict, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def load_brainomni_from_pretrained(
    pretrained_path: str,
    strict: bool = False,
    freeze_tokenizer: bool = False,
    map_location: str | torch.device = "cpu",
) -> Tuple[nn.Module, int]:
    """Build and load a BrainOmni model from pretrained directory.

    Parameters
    ----------
    pretrained_path : str
        Directory containing the ``model_cfg.json`` architecture definition
        and ``BrainOmni.pt`` weights.
    strict : bool, optional
        Whether to enforce strict state-dict loading.
    freeze_tokenizer : bool, optional
        Whether to freeze tokenizer parameters after loading.
    map_location : str | torch.device, optional
        Torch load device.

    Returns
    -------
    Tuple[nn.Module, int]
        Loaded BrainOmni model and embedding dimension ``lm_dim``.
    """
    pretrained_dir = resolve_pretrained_dir(pretrained_path)
    model_cfg = load_brainomni_model_cfg(pretrained_dir)

    model = build_brainomni_from_cfg(model_cfg=model_cfg)
    missing_keys, unexpected_keys = load_brainomni_weights(
        model=model,
        pretrained_path=str(pretrained_dir),
        strict=strict,
        map_location=map_location,
    )

    if missing_keys:
        logger.warning("BrainOmni checkpoint loading missing keys: %s", missing_keys)
    if unexpected_keys:
        logger.warning("BrainOmni checkpoint loading unexpected keys: %s", unexpected_keys)

    if freeze_tokenizer and hasattr(model, "tokenizer"):
        for param in model.tokenizer.parameters():
            param.requires_grad = False

    lm_dim = int(getattr(model, "lm_dim"))
    return model, lm_dim


def _ensure_vendored_runtime_on_sys_path() -> None:
    """Ensure vendored BrainOmni runtime modules are importable.

    Raises
    ------
    FileNotFoundError
        If expected vendored runtime directories are missing.
    """
    expected_dirs = ["brainomni", "braintokenizer", "model_utils"]
    missing = [
        name
        for name in expected_dirs
        if not (_VENDORED_BRAINOMNI_ROOT / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Vendored BrainOmni runtime is incomplete under "
            f"'{_VENDORED_BRAINOMNI_ROOT}'. Missing directories: {missing}."
        )

    vendor_root_str = str(_VENDORED_BRAINOMNI_ROOT)
    if vendor_root_str not in sys.path:
        sys.path.insert(0, vendor_root_str)


def _ensure_deepspeed_comm_fallback() -> None:
    """Install a lightweight ``deepspeed.comm`` fallback when unavailable.

    Vendored BrainOmni quantization utilities import ``deepspeed.comm``. For
    single-process baseline finetuning/inference, only a minimal subset of
    communication APIs is required.
    """
    global _DEEPSPEED_FALLBACK_ENABLED

    try:
        importlib.import_module("deepspeed.comm")
        return
    except Exception:
        pass

    deepspeed_module = sys.modules.get("deepspeed")
    if deepspeed_module is None:
        deepspeed_module = types.ModuleType("deepspeed")
        sys.modules["deepspeed"] = deepspeed_module

    comm_module = types.ModuleType("deepspeed.comm")

    class _ReduceOp:  # pylint: disable=too-few-public-methods
        SUM = "sum"

    def _is_initialized() -> bool:
        return False

    def _get_world_size() -> int:
        return 1

    def _get_rank() -> int:
        return 0

    def _broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        del src
        return tensor

    def _all_reduce(tensor: torch.Tensor, op: Any = None) -> torch.Tensor:
        del op
        return tensor

    comm_module.ReduceOp = _ReduceOp
    comm_module.is_initialized = _is_initialized
    comm_module.get_world_size = _get_world_size
    comm_module.get_rank = _get_rank
    comm_module.broadcast = _broadcast
    comm_module.all_reduce = _all_reduce

    setattr(deepspeed_module, "comm", comm_module)
    sys.modules["deepspeed.comm"] = comm_module

    if not _DEEPSPEED_FALLBACK_ENABLED:
        logger.warning(
            "deepspeed.comm is not installed; using single-process fallback "
            "for BrainOmni import and checkpoint loading."
        )
        _DEEPSPEED_FALLBACK_ENABLED = True


def _module_is_vendored(module_obj: types.ModuleType) -> bool:
    """Return whether an imported module object resolves inside vendored runtime."""
    module_file = getattr(module_obj, "__file__", None)
    if module_file:
        try:
            Path(module_file).resolve().relative_to(_VENDORED_BRAINOMNI_ROOT.resolve())
            return True
        except Exception:
            return False

    module_paths = getattr(module_obj, "__path__", None)
    if module_paths:
        for path_str in module_paths:
            try:
                Path(path_str).resolve().relative_to(_VENDORED_BRAINOMNI_ROOT.resolve())
                return True
            except Exception:
                continue
    return False


def _purge_non_vendored_runtime_modules() -> None:
    """Remove previously imported non-vendored BrainOmni runtime modules.

    This avoids accidentally reusing modules loaded from an external BrainOmni
    repository in the same Python process.
    """
    prefixes = ("brainomni", "braintokenizer", "model_utils")
    for module_name, module_obj in list(sys.modules.items()):
        if not module_name.startswith(prefixes):
            continue
        if isinstance(module_obj, types.ModuleType) and _module_is_vendored(module_obj):
            continue
        sys.modules.pop(module_name, None)


def _extract_state_dict(checkpoint_obj: Any) -> Dict[str, torch.Tensor]:
    """Extract state-dict from arbitrary checkpoint object.

    Parameters
    ----------
    checkpoint_obj : Any
        Loaded torch checkpoint object.

    Returns
    -------
    Dict[str, torch.Tensor]
        Model state dictionary.

    Raises
    ------
    ValueError
        If no usable state dictionary can be extracted.
    """
    if isinstance(checkpoint_obj, dict):
        candidate_keys = ["state_dict", "model_state_dict", "model", "module"]
        for key in candidate_keys:
            candidate = checkpoint_obj.get(key)
            if isinstance(candidate, dict) and candidate:
                return {
                    str(name): tensor
                    for name, tensor in candidate.items()
                    if isinstance(tensor, torch.Tensor)
                }

        if checkpoint_obj and all(isinstance(v, torch.Tensor) for v in checkpoint_obj.values()):
            return {str(name): tensor for name, tensor in checkpoint_obj.items()}

    raise ValueError(
        "Unable to extract a valid state_dict from checkpoint object of type "
        f"'{type(checkpoint_obj).__name__}'."
    )


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip ``module.`` prefix from state-dict keys if present."""
    if not state_dict:
        return state_dict

    has_module_prefix = all(key.startswith("module.") for key in state_dict.keys())
    if not has_module_prefix:
        return state_dict
    return {key[len("module.") :]: value for key, value in state_dict.items()}
