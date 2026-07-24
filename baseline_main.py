#!/usr/bin/env python3
"""Unified baseline training, HPO, and multi-seed campaign entry point.

Usage
-----
Run a fixed configuration for every top-level seed::

    python baseline_main.py conf_file=assets/conf/baseline/example.yaml

Audit campaign reuse without writing artifacts or starting trainers::

    python baseline_main.py audit-campaign conf_file=example.yaml

Override campaign or HPO fields without changing YAML::

    python baseline_main.py conf_file=example.yaml seeds=[42,43,44]
    python baseline_main.py conf_file=example.yaml hpo.n_trials=100

Input
-----
The selected YAML is a model-specific baseline configuration. Public
reproducibility uses seeds (default [42]). An optional top-level hpo mapping
configures validation-only Optuna search with its own independent hpo.seed.
Legacy scalar seed is converted with a warning.

Output
------
Campaign artifacts are written below the configured absolute run_dir. Each
seed retains the existing configs/CSV/TensorBoard/log/dataset layout under
logs/seed_<seed>. HPO trials and aggregate test summaries are stored under
the same campaign root.
"""

import json
import sys
import warnings
from typing import Any

from omegaconf import DictConfig, OmegaConf

from baseline.abstract.factory import ModelRegistry
from baseline.hpo.config import HpoConfig
from baseline.hpo.orchestrator import CampaignRunner
from baseline.registry import register_builtin_models
from baseline.utils.identity import DETERMINISTIC_MODEL_TYPES
from common.path import get_conf_file_path
from common.utils import setup_yaml


def _normalize_legacy_seed(
    config: DictConfig,
    source_name: str,
) -> None:
    """Convert one legacy scalar seed mapping to the public seeds list."""
    if "seed" not in config:
        return
    if "seeds" in config:
        raise ValueError(
            f"{source_name} provides both deprecated 'seed' and 'seeds'. "
            "Remove 'seed' and keep only the ordered seeds list."
        )
    seed = config.pop("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(
            f"{source_name} seed must be a non-negative integer, but got "
            f"{seed!r}."
        )
    config["seeds"] = [seed]
    warnings.warn(
        f"{source_name} uses deprecated scalar 'seed'; converting it to "
        f"'seeds: [{seed}]'.",
        DeprecationWarning,
        stacklevel=2,
    )


def _load_configs() -> tuple[type[Any], Any, HpoConfig]:
    """Load CLI/YAML, separate workflow settings, and validate the model."""
    cli_args = OmegaConf.from_cli()
    if "conf_file" not in cli_args:
        raise ValueError(
            "Please provide a config file: conf_file=path/to/config.yaml"
        )

    conf_file_path = get_conf_file_path(cli_args.conf_file)
    file_cfg = OmegaConf.load(conf_file_path)
    _normalize_legacy_seed(file_cfg, "Configuration file")
    _normalize_legacy_seed(cli_args, "Command line")

    model_type = cli_args.get("model_type")
    if model_type is None:
        model_type = file_cfg.get("model_type")
    available_models = ModelRegistry.list_models()
    if model_type not in available_models:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: "
            f"{available_models}"
        )

    workflow_defaults = OmegaConf.create({
        "hpo": HpoConfig().model_dump(mode="json"),
    })
    workflow_merged = OmegaConf.merge(
        workflow_defaults,
        {"hpo": file_cfg.get("hpo", {})},
        {"hpo": cli_args.get("hpo", {})},
    )
    hpo_dict = OmegaConf.to_container(
        workflow_merged.hpo,
        resolve=True,
        throw_on_missing=True,
    )
    hpo_config = HpoConfig.model_validate(hpo_dict)

    file_model_cfg = OmegaConf.create(
        OmegaConf.to_container(file_cfg, resolve=False)
    )
    cli_model_cfg = OmegaConf.create(
        OmegaConf.to_container(cli_args, resolve=False)
    )
    file_model_cfg.pop("hpo", None)
    cli_model_cfg.pop("hpo", None)

    config_class = ModelRegistry.get_config_class(model_type)
    code_cfg = OmegaConf.create(config_class().model_dump())
    merged_config = OmegaConf.merge(
        code_cfg,
        file_model_cfg,
        cli_model_cfg,
    )
    merged_config.model_type = model_type
    cfg_dict = OmegaConf.to_container(
        merged_config,
        resolve=True,
        throw_on_missing=True,
    )
    cfg = config_class.model_validate(cfg_dict)
    if not cfg.validate_config():
        raise ValueError(
            f"Invalid configuration for model type: {model_type}"
        )
    return config_class, cfg, hpo_config


def _run_feature_extractor(config: Any, hpo_config: HpoConfig) -> None:
    """Run one deterministic extractor outside the multi-seed campaign path."""
    if len(config.seeds) != 1:
        raise ValueError(
            f"{config.model_type} supports exactly one deterministic seed, "
            f"but got {config.seeds}."
        )
    if hpo_config.enabled:
        warnings.warn(
            f"{config.model_type} is deterministic; ignoring the configured "
            "HPO section.",
            UserWarning,
            stacklevel=2,
        )
    trainer = ModelRegistry.create_trainer(config)
    trainer.run()


def main() -> None:
    """Run one validated baseline campaign."""
    register_builtin_models()
    setup_yaml()
    config_class, config, hpo_config = _load_configs()
    if config.model_type in DETERMINISTIC_MODEL_TYPES:
        _run_feature_extractor(config, hpo_config)
        return
    CampaignRunner(config, hpo_config, config_class).run()


def audit_campaign() -> None:
    """Print a read-only semantic campaign and artifact audit as JSON."""
    register_builtin_models()
    setup_yaml()
    config_class, config, hpo_config = _load_configs()
    report = CampaignRunner(
        config,
        hpo_config,
        config_class,
    ).audit()
    print(json.dumps(report, indent=2, sort_keys=True))


def list_available_models() -> None:
    """Print all registered baseline model identifiers."""
    register_builtin_models()
    print("Available baseline models:")
    for model_type in ModelRegistry.list_models():
        print(f"  - {model_type}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "list-models":
        list_available_models()
    elif len(sys.argv) > 1 and sys.argv[1] == "audit-campaign":
        sys.argv.pop(1)
        audit_campaign()
    else:
        main()
