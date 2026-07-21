#!/usr/bin/env python3
"""Unified baseline training entry point and YAML reference.

Usage:
    python baseline_main.py conf_file=assets/conf/baseline/eegpt/eegpt_unified.yaml
    python baseline_main.py conf_file=assets/conf/baseline/eegnet/eegnet.yaml model_type=eegnet

``conf_file`` (``str``, CLI only) identifies the YAML to load.
``model_type`` (``str``) must be supplied in that YAML or on the command line;
it selects the registered config model and trainer (a CLI value takes precedence)

Top-level fields (all optional)
----------------
* ``seed`` (``int``): initializes the trainer's reproducibility and loader
  seeds.
* ``master_port`` (``int``): TCP port used to initialize distributed training.
* ``multitask`` (``bool``): true creates one mixed training loader and shared
  model for all datasets; false trains each dataset separately. Classical
  baselines do not support the mixed mode.
* ``fs`` (``int``): sampling rate expected by shape discovery and the loaders;
  it must equal the rate used when the datasets were preprocessed.


Four sections are required: data, model, training, logging.
Please see the docstring of model-specific classes under `baseline` module
for how to set these parameters.
"""

import sys
from omegaconf import OmegaConf

from baseline.abstract.factory import ModelRegistry
from baseline.registry import register_builtin_models
from common.path import get_conf_file_path
from common.utils import setup_yaml


def main():
    """Main training function that can handle any registered baseline model."""
    register_builtin_models()
    setup_yaml()
    
    # Parse CLI arguments
    cli_args = OmegaConf.from_cli()

    if 'conf_file' not in cli_args:
        raise ValueError("Please provide a config file: conf_file=path/to/config.yaml")
    
    # Get model type from CLI args or config
    model_type: str = cli_args.get('model_type', None)

    # Load config file
    conf_file_path = get_conf_file_path(cli_args.conf_file)
    file_cfg = OmegaConf.load(conf_file_path)

    if model_type is None:
        model_type = file_cfg.get('model_type')

    # Validate model type
    available_models = ModelRegistry.list_models()
    if model_type not in available_models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")
    
    # Create base config for the specified model type
    config_class = ModelRegistry.get_config_class(model_type)
    code_cfg = OmegaConf.create(config_class().model_dump())
    
    # Merge configurations: code defaults < file config < CLI args
    merged_config = OmegaConf.merge(code_cfg, file_cfg, cli_args)
    
    # Ensure model_type is set correctly
    merged_config.model_type = model_type
    
    # Convert to config object
    cfg_dict = OmegaConf.to_container(merged_config, resolve=True, throw_on_missing=True)
    cfg = config_class.model_validate(cfg_dict)
    
    # Validate configuration
    if not cfg.validate_config():
        raise ValueError(f"Invalid configuration for model type: {model_type}")

    # Create and run trainer
    trainer = ModelRegistry.create_trainer(cfg)
    trainer.run()


def list_available_models():
    """List all available model types."""
    register_builtin_models()
    print("Available baseline models:")
    for model_type in ModelRegistry.list_models():
        print(f"  - {model_type}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "list-models":
        list_available_models()
    else:
        main() 