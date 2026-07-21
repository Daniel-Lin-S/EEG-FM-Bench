#!/usr/bin/env python3
"""Traditional visualisation entry for models.

This entry point runs one of three visualisation workflows selected by
``vis_type``:

``"t_sne"``
    Extracts classifier features, optionally standardizes them and applies
    PCA, then embeds them in two dimensions with t-SNE. The output is one
    plot per dataset.
``"grad_cam"``
    Computes Grad-CAM values from model activations and gradients. Depending
    on ``GradCamVisArgs.grad_cam_target``, it produces a channel topomap (or
    bar plot fallback) or a temporal heatmap, together with optional
    per-sample and correctly-predicted class-average plots.
``"integrated_gradients"``
    Computes Captum Integrated Gradients wrapped in a Noise Tunnel. Depending
    on ``IntegratedGradientsVisArgs.ig_target``, attributions are reduced over
    time for a channel topomap (or bar plot fallback) or over channels for a
    temporal heatmap, together with the same optional plot groups.

The ``model_config`` YAML is selected by its ``model_type`` field and merged
with the concrete configuration class registered in
``baseline.abstract.factory.ModelRegistry``. See
``baseline.abstract.config.AbstractConfig``, ``ModelRegistry`` and the
model-specific ``*Config`` classes in ``baseline/*/*_config.py`` to determine
which model fields can be set. ``plot_vis.load_model_config`` also forces
``data.batch_size`` to ``1`` and clears ``model.pretrained_path``; use
``VisArgs.ckpt_path`` in the visualisation YAML to load a checkpoint.

The ``vis_config`` YAML is parsed into one of
``plot.utils.conf.TsneVisArgs``, ``plot.utils.conf.GradCamVisArgs`` or
``plot.utils.conf.IntegratedGradientsVisArgs``. See those classes for the
fields and defaults accepted for each ``vis_type``. The runtime behavior of
those fields is implemented by ``plot.utils.base_visualizer.BaseVisualizer``.

Usage:
    python plot_vis.py t_sne <model_config.yaml> <vis_config.yaml>
    python plot_vis.py grad_cam <model_config.yaml> <vis_config.yaml>
    python plot_vis.py integrated_gradients <model_config.yaml> \
        <vis_config.yaml>
"""

import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

from baseline.abstract.config import AbstractConfig
from baseline.abstract.factory import ModelRegistry
from baseline.utils.common import seed_torch
from baseline.utils.run_artifacts import (
    load_final_checkpoint,
    load_saved_run_config,
)
from common.log import setup_log
from common.path import get_conf_file_path
from common.utils import setup_yaml
from plot.baseline_visualizer import BaselineVisualizer
from plot.utils.conf import load_vis_conf_dict, TsneVisArgs, GradCamVisArgs, IntegratedGradientsVisArgs

logger = logging.getLogger()


def load_model_config(config_path: str) -> AbstractConfig:
    """Load baseline model config from YAML."""
    config_path = get_conf_file_path(config_path)
    file_cfg = OmegaConf.load(config_path)
    specific_model_type = str(file_cfg.get('model_type', ''))

    if specific_model_type not in ModelRegistry.list_models():
        raise ValueError(
            f"Unsupported model_type '{specific_model_type}'. "
            f"Supported baseline models: {', '.join(ModelRegistry.list_models())}"
        )

    config_class = ModelRegistry.get_config_class(specific_model_type)
    code_cfg = OmegaConf.create(config_class().model_dump())

    cfg = OmegaConf.merge(code_cfg, file_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    cfg = config_class.model_validate(cfg)

    logger.info('change batch_size forcefully to 1')
    cfg.data.batch_size = 1

    logger.info('change pretrained model path to none')
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'pretrained_path'):
        cfg.model.pretrained_path = None

    return cfg


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="Traditional visualization for baseline models")
    parser.add_argument("vis_type", choices=["t_sne", "grad_cam", "integrated_gradients"])
    parser.add_argument("model_config", nargs="?",
                        help="Path to baseline model config YAML")
    parser.add_argument("--run-dir", default=None,
                        help="Saved baseline run artifact directory")
    parser.add_argument("vis_config", help="Path to visualization config yaml")
    args = parser.parse_args()
    if bool(args.model_config) == bool(args.run_dir):
        parser.error('provide exactly one of model_config or --run-dir')

    setup_log()
    setup_yaml()

    logger.info(f"Starting {args.vis_type} visualization")
    logger.info(f"Model config: {args.model_config or args.run_dir}")
    logger.info(f"Visualization config: {args.vis_config}")

    if not Path(args.vis_config).exists():
        raise FileNotFoundError(f"Visualization config file not found: {args.vis_config}")

    model_config = None

    if args.vis_type == 't_sne':
        vis_config: TsneVisArgs = load_vis_conf_dict(args.vis_config, args.vis_type)
        model_config.model.t_sne = True
    elif args.vis_type == 'grad_cam':
        vis_config: GradCamVisArgs = load_vis_conf_dict(args.vis_config, args.vis_type)
        model_config.model.grad_cam = True
        model_config.model.grad_cam_target = vis_config.grad_cam_target
    else:
        vis_config: IntegratedGradientsVisArgs = load_vis_conf_dict(args.vis_config, args.vis_type)

    if args.run_dir:
        dataset_names = list(vis_config.datasets)
        if len(dataset_names) != 1:
            parser.error('--run-dir visualization requires exactly one dataset')
        dataset_name = dataset_names[0]
        model_config = load_saved_run_config(args.run_dir, dataset_name)
        vis_config.ckpt_path = str(
            load_final_checkpoint(args.run_dir, dataset_name)
        )
        model_config.data.batch_size = 1
        if hasattr(model_config.model, 'pretrained_path'):
            model_config.model.pretrained_path = None
    else:
        model_config = load_model_config(args.model_config)
    logger.info(f'visualization config {vis_config}')
    logger.info(f'target model config {model_config}')

    model_config.data.datasets = vis_config.datasets

    seed_torch(vis_config.seed)
    visualizer = BaselineVisualizer(model_config, vis_config)
    visualizer.run()

    logger.info(f"{args.vis_type} visualization completed successfully")



if __name__ == "__main__":
    main()