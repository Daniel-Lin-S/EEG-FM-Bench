"""Pydantic configuration models for traditional visualisation workflows.

Input files are YAML mappings. ``load_vis_conf_dict`` merges a YAML mapping
with the defaults of the class selected by ``vis_type`` and validates the
result. The resulting object is passed to
``plot.utils.base_visualizer.BaseVisualizer``. Shared fields are declared by
``VisArgs``; workflow-specific fields are declared by
``TsneVisArgs``, ``GradCamVisArgs`` and ``IntegratedGradientsVisArgs``.
"""

import logging
from typing import Optional, Union

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, Field


logger = logging.getLogger("plot_vis")


class VisArgs(BaseModel):
    """Shared settings accepted by every traditional visualisation config.

    Parameters
    ----------
    ckpt_path : str
        Optional checkpoint loaded by ``BaseVisualizer.load_checkpoint``. An
        empty string skips checkpoint loading.
    output_dir : str
        Root directory for timestamped visualisation output and the dumped
        copy of the resolved visualisation config.
    tag : list[str]
        Labels stored in the dumped visualisation config. They are not used
        to select datasets, models or output paths.
    seed : int
        Seed passed to the entry point's ``seed_torch`` call and used as the
        t-SNE ``random_state``.
    split : str
        Dataset split passed to the dataloader. Supported values are
        ``"train"``, ``"valid"`` and ``"test"``.
    model_type : str
        Baseline model registry identifier used by
        ``BaselineVisualizer.build_model``. Set it to the same identifier as
        ``model_config.model_type`` so the visualizer builds the configured
        model trainer.
    datasets : dict[str, str]
        Mapping from dataset names to dataset configuration names. The entry
        point also assigns this mapping to ``model_config.data.datasets``;
        these are the datasets processed by the visualizer.
    """
    ckpt_path: str = ''
    output_dir: str = ''
    tag: list[str] = Field(default_factory=lambda: [])
    seed: int = 42

    split: str = 'test'
    model_type: str = 'baseline'
    datasets: dict[str, str] = Field(default_factory=lambda: {})

    def dump_to_yaml(self, path: Optional[str ] =None, sort_keys: bool = False):
        conf = self.model_dump()
        conf_yaml = yaml.dump(
            conf,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=sort_keys
        )

        logger.info('Config is as follows in this run:')
        logger.info(conf_yaml)

        if path is not None:
            with open(path, 'w') as f:
                f.write(conf_yaml)


class IntegratedGradientsVisArgs(VisArgs):
    """Settings for the ``integrated_gradients`` visualisation workflow.

    Parameters
    ----------
    n_steps : int
        Number of integration steps passed to Captum ``IntegratedGradients``.
    baseline_type : str
        Baseline used for the input tensor. ``"zero"`` creates zeros,
        ``"random"`` samples uniformly in ``[-150, 150)``, and
        ``"gaussian"`` samples using the input tensor's mean and standard
        deviation.
    ig_target : str
        Attribution display axis. ``"channel"`` averages attributions over
        time for a channel topomap (or bar plot fallback); ``"temporal"``
        averages over channels for a temporal heatmap. This does not select
        the Captum target class; the workflow uses ``labels`` for that.
    noise_tunnel_type : str
        Captum Noise Tunnel aggregation passed as ``nt_type``. Supported
        values are ``"smoothgrad"``, ``"smoothgrad_sq"`` and ``"vargrad"``.
    noise_tunnel_samples : int
        Number of Noise Tunnel samples passed as ``nt_samples``.
    noise_tunnel_stdevs : float
        Noise standard deviation passed as ``stdevs`` to Captum Noise Tunnel.
    num_batch : int
        Maximum number of batches processed per dataset.
    generate_class_average : bool
        Whether to collect correctly-predicted samples and save one averaged
        attribution visualisation per class.
    generate_per_sample : bool
        Whether to save an attribution visualisation for each processed
        sample.
    """
    # IntegratedGradients parameters
    n_steps: int = 50
    baseline_type: str = 'random'  # 'zero', 'random', 'gaussian'
    ig_target: str = 'channel'  # channel or temporal
    
    # NoiseTunnel parameters
    noise_tunnel_type: str = 'smoothgrad'  # 'smoothgrad', 'smoothgrad_sq', 'vargrad'
    noise_tunnel_samples: int = 25
    noise_tunnel_stdevs: float = 0.2
    
    # Visualization parameters
    num_batch: int = 5
    generate_class_average: bool = True
    generate_per_sample: bool = True


class GradCamVisArgs(VisArgs):
    """Settings for the ``grad_cam`` visualisation workflow.

    Parameters
    ----------
    grad_cam_target : str
        Axis over which Grad-CAM values are aggregated. ``"channel"``
        averages over time and feature dimensions and creates a channel
        topomap (or bar plot fallback); ``"temporal"`` averages over channel
        and feature dimensions and creates a temporal heatmap. The entry
        point also copies this value to ``model_config.model.grad_cam_target``.
    num_batch : int
        Maximum number of batches processed per dataset.
    label_option : str
        Logit selected for Grad-CAM backpropagation. ``"pred"`` uses the
        predicted label and ``"truth"`` uses the ground-truth label. The
        value also determines the per-sample output directory label.
    generate_class_average : bool
        Whether to collect correctly-predicted samples and save one averaged
        Grad-CAM visualisation per class.
    generate_per_sample : bool
        Whether to save a Grad-CAM visualisation for each processed sample.
    """
    grad_cam_target: str = 'channel' # channel or temporal
    num_batch: int = 5
    label_option: str = 'pred' # pred or truth
    generate_class_average: bool = True
    generate_per_sample: bool = True


class TsneVisArgs(VisArgs):
    """Settings for the ``t_sne`` visualisation workflow.

    Parameters
    ----------
    num_batch : int
        Maximum number of batches from each dataset used for feature
        extraction.
    perplexity : int
        t-SNE perplexity used for every dataset except ``"workload"`` and
        ``"tusl"``.
    small_perplexity : int
        t-SNE perplexity used specifically for the ``"workload"`` and
        ``"tusl"`` datasets.
    use_pca : bool
        Whether to standardize the extracted features and reduce them with
        PCA before t-SNE.
    pca_dims : int
        Maximum number of PCA components when ``use_pca`` is true. The
        workflow also limits this value to the number of feature samples.
    max_iter : int
        Number of t-SNE optimization iterations passed to scikit-learn.
    """
    num_batch: int = 500
    perplexity: int = 30
    small_perplexity: int = 10
    use_pca: bool = False
    pca_dims: int = 50
    max_iter: int = 1000


def load_vis_conf_dict(config_path, vis_type: str) -> Union[TsneVisArgs, GradCamVisArgs, IntegratedGradientsVisArgs]:
    file_cfg = OmegaConf.load(config_path)
    # Backward-compatible field naming
    if vis_type == 'grad_cam':
        if 'target_layer_name' in file_cfg and 'grad_cam_target' not in file_cfg:
            file_cfg['grad_cam_target'] = file_cfg['target_layer_name']
    if vis_type == 't_sne':
        config_class = TsneVisArgs
    elif vis_type == 'grad_cam':
        config_class = GradCamVisArgs
    elif vis_type == 'integrated_gradients':
        config_class = IntegratedGradientsVisArgs

    else:
        raise ValueError(f'Unknown vis_type: {vis_type}')

    code_cfg = OmegaConf.create(config_class().model_dump())
    merged_config = OmegaConf.merge(code_cfg, file_cfg)
    cfg_dict = OmegaConf.to_container(merged_config, resolve=True, throw_on_missing=True)
    cfg = config_class.model_validate(cfg_dict)

    return cfg
