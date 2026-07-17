#!/usr/bin/env python3
"""Unified baseline training entry point and YAML reference.

``conf_file`` (``str``, CLI only) identifies the YAML to load. ``model_type``
(``str``) must be supplied in that YAML or on the command line; it selects the
registered config model and trainer, and a CLI value takes precedence. Defaults,
then YAML values, then CLI values are merged. Fields absent from the selected
Pydantic schema are ignored, so options copied from a different baseline do not
alter the run.

Requirements and defaults
-------------------------
``conf_file`` is **required** on the command line; there is no fallback YAML.
``model_type`` is **required after merging**: provide it in the YAML or on the
command line. Every other field documented below is **optional**. When absent,
this entry point first creates ``config_class().model_dump()`` for the selected
model and retains that field's Pydantic default; an example YAML's value is not
the default. Nested mappings (``data``, ``model``, ``training``, ``logging``,
``classifier_head``, and ``lora``) likewise default to their selected schema's
new nested configuration object. Model-specific defaults can differ, so the
relevant ``*_config.py`` class is the authoritative value source.

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

``data`` mapping
----------------
* ``datasets`` (``dict[str, str]``): dataset registry name to prepared builder
  configuration. This chooses the splits, labels, montage, and shapes from
  which loaders and classifier heads are created.
* ``batch_size`` (``int``): samples per process in a loader batch; it also
  sizes distributed result-gather buffers.
* ``num_workers`` (``int``): PyTorch DataLoader worker processes.
* ``patch_size`` (``int``, CSBrain): temporal patch size in samples passed to
  the CSBrain adapter.
* ``max_seq_len`` (``int | null``, CSBrain): adapter sequence-length cap;
  ``null`` permits dynamic length.

Common ``model`` mapping
------------------------
* ``pretrained_path`` (``str | null``): initializes the model-specific encoder
  from a checkpoint; ``null`` uses the architecture's random initialization.
  BENDR also accepts ``pretrained_conv_path`` (``str | null``) for its
  convolutional encoder checkpoint.
* ``grad_cam`` (``bool``): retain encoder activations for the Grad-CAM path.
* ``t_sne`` (``bool``): collect classifier features for the t-SNE path.
* ``grad_cam_target`` (``str``): retained by the base schema (default
  ``channel``), but the registered baseline trainers do not currently consume it.
* ``classifier_head`` (mapping): the per-dataset classifier after an encoder:
  ``head_type`` (``avg_pool | attention_pool | dual_stream_fusion |
  flatten_mlp | flatten_linear``) selects feature aggregation; ``hidden_dims``
  (``list[int]``) and ``dropout`` (``float``) configure applicable MLP heads.
  ``attn_n_head`` and ``attn_head_dim`` (``int``) apply only to
  ``attention_pool``. ``fusion_mode`` (``time_first | channel_first | dual``),
  ``fusion_n_head``/``fusion_head_dim`` (``int``), ``fusion_use_rope``
  (``bool``), ``fusion_rope_theta`` (``float``), and ``fusion_max_seq_len``
  (``int``) apply only to ``dual_stream_fusion``.

Common ``training`` mapping
---------------------------
* ``max_epochs`` (``int``): full passes over the training loader.
* ``weight_decay`` (``float``): decoupled AdamW regularization.
* ``max_grad_norm`` (``float``): global gradient-norm clipping threshold,
  applied after AMP gradients are unscaled.
* ``lr_schedule`` (``onecycle | cosine``): per-step schedule. ``cosine`` uses
  linear warmup then cosine annealing; REVE implements its own
  ``reduce_on_plateau`` schedule.
* ``max_lr`` (``float``): classifier learning rate and schedule peak/base.
* ``encoder_lr_scale`` (``float``): multiplier on ``max_lr`` for unfrozen
  encoder parameters; classifier parameters use ``max_lr``.
* ``warmup_epochs`` (``int``) and ``warmup_scale`` (``float``): duration and
  initial factor of linear warmup for the standard cosine schedule.
* ``pct_start`` (``float``): rising-phase fraction for OneCycleLR only.
* ``min_lr`` (``float``): cosine floor; REVE also uses it as warmup's start.
* ``use_amp`` (``bool``): enable PyTorch autocast and gradient scaling.
* ``freeze_encoder`` (``bool``): exclude encoder weights from optimization.
* ``label_smoothing`` (``float``, EEGPT/LABRAM/CSBrain): value passed to
  cross-entropy loss to soften one-hot targets.
* ``lora`` (mapping): train low-rank projections while freezing original
  encoder weights. ``use_lora`` (``bool``) enables it; ``lora_r``/``lora_alpha``
  (``int``) set rank and scaling numerator (effective scale ``alpha / r``);
  ``lora_dropout`` (``float``) is adapter-branch dropout. ``lora_target_modules``
  (``list[str]``) and ``lora_exclude_modules`` (``list[str] | null``) match
  modules; ``lora_target_type`` (``default | full | attention | ffn``) and
  ``lora_scope`` (``transformer | full``) select the matching region; and
  ``lora_lr_scale`` (``float``) multiplies the adapter learning rate.

``logging`` mapping
-------------------
* ``experiment_name`` (``str``): fallback cloud project name.
* ``run_dir`` (``str``): root for checkpoints and local cloud-log files.
* ``use_cloud`` (``bool``): initialize experiment logging on the master rank.
* ``cloud_backend`` (``wandb | comet | both``): logger(s) to initialize.
* ``project``/``entity``/``api_key`` (``str | null``), ``offline`` (``bool``),
  and ``tags`` (``list[str]``): cloud project/workspace/credential settings,
  W&B offline mode, and run labels. Prefer environment variables for keys.
* ``log_step_interval`` (``int``): steps between training-metric logs.
* ``ckpt_interval`` (``int``): epochs between checkpoint/evaluation saves.

Model-specific ``model`` fields
-------------------------------
* **EEGPT:** ``patch_size`` (``int``) and ``patch_stride`` (``int | null``)
  create temporal tokens. ``embed_num``, ``embed_dim``, ``depth``, and
  ``num_heads`` (``int``), plus ``mlp_ratio`` (``float``), set transformer
  capacity. ``dropout_rate``, ``attn_dropout_rate``, ``drop_path_rate``
  (``float``), ``init_std`` (``float``), and ``qkv_bias`` (``bool``) control
  regularization and attention initialization. ``use_channel_conv`` (``bool``)
  and ``conv_chan_dim`` (``int``) enable/size channel adaptation.
  ``linear_probe1_dim`` (``int``), ``linear_probe1_max_norm`` and
  ``linear_probe2_max_norm`` (``float``) parameterize its probe layers.
* **LABRAM:** ``eeg_size`` and ``patch_size`` (``int``) define fixed input and
  token size (the first must divide by the second); ``in_chans``/``out_chans``
  (``int``) configure channel projections. ``embed_dim``, ``depth``,
  ``num_heads`` (``int``), ``mlp_ratio`` (``float``), dropout fields,
  ``init_values``/``init_scale``/``layer_scale_init_value`` (``float``), and
  ``qkv_bias`` (``bool``) build and initialize the transformer.
  ``use_abs_pos_emb`` (``bool``) adds absolute positions;
  ``use_rel_pos_bias`` (``bool``) adds relative position bias;
  ``use_shared_rel_pos_bias`` (``bool``) shares that bias across layers; and
  ``use_mean_pooling`` (``bool``) selects final mean pooling. The accepted
  training fields ``weight_decay_end`` (``float | null``), ``layer_decay``
  (``float``), and ``model_ema``/``model_ema_decay``/``model_ema_force_cpu``
  (``bool, float, bool``) are not consumed by the current LABRAM trainer.
* **BENDR:** ``emb_dim`` (``int``) is convolutional feature width;
  ``conv_width``/``conv_stride`` (``list[int]``) define paired convolution
  layers; ``conv_drop_rate`` (``float``) and ``conv_proj_head`` (``bool``)
  configure that encoder. ``ffn_dim``, ``heads``, ``context_layers``,
  ``position_encoder`` (``int``), ``context_drop_rate``/``layer_drop``
  (``float``), and ``activation`` (``str``) configure contextualization.
  ``mask_p_t``/``mask_p_c`` (``float``) and ``mask_t_span`` (``int``) control
  temporal/channel masking; ``max_channels`` (``int``) limits input channels;
  ``finetuning`` (``bool``) enables contextualizer fine-tuning behavior.
* **BIOT:** ``emb_size``, ``heads``, ``depth``, ``max_channels`` (``int``)
  define the encoder; ``n_fft``/``hop_length`` (``int``) make STFT tokens; and
  ``use_channel_conv`` (``bool``) adapts input channels before encoding.
* **CBraMod:** ``in_dim``/``out_dim`` (``int``) are token-projection sizes;
  ``d_model``, ``dim_ffn``, ``n_layer``, ``n_head`` (``int``) define the
  transformer; ``dropout_rate`` (``float``) regularizes it.
* **CSBrain:** ``in_dim``/``out_dim`` (``int``) size temporal projections;
  ``d_model``, ``dim_feedforward``, ``n_layer``, ``nhead`` (``int``) define
  region-aware attention; ``tem_embed_kernel_sizes`` (``list[list[int]]``)
  supplies multiscale temporal convolution kernels.
* **REVE:** ``pos_bank_pretrained_path`` (``str | null``) loads the electrode
  position bank for its adapter. ``embed_dim``, ``depth``, ``heads``,
  ``head_dim`` (``int``), ``mlp_dim_ratio`` (``float``), ``use_geglu``
  (``bool``), ``freqs`` (``int``), ``noise_ratio`` (``float``),
  ``patch_size``/``patch_overlap`` (``int``), and ``dropout`` (``float``)
  configure its encoder, positional code, patching, and regularization. Its
  training-only options are ``optimizer_name`` (``adamw | stableadamw``),
  ``eps`` (``float``), ``warmup_freeze_encoder`` (``bool``),
  ``warmup_freeze_encoder_epochs`` (``int``), and ``adam_beta_1``/
  ``adam_beta_2`` (``float``), which select optimizer behavior and temporary
  encoder freezing during warmup.
* **BrainOmni:** ``position_montage`` (``str``), normalization booleans, and
  ``signal_normalize_eps``/``position_normalize_eps`` (``float``) configure
  input coordinates. ``strict_load`` and ``freeze_tokenizer`` (``bool``)
  control checkpoint compatibility and tokenizer training. Without a
  ``pretrained_path``, ``window_length``, ``n_filters``, kernel sizes,
  ``n_dim``/``n_head``/``n_neuro``, codebook and quantizer sizes, ``lm_dim``/
  ``lm_head``/``lm_depth``, and ``num_quantizers_used`` (``int``), plus
  ``ratios`` (``list[int]``), ``dropout``/``overlap_ratio``/``lm_dropout``/
  ``mask_ratio`` (``float``), ``rotation_trick`` (``bool``), and
  ``quantize_optimize_method`` (``str``), build its vendored tokenizer and
  latent transformer. ``repo_path`` is retained for loader compatibility.
* **EEGNet:** uses the classical schema; beyond common analysis/head fields,
  only ``pretrained_path`` is accepted in its ``model`` mapping.

The shipped LABRAM YAML's ``model_name`` is not a ``LabramModelArgs`` field and
is ignored; it does not select a checkpoint or architecture.

Usage:
    python baseline_main.py conf_file=assets/conf/baseline/eegpt/eegpt_unified.yaml
    python baseline_main.py conf_file=assets/conf/baseline/eegnet/eegnet.yaml model_type=eegnet
"""

import sys

from omegaconf import OmegaConf

from baseline.abstract.factory import ModelRegistry
from common.path import get_conf_file_path
from common.utils import setup_yaml


def main():
    """Main training function that can handle any registered baseline model."""
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
    print("Available baseline models:")
    for model_type in ModelRegistry.list_models():
        print(f"  - {model_type}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "list-models":
        list_available_models()
    else:
        main() 