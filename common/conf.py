import logging
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from common.path import create_parent_dir
from common.type import TrainStage, PretrainTaskListType, TemporalConvType, StdFactorType, SpectralType


class BasePreprocArgs(BaseModel):
    """Configuration accepted by :mod:`preproc` YAML files.

    Parameters
    ----------
    fs : int, optional, default=256
        Sampling rate in Hz supplied to every dataset builder. It defines the
        resampled dataset written to disk and must match the model's requirements
    clean_middle_cache : bool, optional, default=False
        Clears intermediate artifacts and dependent processed Arrow output before
        rebuilding each selected dataset.
    clean_shared_info : bool, optional, default=False
        Includes shared builder metadata in cache cleanup. Has no effect unless
        ``clean_middle_cache`` is enabled.
    num_preproc_arrow_writers : int, optional, default=4
        Process count passed to ``download_and_prepare`` to materialize the
        final Arrow dataset.
    num_preproc_mid_workers : int, optional, default=6
        Process count passed as ``n_proc`` to the builder's intermediate
        ``preproc`` stage.
    pretrain_datasets : list[str], optional, default=[]
        Dataset registry names to prepare with the builder configuration named
        ``pretrain``. Valid names are the keys of
        ``data.processor.wrapper.DATASET_SELECTOR`` (e.g., `tuab`).
    finetune_datasets : dict[str, str], optional, default={}
        Mapping whose keys use the same ``DATASET_SELECTOR`` registry above and
        whose values are configuration names accepted by that exact dataset's
        builder.
        Canonical choices are 'pretrain' and 'finetune',
        but this is dataset dependent.
        The authoritative choices are
        ``DATASET_SELECTOR[dataset_name].builder_configs.keys()`` (defined by
        the selected builder class in ``data/dataset``); they differ by dataset.
        ``preproc.py`` validates both the key and value.
    """

    fs: int = 256
    clean_middle_cache: bool = False
    clean_shared_info: bool = False
    # Recompute only these auxiliary field flows from cached recordings.
    # ``pos`` currently is the sole supported field flow.
    refresh_fields: list[str] = Field(default_factory=lambda: [])
    # Rebuild final Arrow artifacts from valid intermediate/field caches without
    # discarding the signal cache.
    refresh_arrow: bool = False
    num_preproc_arrow_writers: int = 4
    num_preproc_mid_workers: int = 6
    pretrain_datasets: list[str] = Field(default_factory=lambda: [])
    finetune_datasets: dict[str, str] = Field(default_factory=lambda: {})


class BaseEnvVars(BaseModel):
    """Serialized external-runtime environment settings.

    Notes
    -----
    No in-repository launcher copies this object to ``os.environ``. Each value
    is therefore retained in run YAML only; its external PyTorch/NCCL/W&B effect
    occurs only if a caller applies it.

    Parameters
    ----------
    TORCH_DISTRIBUTED_DEBUG : str, optional, default="INFO"
        PyTorch distributed diagnostic level.
    CUDA_LAUNCH_BLOCKING : str, optional, default="1"
        CUDA kernel-launch synchronization setting.
    TORCH_USE_CUDA_DSA : str, optional, default="1"
        CUDA device-side-assert setting for PyTorch.
    MKL_SERVICE_FORCE_INTEL : str, optional, default="GNU"
        MKL threading-runtime selection.
    OMP_NUM_THREADS : str, optional, default="1"
        OpenMP worker-thread limit.
    MKL_NUM_THREADS : str, optional, default="1"
        MKL worker-thread limit.
    ENABLE_INTRA_NODE_COMM : str, optional, default="1"
        Cluster-specific intra-node collective setting.
    NCCL_IB_TIMEOUT : str, optional, default="23"
        NCCL InfiniBand timeout exponent.
    NCCL_DEBUG : str, optional, default="INFO"
        NCCL diagnostic level.
    TORCH_NCCL_ASYNC_ERROR_HANDLING : str, optional, default="1"
        NCCL asynchronous-error policy.
    WANDB_CONSOLE : str, optional, default="off"
        W&B console-capture mode.
    """
    TORCH_DISTRIBUTED_DEBUG: str = "INFO"
    CUDA_LAUNCH_BLOCKING: str = "1"
    TORCH_USE_CUDA_DSA: str = "1"
    # Use GNU openMP (GOMP) instead of Intel OpenMP [Intel Math Kernel Library (MKL)]
    MKL_SERVICE_FORCE_INTEL: str = "GNU"
    OMP_NUM_THREADS: str = "1"
    MKL_NUM_THREADS: str = "1"
    # faster intra-node collectives, seems to be a cluster specific flag
    ENABLE_INTRA_NODE_COMM: str = "1"
    # avoids OOMs with long context
    # TORCH_NCCL_AVOID_RECORD_STREAMS: str = "1"
    # increasing NCCL timeout time before having some NCCL error 22 should give a 16s timeout
    NCCL_IB_TIMEOUT: str = "23"
    NCCL_DEBUG: str = "INFO"
    TORCH_NCCL_ASYNC_ERROR_HANDLING: str = "1"
    # wandb
    WANDB_CONSOLE: str = "off"
    # WANDB_API_KEY: str = WANDB_KEY


class BaseDataLoaderArgs(BaseModel):
    """Inputs used by the shared former-model data loader.

    Parameters
    ----------
    datasets : dict[str, str], optional, default={}
        Pretraining dataset registry mapping. Keys must be names in
        ``data.processor.wrapper.DATASET_SELECTOR``; see
        ``BasePreprocArgs.pretrain_datasets`` for the complete current list.
        ``create_pretrain_concat_loader`` uses only the keys and always selects
        each builder's ``pretrain`` configuration, so mapping values have no
        effect in that loader. The selected builder must provide ``pretrain``
        in its ``builder_configs`` mapping.
    batch_size : int, optional, default=32
        Same-montage records per batch before rank partitioning. It is reduced
        when there are too few records per rank.
    num_workers : int, optional, default=1
        DataLoader workers; positive values enable persistent workers, prefetch,
        and the ``spawn`` multiprocessing context.
    sample_ratio : float, optional, default=0.1
        Fraction retained from every montage group by the distributed sampler;
        ``1.0`` disables downsampling.
    """
    datasets: dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 32
    num_workers: int = 1
    sample_ratio: float = 0.1


class BaseDistRunArgs(BaseModel):
    """Distributed sampler and rendezvous configuration.

    Parameters
    ----------
    master_port : int, optional, default=41216
        Fixed port returned by ``get_master_port`` outside torchrun when random
        selection is disabled.
    is_port_random : bool, optional, default=False
        Uses a scheduler-job-seeded port in ``[20000, 60000]`` instead. Torchrun
        always takes ``MASTER_PORT`` from its environment.
    seed : int, optional, default=42
        Seed for ``DistributedGroupBatchSampler``; the epoch is added when its
        batch order is regenerated.
    debug : bool, optional, default=False
        Stored only; no current in-repository consumer.
    use_cpu : bool, optional, default=False
        Stored only; no current in-repository consumer.
    use_amp : bool, optional, default=True
        Stored only; no current in-repository consumer.
    deterministic : bool, optional, default=False
        Stored only; no current in-repository consumer.
    """
    master_port: int = 41216
    is_port_random: bool = False

    seed: int = 42
    debug: bool = False
    use_cpu: bool = False
    use_amp: bool = True
    deterministic: bool = False


class BaseCloudLogArgs(BaseModel):
    """Serialized cloud-run metadata for the former-model schema.

    Notes
    -----
    No current in-repository cloud logger consumes this class.

    Parameters
    ----------
    project : str or None, optional, default=None
        External project identifier.
    entity : str or None, optional, default=None
        External workspace/account identifier.
    id : str or None, optional, default=None
        External run identifier.
    name : str or None, optional, default=None
        External display name.
    notes : str or None, optional, default=None
        External run notes.
    tags : list[str] or None, optional, default=[]
        External run labels.
    job_type : str or None, optional, default="debug"
        External run category.
    mode : str or None, optional, default="online"
        External logger connection mode.
    """
    project: Optional[str] = None
    entity: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[list[str]] = Field(default_factory=lambda: [])
    job_type: Optional[str] = 'debug'
    mode: Optional[str] = 'online'


class BaseLogArgs(BaseModel):
    """Local output path and former-model logging controls.

    Parameters
    ----------
    run_dir : str, optional, default=""
        Root used by ``get_train_io_path``. Empty selects ``RUN_ROOT``; the
        helper creates ``log/former`` and ``ckpt/former`` under that root.
    log_train_step_interval : int, optional, default=10
        Stored training-log cadence; no current former trainer reads it.
    log_valid_step_interval : int, optional, default=5
        Stored validation-log cadence; no current former trainer reads it.
    ckpt_epoch_interval : int, optional, default=5
        Stored checkpoint epoch cadence; no current former trainer reads it.
    ckpt_step_ratio_interval_epoch : float, optional, default=1.0
        Stored within-epoch checkpoint ratio; currently unconsumed.
    save_scaling_ckpt : bool, optional, default=False
        Stored scaling-checkpoint switch; currently unconsumed.
    use_cloud : bool, optional, default=True
        Stored former cloud-logging switch; currently unconsumed.
    cloud : BaseCloudLogArgs, optional, default=BaseCloudLogArgs()
        Nested cloud metadata; Pydantic creates a fresh nested object.
    """
    run_dir: str = ''
    log_train_step_interval: int = 10
    log_valid_step_interval: int = 5

    ckpt_epoch_interval: int = 5
    ckpt_step_ratio_interval_epoch: float = 1.0
    save_scaling_ckpt: bool = False

    use_cloud: bool = True
    cloud: BaseCloudLogArgs =  Field(default_factory=BaseCloudLogArgs)


class BaseOptimArgs(BaseModel):
    """Serialized former-model pretraining optimization settings.

    Notes
    -----
    No current in-repository former trainer or optimizer reads these fields.

    Parameters
    ----------
    eps : float, optional, default=1e-8
        Optimizer numerical epsilon.
    epochs : int, optional, default=10
        Intended pretraining epoch count.
    warmup : int, optional, default=1
        Intended warmup epoch count.
    lr : float, optional, default=1e-4
        Intended base learning rate.
    weight_decay : float, optional, default=1e-2
        Intended decoupled weight regularization.
    clip : float, optional, default=1.0
        Intended global gradient-norm threshold.
    min_lr : float, optional, default=1e-5
        Intended scheduler floor.
    warmup_lr_factor : float, optional, default=0.1
        Intended initial learning-rate fraction during warmup.
    betas : list[float], optional, default=[0.9, 0.999]
        Intended Adam-like first- and second-moment coefficients.
    loss_scale_gpt : float, optional, default=0.34
        GPT loss multiplier.
    loss_scale_mae_tp : float, optional, default=0.33
        Temporal reconstruction loss multiplier.
    loss_scale_mae_ch : float, optional, default=0.33
        Channel reconstruction loss multiplier.
    pretrain_task_list_type : PretrainTaskListType, optional, default=ALL
        Pretraining task-combination selector. Accepted serialized values are
        ``"all"``, ``"gpt"``, and ``"mae"``, defined by
        ``common.type.PretrainTaskListType``.
    moe_seq_aux_global_coef : float, optional, default=1e-5
        Sequence-wise MoE auxiliary-loss multiplier.
    """
    eps: float = 1e-8
    epochs: int = 10
    warmup: int = 1

    lr: float = 1e-4
    weight_decay: float = 1e-2
    clip: float = 1.0
    min_lr: float = 1e-5
    warmup_lr_factor: float = 1e-1
    betas: list[float] = Field(default_factory=lambda: [0.9, 0.999])

    loss_scale_gpt: float = 0.34
    loss_scale_mae_tp: float = 0.33
    loss_scale_mae_ch: float = 0.33
    pretrain_task_list_type: PretrainTaskListType = PretrainTaskListType.ALL

    # DeepSeek V3 style MoE load balancing (auxiliary-loss-free via bias)
    # Note: bias is updated inside MoE layers based on load deviation.
    # These are trainer-level controls for the sequence-wise auxiliary loss.
    moe_seq_aux_global_coef: float = 1e-5  # Global scaling for seq aux loss (applied in trainer)


class BaseLoRAArgs(BaseModel):
    """Serialized low-rank adapter settings for the former model.

    Notes
    -----
    No current in-repository former model applies these adapters.

    Parameters
    ----------
    use_lora : bool, optional, default=False
        Switch for adapter insertion.
    lora_r : int, optional, default=8
        Adapter bottleneck width.
    lora_alpha : float, optional, default=16.0
        Adapter scaling numerator; conventional scale is ``alpha / r``.
    lora_dropout : float, optional, default=0.1
        Dropout probability on the adapter branch.
    lora_target_modules : list[str], optional, default=["proj_q", "proj_k", "proj_v", "proj_out", "linear_in", "linear_gate", "linear_out"]
        Query/key/value/output attention and input/gate/output feed-forward
        projection suffixes targeted by the intended adapter path.
    """
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = Field(
        default_factory=lambda: [
            'proj_q', 'proj_k', 'proj_v', 'proj_out',  # Attention layers
            'linear_in', 'linear_gate', 'linear_out'  # FFN layers (including MoE expert FFN)
        ]
    )


class BaseFinetuneArgs(BaseModel):
    """Former-model fine-tuning settings and loader dataset selection.

    Notes
    -----
    Except where noted, no current former trainer consumes these optimization
    and loss fields. This class is still used by fine-tuning loader helpers.

    Parameters
    ----------
    multitask_mode : bool, optional, default=False
        Stored former trainer mode selector; unconsumed by shared loaders.
    freeze_encoder : bool, optional, default=False
        Intended permanent encoder freeze; currently unconsumed.
    freeze_encoder_epochs : int, optional, default=0
        Intended initial frozen-encoder epoch count; currently unconsumed.
    batch_size : int, optional, default=32
        Stored only; shared loaders instead use ``data.batch_size``.
    sample_ratio : float, optional, default=1.0
        Stored only; shared loaders instead use ``data.sample_ratio``.
    epochs : int, optional, default=30
        Intended fine-tuning epoch count; currently unconsumed.
    warmup : int, optional, default=3
        Intended warmup epoch count; currently unconsumed.
    lr : float, optional, default=1e-4
        Intended classifier learning rate; currently unconsumed.
    head_lr_scale : float, optional, default=10.0
        Intended head-rate multiplier; currently unconsumed.
    fusion_lr_scale : float, optional, default=5.0
        Intended fusion-parameter rate multiplier; currently unconsumed.
    min_lr : float, optional, default=1e-5
        Intended schedule floor; currently unconsumed.
    clip : float, optional, default=5.0
        Intended gradient-norm threshold; currently unconsumed.
    warmup_lr_factor : float, optional, default=0.01
        Intended initial warmup rate fraction; currently unconsumed.
    with_reconstruct : bool, optional, default=True
        Intended reconstruction-loss switch; currently unconsumed.
    lambda_recon : float, optional, default=0.1
        Intended reconstruction-loss multiplier; currently unconsumed.
    label_smoothing : float, optional, default=0.1
        Intended target smoothing for cross-entropy; currently unconsumed.
    enable_moe_balance : bool, optional, default=True
        Intended MoE balancing-loss switch; currently unconsumed.
    moe_seq_aux_global_coef : float, optional, default=1e-5
        Intended sequence-wise MoE loss multiplier; currently unconsumed.
    contrast_loss_weight : float, optional, default=0.05
        Intended contrastive-loss multiplier; currently unconsumed.
    apply_loss_weight : bool, optional, default=True
        Stored only; the mixed loader does not consult this switch.
    loss_weight_type : str, optional, default="sqrt"
        Weighting rule passed to ``load_concat_eeg_datasets`` by the mixed
        fine-tuning loader. Accepted values are ``"statistics"`` (raw class
        counts), ``"sqrt"`` (sample count divided by square-root count),
        ``"log"`` (sample count divided by log count), and ``"absolute"``
        (sample count divided by count), as implemented by
        ``data.processor.wrapper.calc_distribution_weight``.
    loss_scale_per_dataset : bool, optional, default=False
        Intended per-dataset loss rescaling; currently unconsumed.
    log_train_step_interval : int, optional, default=10
        Intended training-log cadence; currently unconsumed.
    log_valid_step_interval : int, optional, default=5
        Intended validation-log cadence; currently unconsumed.
    ckpt_epoch_interval : int, optional, default=5
        Intended checkpoint epoch cadence; currently unconsumed.
    ckpt_step_ratio_interval_epoch : float, optional, default=1.0
        Intended within-epoch checkpoint ratio; currently unconsumed.
    checkpoint : str or None, optional, default=None
        Intended initialization checkpoint; currently unconsumed.
    datasets : dict[str, str], optional, default={}
        Registry-name to builder-configuration mapping used by the single,
        mixed, and per-dataset fine-tuning loader helpers. Keys must be in
        ``data.processor.wrapper.DATASET_SELECTOR`` (the complete current list
        is documented in ``BasePreprocArgs.pretrain_datasets``). Each value
        must be a key in ``DATASET_SELECTOR[dataset_name].builder_configs`` for
        that same dataset; inspect the selected builder class in
        ``data/dataset`` for its available configuration names.
    lora : BaseLoRAArgs, optional, default=BaseLoRAArgs()
        Nested former-model adapter settings.
    """
    multitask_mode: bool = False
    freeze_encoder: bool = False
    # Freeze encoder for the first N epochs, then unfreeze and continue finetuning.
    # - When freeze_encoder=True, the encoder stays frozen for all epochs (takes precedence).
    # - Ignored when LoRA is enabled (apply_lora will manage trainable params).
    freeze_encoder_epochs: int = 0

    batch_size: int = 32
    sample_ratio: float = 1.0

    epochs: int = 30
    warmup: int = 3
    lr: float = 1e-4
    head_lr_scale: float = 10.0
    fusion_lr_scale: float = 5.0
    min_lr: float = 1e-5
    clip: float = 5.0
    warmup_lr_factor: float = 1e-2

    with_reconstruct: bool = True
    lambda_recon: float = 0.1
    label_smoothing: float = 0.1

    enable_moe_balance: bool = True
    moe_seq_aux_global_coef: float = 1e-5

    # Contrastive learning loss weight (used when model.use_contrast=true)
    contrast_loss_weight: float = 0.05

    apply_loss_weight: bool = True
    loss_weight_type: str = 'sqrt'
    loss_scale_per_dataset: bool = False

    log_train_step_interval: int = 10
    log_valid_step_interval: int = 5

    ckpt_epoch_interval: int = 5
    ckpt_step_ratio_interval_epoch: float = 1.0

    checkpoint: Optional[str] = None
    datasets: dict[str, str] = Field(default_factory=lambda: {})

    lora: BaseLoRAArgs = Field(default_factory=BaseLoRAArgs)


class BaseClassifierHeaderArgs(BaseModel):
    """Serialized former-model classifier-head settings.

    Parameters
    ----------
    n_class : int, optional, default=2
        Placeholder class count expected to be replaced from dataset metadata;
        no current in-repository former model reads it.
    hidden_dims : list[int], optional, default=[64]
        Intended hidden classifier widths; currently unconsumed.
    mlp_dropout : float, optional, default=0.3
        Intended classifier MLP dropout probability; currently unconsumed.
    """
    # n_class will be automatically defined in runtime
    n_class: int = 2

    hidden_dims:  list[int] = Field(default_factory=lambda: [64])
    mlp_dropout: float = 0.3


class BaseStackConvArgs(BaseModel):
    """Serialized multiscale temporal-convolution layout.

    Notes
    -----
    No current in-repository former model builder consumes this class.

    Parameters
    ----------
    stack_out_channels : list[int], optional, default=[16, 24, 48, 32, 64]
        Intended output widths of successive convolution stages.
    stack_kernel_size : list[list[int]], optional, default=[[9, 5], [13, 9], [15, 11, 9], [33, 25, 17], [33, 25, 17, 9]]
        Intended parallel temporal kernel lengths for each stage.
    stack_stride : list[list[int]], optional, default=[[2, 1], [2, 2], [2, 2, 2], [2, 2, 1], [2, 2, 2, 1]]
        Intended stride of each corresponding convolution branch.
    """
    stack_out_channels: list[int] = Field(
        default_factory=lambda: [16, 24, 48, 32, 64])
    stack_kernel_size: list[list[int]] = Field(
        default_factory=lambda: [[9, 5], [13, 9], [15, 11, 9], [33, 25, 17], [33, 25, 17, 9]])
    stack_stride: list[list[int]] = Field(
        default_factory=lambda: [[2, 1], [2, 2], [2, 2, 2], [2, 2, 1], [2, 2, 2, 1]])


class BaseModelArgs(BaseModel):
    """Former-model architecture settings with compatibility validation.

    Notes
    -----
    In the current codebase only ``dim``, ``dim_temporal``, ``dim_fft``,
    ``head_dim``, ``n_head``, and ``f_embed`` are read by this class's
    validator. No former-model builder consumes the remaining values.

    Parameters
    ----------
    dim : int, optional, default=640
        Total feature width; validated against attention and spectral widths.
    dim_temporal : int, optional, default=512
        Temporal width used in spectral-feature validation.
    dim_fft : int, optional, default=128
        Spectral width used in spectral-feature validation.
    patch_size : int, optional, default=256
        Stored temporal patch length.
    max_rope_seq_len : int, optional, default=2304
        Stored rotary-position sequence cap.
    head_dim : int, optional, default=80
        Per-query-head width; ``head_dim * n_head`` must equal ``dim``.
    n_head : int, optional, default=8
        Query-head count used by the compatibility check.
    n_kv_head : int, optional, default=4
        Stored grouped key/value head count.
    n_layer : int, optional, default=8
        Stored transformer block count.
    mae_temporal_ratio : float, optional, default=0.5
        Stored temporal masked-reconstruction fraction.
    mae_channel_ratio : float, optional, default=0.5
        Stored channel masked-reconstruction fraction.
    multiple_of : int, optional, default=256
        Stored feed-forward width rounding multiple.
    ffn_dim_multiplier : int or None, optional, default=4
        Stored feed-forward width multiplier.
    moe_ffn_dim_multiplier : float or None, optional, default=1.0
        Stored MoE-expert width multiplier.
    norm_eps : float, optional, default=1e-8
        Stored normalization epsilon.
    rope_theta : float, optional, default=100000.0
        Stored rotary-position base.
    attn_dropout_rate : float, optional, default=0.1
        Stored attention dropout probability.
    ffn_dropout_rate : float, optional, default=0.1
        Stored feed-forward dropout probability.
    t_embed : TemporalConvType, optional, default=TemporalConvType.MULTISCALE
        Stored temporal embedding selector. Accepted serialized values are
        ``"stride"`` and ``"multiscale"``, defined by
        ``common.type.TemporalConvType``.
    stack_args : BaseStackConvArgs, optional, default=BaseStackConvArgs()
        Stored multiscale convolution layout.
    patch_kernel_size : int, optional, default=7
        Stored patch-convolution kernel length.
    patch_stride : int, optional, default=5
        Stored patch-convolution stride.
    f_embed : SpectralType, optional, default=SpectralType.STFT
        Spectral selector. Accepted serialized values are ``"fft"``,
        ``"stft"``, and ``"no"``, defined by ``common.type.SpectralType``.
        ``"fft"`` and ``"stft"`` require
        ``dim == dim_temporal + dim_fft``; ``"no"`` skips that validation.
    f_embed_fmax : float, optional, default=100.0
        Stored spectral frequency ceiling.
    stft_win_len : int, optional, default=160
        Stored STFT window length.
    stft_hop_len : int, optional, default=64
        Stored STFT frame hop length.
    stft_f_hidden : int, optional, default=16
        Stored spectral projection width.
    is_finetune : bool, optional, default=False
        Stored former-model mode flag.
    classifier_args : BaseClassifierHeaderArgs, optional, default=BaseClassifierHeaderArgs()
        Stored former-model classifier settings.
    std_base : float or None, optional, default=None
        Stored standard-deviation reference scale.
    std_factor : StdFactorType, optional, default=StdFactorType.DISABLED
        Stored standard-deviation adjustment mode. Accepted serialized values
        are ``"current_depth"``, ``"global_depth"``, ``"dim_ratio"``, and
        ``"disabled"``, defined by ``common.type.StdFactorType``.
    grad_cam : bool, optional, default=False
        Stored former-model Grad-CAM switch.
    t_sne : bool, optional, default=False
        Stored former-model t-SNE switch.
    """
    dim: int = 640
    dim_temporal: int = 512
    dim_fft: int = 128
    patch_size: int = 256
    max_rope_seq_len: int = 2304

    head_dim: int = 80
    n_head: int = 8
    n_kv_head: int = 4

    n_layer: int = 8

    mae_temporal_ratio: float = 0.5
    mae_channel_ratio: float = 0.5

    multiple_of: int = 256
    ffn_dim_multiplier: Optional[int] = 4
    moe_ffn_dim_multiplier: Optional[float] = 1.0
    norm_eps: float = 1e-8
    rope_theta: float = float(1e5)

    attn_dropout_rate: float = 0.1
    ffn_dropout_rate: float = 0.1

    t_embed: TemporalConvType = TemporalConvType.MULTISCALE
    stack_args: BaseStackConvArgs = Field(default_factory=BaseStackConvArgs)
    patch_kernel_size: int = 7
    patch_stride: int = 5

    f_embed: SpectralType = SpectralType.STFT
    f_embed_fmax: float = 100.0
    stft_win_len: int = 160
    stft_hop_len: int = 64
    stft_f_hidden: int = 16

    is_finetune: bool = False
    classifier_args: BaseClassifierHeaderArgs = Field(default_factory=BaseClassifierHeaderArgs)

    std_base: Optional[float] = None
    std_factor: StdFactorType = StdFactorType.DISABLED

    grad_cam: bool = False
    t_sne: bool = False

    @model_validator(mode='after')
    def check_dim_match(self):
        if self.f_embed != SpectralType.NO:
            if self.dim != self.dim_temporal + self.dim_fft:
                raise ValueError('Feature dims do not match')

        if self.dim != self.head_dim * self.n_head:
            raise ValueError('Head dims do not match')
        return self


class BaseSetupArgs(BaseModel):
    """Top-level former-model run configuration and YAML serializer.

    Parameters
    ----------
    model_type : str, optional, default="default"
        Stored implementation selector; no current former-model factory reads it.
    conf_file : str or None, optional, default=None
        Stored source path; ``dump_to_yaml`` serializes it but does not load it.
    stage : TrainStage, optional, default=TrainStage.PRETRAIN
        Selects loader helper validity: pretraining helpers require
        ``PRETRAIN`` and fine-tuning helpers require ``FINETUNE``. Accepted
        serialized values are ``"pretrain"``, ``"finetune"``, ``"eval"``, and
        ``"all"``, defined by ``common.type.TrainStage``; the latter two are
        valid enum values but are rejected by the shared pretraining and
        fine-tuning loader helpers.
    fs : int, optional, default=256
        Sampling rate passed to dataset loading and shape discovery; it must
        match the preprocessing rate.
    data : BaseDataLoaderArgs, optional, default=BaseDataLoaderArgs()
        Shared pretraining loader settings.
    model : BaseModelArgs, optional, default=BaseModelArgs()
        Former-model architecture settings and compatibility validation.
    optim : BaseOptimArgs, optional, default=BaseOptimArgs()
        Serialized former-pretraining optimizer/loss settings.
    ft : BaseFinetuneArgs, optional, default=BaseFinetuneArgs()
        Fine-tuning loader datasets and former-model settings.
    env : BaseEnvVars, optional, default=BaseEnvVars()
        Serialized external-runtime environment settings.
    dist : BaseDistRunArgs, optional, default=BaseDistRunArgs()
        Distributed sampler seed and rendezvous settings.
    log : BaseLogArgs, optional, default=BaseLogArgs()
        Local output root and former-model logging settings.

    Notes
    -----
    ``dump_to_yaml`` logs the Pydantic dump and, when given a path, creates the
    parent directory before writing the YAML document.
    """
    model_type: str = 'default'
    conf_file: Optional[str] = None
    stage: TrainStage = TrainStage.PRETRAIN
    # Global sampling rate for data loading (must match preprocessed data)
    fs: int = 256

    data: BaseDataLoaderArgs = Field(default_factory=BaseDataLoaderArgs)
    model: BaseModelArgs = Field(default_factory=BaseModelArgs)
    optim: BaseOptimArgs = Field(default_factory=BaseOptimArgs)
    ft: BaseFinetuneArgs = Field(default_factory=BaseFinetuneArgs)

    env: BaseEnvVars = Field(default_factory=BaseEnvVars)
    dist: BaseDistRunArgs = Field(default_factory=BaseDistRunArgs)
    log: BaseLogArgs = Field(default_factory=BaseLogArgs)

    def dump_to_yaml(self, path: Optional[str]=None, sort_keys: bool = False):
        conf = self.model_dump()
        conf_yaml = yaml.dump(
            conf,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=sort_keys
        )

        log = logging.getLogger()
        log.info('Config is as follows in this run:')
        log.info(conf_yaml)

        if path is not None:
            create_parent_dir(path)
            with open(path, 'w') as f:
                f.write(conf_yaml)
