"""
Abstract configuration base class for baseline models.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ClassifierHeadType(str, Enum):
    """Available classifier-head implementations.

    ``"avg_pool"`` pools all time-channel features before an MLP;
    ``"attention_pool"`` uses a learned query to pool them; and
    ``"dual_stream_fusion"`` pools time and channels with cross-attention.
    ``"flatten_mlp"`` and ``"flatten_linear"`` instead require fixed dataset
    shape metadata and flatten every feature before classification.
    """
    AVG_POOL = "avg_pool"                      # Adaptive average pooling (current default)
    ATTENTION_POOL = "attention_pool"          # Attention pooling with learnable query
    DUAL_STREAM_FUSION = "dual_stream_fusion"  # Dual stream attention fusion
    FLATTEN_MLP = "flatten_mlp"                # Flatten + Large
    FLATTEN_LINEAR = "flatten_linear"          # Flatten + single linear


class ClassifierHeadConfig(BaseModel):
    """Settings for the dataset-specific classifier applied to encoder features.

    Parameters
    ----------
    head_type : ClassifierHeadType
        Selects the classifier implementation. Options are ``"avg_pool"``
        (global time-channel average pooling then MLP), ``"attention_pool"``
        (learned-query pooling then MLP), ``"dual_stream_fusion"``
        (time/channel cross-attention then MLP), ``"flatten_mlp"`` (fixed
        shape-aware three-layer MLP), and ``"flatten_linear"`` (fixed
        shape-aware linear classifier).
        Note: flatten-based heads use dataset shape.
        FLATTEN_MLP: (n_ch * n_patches * dim) -> (n_patches * dim) -> dim -> n_class
        FLATTEN_LINEAR: (n_ch * n_patches * dim) -> n_class
    hidden_dims : list[int]
        Hidden widths of the MLP used by classifier heads that include an MLP.
    dropout : float
        Drop probability used by classifier-head layers that support dropout.
    attn_n_head : int
        Number of heads in the learned-query attention-pooling head.
    attn_head_dim : int
        Per-head projection width in the attention-pooling head.
    fusion_mode : {"time_first", "channel_first", "dual"}
        Attention order used by the dual-stream fusion head. ``"time_first"``
        pools time within each channel before pooling channels;
        ``"channel_first"`` does the reverse; ``"dual"`` runs both branches
        and fuses their pooled features.
    fusion_n_head : int
        Number of attention heads in each fusion stream.
    fusion_head_dim : int
        Per-head projection width in fusion attention.
    fusion_use_rope : bool
        Whether fusion attention applies rotary positional embeddings.
    fusion_rope_theta : float
        Rotary embedding frequency base used when ``fusion_use_rope`` is true.
    fusion_max_seq_len : int
        Initial sequence length cached for fusion rotary embeddings; the cache
        expands automatically for longer inputs.
    """
    head_type: ClassifierHeadType = ClassifierHeadType.AVG_POOL

    # Common parameters
    hidden_dims: list[int] = Field(default_factory=lambda: [128])
    dropout: float = 0.3

    # Attention Pool parameters (for ATTENTION_POOL type)
    attn_n_head: int = 4
    attn_head_dim: int = 64

    # Dual Stream Fusion parameters (for DUAL_STREAM_FUSION type)
    fusion_mode: str = "dual"  # "time_first", "channel_first", or "dual"
    fusion_n_head: int = 4
    fusion_head_dim: int = 64
    fusion_use_rope: bool = True
    fusion_rope_theta: float = 10000.0
    fusion_max_seq_len: int = 1024


class BaseDataArgs(BaseModel):
    """Shared dataset and data-loader settings.

    Parameters
    ----------
    datasets : dict[str, str]
        Mapping from benchmark dataset names to dataset configuration names.
        The trainer uses it to construct single-dataset or multitask loaders.
    batch_size : int
        Number of samples requested from each data loader per batch.
    num_workers : int
        Number of worker processes used by each PyTorch data loader.
    """
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 32
    num_workers: int = 2


class BaseModelArgs(BaseModel):
    """Shared model-loading, analysis, and classifier-head settings.

    Parameters
    ----------
    pretrained_path : str or None
        Optional model-specific pretrained checkpoint location. Each baseline
        loader defines the required files and loading behavior for this path.
    grad_cam : bool
        Whether model wrappers retain activations needed by Grad-CAM analysis.
    t_sne : bool
        Whether the classifier exposes features for its t-SNE analysis path.
    grad_cam_target : str
        Target label used by model-specific Grad-CAM utilities when selecting
        features or axes to analyze. There is no shared fixed option set;
        accepted values are defined by the model-specific Grad-CAM consumer.
    classifier_head : ClassifierHeadConfig
        Architecture settings for the dataset-specific classification head.
    """
    pretrained_path: Optional[str] = None

    grad_cam: bool = False
    t_sne: bool = False
    grad_cam_target: str = 'channel'

    # Classifier head configuration
    classifier_head: ClassifierHeadConfig = Field(default_factory=ClassifierHeadConfig)

class BaseLoRAArgs(BaseModel):
    """Low-rank adaptation settings applied by the shared trainer.

    Parameters
    ----------
    use_lora : bool
        Whether to replace eligible linear layers with LoRA adapters and train
        adapters plus classifier heads instead of the full encoder.
    lora_r : int
        Rank of each low-rank update.
    lora_alpha : int
        LoRA scaling numerator; the effective adapter scale is ``alpha / r``.
    lora_dropout : float
        Drop probability applied to adapter inputs during training.
    lora_target_modules : list[str]
        Module-name patterns used to select layers for adaptation. Use the
        sole value ``["default"]`` to select the preset named by
        ``lora_target_type``; otherwise provide explicit model-module
        patterns, which take precedence over the preset.
    lora_exclude_modules : list[str] or None
        Optional module-name patterns excluded after target selection.
    lora_target_type : {"default", "full", "attention", "ffn"}
        Predefined target-module family used when resolving LoRA layers.
        ``"default"`` selects the model's recommended layers; ``"full"``
        selects its attention and feed-forward layers; ``"attention"``
        selects attention projections only; ``"ffn"`` selects feed-forward
        projections only.
    lora_scope : {"transformer", "full"}
        Limits injection after module-pattern matching. ``"transformer"``
        restricts it to transformer blocks; ``"full"`` permits all supported
        matching layers in the model.
    lora_lr_scale : float
        Multiplier applied to the classifier learning rate for LoRA parameters.
    """
    use_lora: bool = False
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 16  # LoRA scaling factor (effective scaling = alpha/r)
    lora_dropout: float = 0.0  # Dropout for LoRA layers
    lora_target_modules: List[str] = Field(default_factory=lambda: ["default"])  # Target module patterns
    lora_exclude_modules: Optional[List[str]] = None  # Modules to exclude from LoRA
    lora_target_type: str = "default"  # Predefined target type: "default", "full", "attention", "ffn"
    lora_scope: str = "transformer"  # Scope: "transformer" (only in Transformer blocks) or "full" (all layers)
    lora_lr_scale: float = 1.0  # Learning rate scale for LoRA parameters relative to head_lr


class BaseTrainingArgs(BaseModel):
    """Shared optimizer, schedule, precision, and adaptation settings.

    Parameters
    ----------
    max_epochs : int
        Number of complete passes over each training loader.
    weight_decay : float
        AdamW decoupled weight-decay coefficient.
    max_grad_norm : float
        Maximum gradient norm used by the training loop for clipping.
    lr_schedule : {"onecycle", "cosine"}
        Learning-rate schedule. ``"onecycle"`` uses PyTorch ``OneCycleLR``
        and ``pct_start``; ``"cosine"`` uses linear warmup followed by
        ``CosineAnnealingLR`` with ``min_lr``.
    max_lr : float
        Learning rate for classifier parameters and the base rate used to
        derive encoder and LoRA rates.
    encoder_lr_scale : float
        Multiplier applied to ``max_lr`` for trainable encoder parameters.
    warmup_epochs : int
        Number of loader-length epochs in cosine-schedule warmup.
    warmup_scale : float
        Initial rate fraction during cosine-schedule warmup.
    pct_start : float
        Fraction of total OneCycleLR steps spent increasing the rate.
    min_lr : float
        Minimum learning rate reached by cosine annealing.
    use_amp : bool
        Whether automatic mixed precision and gradient scaling are enabled.
    freeze_encoder : bool
        Whether non-classifier encoder parameters are excluded from updates.
    lora : BaseLoRAArgs
        LoRA targeting and optimization settings.
    """
    max_epochs: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    lr_schedule: str = "onecycle"  # 'onecycle' or 'cosine'
    max_lr: float = 1e-4
    encoder_lr_scale: float = 0.1
    warmup_epochs: int = 5
    warmup_scale: float = 1e-2
    pct_start: float = 0.2  # For OneCycleLR
    min_lr: float = 1e-6  # For CosineAnnealingLR

    use_amp: bool = True
    freeze_encoder: bool = False

    # LoRA configuration
    lora: BaseLoRAArgs = Field(default_factory=BaseLoRAArgs)


class BaseLoggingArgs(BaseModel):
    """Shared run-directory, experiment-tracking, and checkpoint settings.

    Parameters
    ----------
    experiment_name : str
        Name used to identify the training run in local and cloud logs.
    run_dir : str
        Root directory used by the trainer for run artifacts and checkpoints.
    use_cloud : bool
        Whether to initialize cloud experiment tracking.
    cloud_backend : {"wandb", "comet", "both"}
        Cloud logging backend selected by the trainer. ``"wandb"`` logs only
        to Weights & Biases, ``"comet"`` only to Comet, and ``"both"``
        initializes and logs to both services.
    project : str or None
        Optional project name passed to the cloud backend.
        By default, the trainer uses the model type as the project name.
    entity : str or None
        Optional account or team namespace passed to the cloud backend.
    api_key : str or None
        Optional credential supplied to the cloud logging backend.
    offline : bool
        Whether cloud tracking is initialized in offline mode.
        Default: False
    tags : list[str], optional
        Labels attached to the cloud experiment.
    log_step_interval : int
        Number of optimizer steps between training-metric log events.
        Default: 1
    use_tensorboard : bool
        Whether to write scalar training and evaluation metrics to TensorBoard
        event files in ``<run log directory>/tensorboard``. Default: False.
    ckpt_interval : int
        Number of completed epochs between checkpoint saves.
    outputs : list[{'log', 'tensorboard', 'csv'}]
        Local artifacts to persist. ``'csv'`` writes metric event traces,
        ``'log'`` writes one console log file per invocation, and
        ``'tensorboard'`` writes TensorBoard event files. Default: ``['csv']``.
    """
    experiment_name: str = "baseline"
    run_dir: str = "assets/run"

    use_cloud: bool = False
    cloud_backend: str = "wandb"
    project: Optional[str] = None
    entity: Optional[str] = None

    api_key: Optional[str] = None
    offline: bool = False
    tags: List[str] = Field(default_factory=lambda: [])

    log_step_interval: int = 1
    ckpt_interval: int = 1
    use_tensorboard: bool = False

    outputs: List[Literal['log', 'tensorboard', 'csv']] = Field(
        default_factory=lambda: ['csv']
    )

    @field_validator('outputs')
    @classmethod
    def validate_outputs(
        cls,
        outputs: List[Literal['log', 'tensorboard', 'csv']],
    ) -> List[Literal['log', 'tensorboard', 'csv']]:
        """Validate requested local artifact types."""
        if not outputs:
            raise ValueError('logging.outputs must contain at least one trace type.')
        if len(outputs) != len(set(outputs)):
            raise ValueError('logging.outputs must not contain duplicate trace types.')
        return outputs

class AbstractConfig(BaseModel, ABC):
    """Top-level configuration shared by all baseline trainers.

    Parameters
    ----------
    seed : int
        Random seed supplied to shared training and data-loading setup.
    master_port : int
        Preferred port used when initializing distributed training workers.
    multitask : bool
        Whether the trainer uses one shared model across configured datasets;
        model-specific trainers may interpret this setting further.
    model_type : str
        Registry identifier used to select the model trainer and artifact names.
    conf_file : str or None
        Optional source configuration-file reference retained with the config.
    fs : int
        Sampling frequency supplied to dataset shape and preprocessing helpers.
    data : BaseDataArgs
        Dataset mapping and data-loader settings.
    model : BaseModelArgs
        Model-loading, analysis, and classifier-head settings.
    training : BaseTrainingArgs
        Optimizer, scheduler, precision, freezing, and LoRA settings.
    logging : BaseLoggingArgs
        Run-artifact, cloud-tracking, and checkpoint settings.
    """
    
    seed: int = 42
    master_port: int = 41216
    multitask: bool = False
    model_type: str = "base"  # To identify which model is being used
    conf_file: Optional[str] = None
    fs: int = 256
    
    data: BaseDataArgs = Field(default_factory=BaseDataArgs)
    model: BaseModelArgs = Field(default_factory=BaseModelArgs)
    training: BaseTrainingArgs = Field(default_factory=BaseTrainingArgs)
    logging: BaseLoggingArgs = Field(default_factory=BaseLoggingArgs)

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate model-specific configuration requirements."""
        pass
