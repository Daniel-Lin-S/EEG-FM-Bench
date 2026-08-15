"""
CBraMod Configuration that inherits from AbstractConfig.
"""

from typing import Dict, Optional, List
from pydantic import Field

from baseline.abstract.config import (
    AbstractConfig,
    BaseDataArgs,
    BaseLoggingArgs,
    BaseModelArgs,
    BaseTrainingArgs,
)


class CBraModDataArgs(BaseDataArgs):
    """CBraMod data-loader configuration.

    Shared dataset and data-loader parameters are documented in
    :class:`BaseDataArgs`.
    """
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 32
    num_workers: int = 2


class CBraModModelArgs(BaseModelArgs):
    """CBraMod architecture and regularization configuration.

    Shared model-loading, analysis, and classifier-head parameters are
    documented in :class:`BaseModelArgs`.

    Parameters
    ----------
    pretrained_path : str or None, optional, default=None
        Optional path to a PyTorch checkpoint containing CBraMod encoder
        weights. The trainer loads the checkpoint into the encoder with
        non-strict matching and warns about missing or unexpected keys. If
        omitted, training starts with a newly initialized encoder.
    in_dim : int, optional, default=200
        Number of input samples in each temporal patch.
    out_dim : int, optional, default=200
        Width of the encoder feature representation returned to the trainer.
    d_model : int, optional, default=200
        Transformer embedding width.
    dim_ffn : int, optional, default=800
        Hidden width of each transformer's feed-forward network.
    n_layer : int, optional, default=12
        Number of transformer encoder layers.
    n_head : int, optional, default=8
        Number of attention heads in each transformer layer.
    dropout_rate : float, optional, default=0.1
        Dropout probability used by the transformer encoder.
    """
    # Pretrained model path
    pretrained_path: Optional[str] = None

    # CBraMod architecture parameters
    in_dim: int = 200
    out_dim: int = 200
    d_model: int = 200
    dim_ffn: int = 800
    n_layer: int = 12
    n_head: int = 8
    
    # Regularization
    dropout_rate: float = 0.1


class CBraModTrainingArgs(BaseTrainingArgs):
    """CBraMod training defaults.

    Shared optimizer, schedule, precision, freezing, and adaptation
    parameters are documented in :class:`BaseTrainingArgs`.
    """
    max_epochs: int = 50

    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Learning rate schedule
    lr_schedule: str = "cosine"  # 'onecycle' or 'cosine'
    max_lr: float = 1e-4
    encoder_lr_scale: float = 0.1
    warmup_epochs: int = 5
    warmup_scale: float = 1e-2
    pct_start: float = 0.2  # For OneCycleLR
    min_lr: float = 1e-6  # For CosineAnnealingLR

    use_amp: bool = True
    freeze_encoder: bool = False


class CBraModLoggingArgs(BaseLoggingArgs):
    """CBraMod logging defaults.

    Shared run-artifact, cloud-tracking, and checkpoint parameters are
    documented in :class:`BaseLoggingArgs`.
    """
    experiment_name: str = "cbramod"
    run_dir: str = "assets/run"

    # Cloud logging options
    use_cloud: bool = True
    cloud_backend: str = "wandb"  # 'wandb', 'comet', or 'both'
    project: Optional[str] = "cbramod"
    entity: Optional[str] = None

    api_key: Optional[str] = None
    offline: bool = False
    tags: List[str] = Field(default_factory=lambda: [])

    # Logging intervals
    log_step_interval: int = 1
    ckpt_interval: int = 1


class CBraModConfig(AbstractConfig):
    """Top-level CBraMod configuration.

    Shared top-level parameters are documented in :class:`AbstractConfig`.
    Shared nested parameters are documented in :class:`BaseDataArgs`,
    :class:`BaseModelArgs`, :class:`BaseTrainingArgs`, and
    :class:`BaseLoggingArgs`.
    """
    model_type: str = "cbramod"
    
    data: CBraModDataArgs = Field(default_factory=CBraModDataArgs)
    model: CBraModModelArgs = Field(default_factory=CBraModModelArgs)
    training: CBraModTrainingArgs = Field(default_factory=CBraModTrainingArgs)
    logging: CBraModLoggingArgs = Field(default_factory=CBraModLoggingArgs)

    def validate_config(self) -> bool:
        """Validate CBraMod-specific configuration."""
        # Check model dimensions
        if self.model.d_model <= 0 or self.model.dim_ffn <= 0:
            return False
            
        # Check attention heads configuration
        if self.model.d_model % self.model.n_head != 0:
            return False
            
        # Check learning rate schedule
        if self.training.lr_schedule not in ["onecycle", "cosine"]:
            return False
            
        return True
