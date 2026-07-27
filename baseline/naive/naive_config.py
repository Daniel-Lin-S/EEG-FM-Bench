"""Configuration for label-only majority-vote benchmarking.

Inputs are configured dataset identifiers and their fixed split names. The
baseline reads only integer label columns from the processed datasets. Outputs
are deterministic metric artifacts below ``logging.run_dir/log/baseline``.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from baseline.abstract.config import AbstractConfig, BaseDataArgs
from baseline.abstract.config import BaseLoggingArgs


class NaiveDataArgs(BaseDataArgs):
    """Dataset selection settings for the label-only baseline."""


class NaiveModelArgs(BaseModel):
    """Fixed classifier behavior recorded in the resolved configuration.

    Parameters
    ----------
    tie_breaker : {"seeded_uniform"}, optional
        Rule used only when multiple classes have the same largest training
        count. Default: ``"seeded_uniform"``.
    """

    tie_breaker: Literal["seeded_uniform"] = "seeded_uniform"


class NaiveTrainingArgs(BaseModel):
    """Empty namespace retained for the standard baseline configuration."""


class NaiveLoggingArgs(BaseLoggingArgs):
    """Artifact settings for the non-checkpointing naive baseline."""

    experiment_name: str = "naive"
    project: Optional[str] = "naive"


class NaiveConfig(AbstractConfig):
    """Configuration for one seed-scoped majority-class evaluation.

    The classifier has no trainable parameters. It counts training labels and
    applies a constant prediction to the validation and test splits.
    """

    model_type: str = "naive"
    multitask: bool = False
    data: NaiveDataArgs = Field(default_factory=NaiveDataArgs)
    model: NaiveModelArgs = Field(default_factory=NaiveModelArgs)
    training: NaiveTrainingArgs = Field(default_factory=NaiveTrainingArgs)
    logging: NaiveLoggingArgs = Field(default_factory=NaiveLoggingArgs)

    def validate_config(self) -> bool:
        """Validate the label-only single-seed evaluation constraints."""
        if self.multitask:
            raise ValueError("naive supports only multitask=false.")
        if not self.data.datasets:
            raise ValueError("naive requires at least one configured dataset.")
        if len(self.seeds) != 1:
            raise ValueError(
                "naive supports exactly one seed because it is deterministic "
                "except for seeded exact-majority ties."
            )
        return True
