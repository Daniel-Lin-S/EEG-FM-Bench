"""Validated configuration for Optuna-backed baseline optimization.

The top-level YAML hpo mapping is parsed separately from model-specific
configuration. Search-space keys are dotted paths into the resolved model
configuration; sampled values replace only those leaves.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


PROTECTED_SEARCH_PATHS = frozenset({
    "data.datasets",
    "logging.experiment_name",
    "logging.run_dir",
})
PROTECTED_SEARCH_PREFIXES = (
    "hpo",
    "seeds",
    "master_port",
    "model_type",
    "multitask",
    "conf_file",
    "fs",
)


class SearchDistribution(BaseModel):
    """One numeric or categorical Optuna distribution.

    Parameters
    ----------
    distribution : {"float", "int", "categorical"}
        Optuna suggestion method used for this parameter. This field has no
        default and must be configured explicitly.
    low : float or None, optional, default=None
        Inclusive lower bound for a float or integer distribution. It must be
        provided for numeric distributions and omitted for categorical ones.
    high : float or None, optional, default=None
        Inclusive upper bound for a float or integer distribution. It must be
        greater than ``low`` and omitted for categorical distributions.
    step : float or None, optional, default=None
        Numeric increment between candidate values. ``None`` uses Optuna's
        continuous float behavior or an integer step of one. A step cannot be
        combined with logarithmic sampling.
    log : bool, optional, default=False
        Whether numeric values are sampled on a logarithmic scale. Logarithmic
        sampling requires a positive ``low`` and cannot be used with ``step``.
    choices : list[Any] or None, optional, default=None
        Unique values available to a categorical distribution. Values may be
        nested lists, such as classifier hidden-layer dimensions. Choices are
        required for categorical distributions and forbidden for numeric ones.
    """

    distribution: Literal["float", "int", "categorical"]
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    log: bool = False
    choices: Optional[List[Any]] = None

    @model_validator(mode="after")
    def validate_distribution(self) -> "SearchDistribution":
        """Validate fields required by the selected distribution."""
        if self.distribution == "categorical":
            if not self.choices:
                raise ValueError(
                    "Categorical distributions require nonempty choices."
                )
            encoded = [
                json.dumps(choice, sort_keys=True)
                for choice in self.choices
            ]
            if len(encoded) != len(set(encoded)):
                raise ValueError(
                    "Categorical distribution choices must be unique."
                )
            if any(
                value is not None
                for value in (self.low, self.high, self.step)
            ) or self.log:
                raise ValueError(
                    "Categorical distributions cannot define low, high, "
                    "step, or log."
                )
            return self

        if self.low is None or self.high is None:
            raise ValueError(
                f"{self.distribution} distributions require low and high."
            )
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError("Distribution bounds must be finite.")
        if self.low >= self.high:
            raise ValueError(
                f"Expected low < high, but got {self.low} >= {self.high}."
            )
        if self.log and self.low <= 0:
            raise ValueError(
                "Logarithmic distributions require a positive lower bound."
            )
        if self.step is not None:
            if not math.isfinite(self.step) or self.step <= 0:
                raise ValueError(
                    "Distribution step must be finite and positive."
                )
            if self.log:
                raise ValueError(
                    "A numeric distribution cannot use both step and log."
                )
        if self.choices is not None:
            raise ValueError(
                "Numeric distributions cannot define categorical choices."
            )
        if self.distribution == "int":
            values = (self.low, self.high, self.step)
            if any(
                value is not None and not float(value).is_integer()
                for value in values
            ):
                raise ValueError(
                    "Integer distribution bounds and step must be integers."
                )
        return self


class HpoObjectiveArgs(BaseModel):
    """Validation objective and multitask reduction.

    Parameters
    ----------
    metric : str, optional, default="loss"
        Suffix of the validation metric emitted as
        ``<dataset>/eval/<metric>`` and optimized by the study.
    direction : {"minimize", "maximize"}, optional, default="minimize"
        Whether smaller or larger objective values represent better trials.
    multitask_reduction : {"macro_mean", "log_train_size_weighted"}, optional,
        default="macro_mean"
        Method used to combine per-dataset validation values in a multitask
        study. ``macro_mean`` weights datasets equally;
        ``log_train_size_weighted`` uses normalized logarithmic training-set
        sizes.
    """

    metric: str = "loss"
    direction: Literal["minimize", "maximize"] = "minimize"
    multitask_reduction: Literal[
        "macro_mean",
        "log_train_size_weighted",
    ] = "macro_mean"


class TpeSamplerArgs(BaseModel):
    """Tree-structured Parzen estimator sampler settings.

    Parameters
    ----------
    type : {"tpe"}, optional, default="tpe"
        Sampler implementation. Only Optuna TPE is currently supported.
    seed : int, optional, default=0
        Non-negative random seed used by the TPE sampler. This is independent
        of both top-level evaluation seeds and the HPO training seed.
    n_startup_trials : int, optional, default=10
        Number of non-negative initial trials sampled before TPE begins using
        its fitted probability models.
    """

    type: Literal["tpe"] = "tpe"
    seed: int = Field(default=0, ge=0)
    n_startup_trials: int = Field(default=10, ge=0)


class MedianPrunerArgs(BaseModel):
    """Median-pruner settings expressed in training epochs.

    Parameters
    ----------
    type : {"median"}, optional, default="median"
        Pruner implementation. Only Optuna's median pruner is supported.
    n_startup_trials : int, optional, default=5
        Number of completed trials required before any trial may be pruned.
    n_warmup_epochs : int, optional, default=3
        Number of initial epochs in each trial during which pruning is
        disabled.
    interval_epochs : int, optional, default=1
        Positive number of epochs between successive pruning checks after the
        warmup period.
    """

    type: Literal["median"] = "median"
    n_startup_trials: int = Field(default=5, ge=0)
    n_warmup_epochs: int = Field(default=3, ge=0)
    interval_epochs: int = Field(default=1, ge=1)


class HpoConfig(BaseModel):
    """Top-level hyperparameter optimization settings.

    Parameters
    ----------
    enabled : bool, optional, default=False
        Whether to run HPO before ordinary evaluation training. Deterministic
        feature baselines ignore an enabled HPO configuration with a warning.
    seed : int, optional, default=0
        Non-negative training and data-loader seed used by every HPO trial.
        This value is independent of top-level evaluation seeds.
    n_trials : int or None, optional, default=None
        Positive complete-or-pruned trial target for each study. This value is
        required when ``enabled`` is true. Failed trials do not consume the
        target; their rows and artifacts are removed before SQLite resumption.
    max_consecutive_failed_trials : int, optional, default=5
        Positive terminal-failure streak that aborts an invalid study. A
        complete or pruned trial resets the streak.
    objective : HpoObjectiveArgs, optional, default=HpoObjectiveArgs()
        Validation metric, direction, and multitask reduction used to score
        trials.
    sampler : TpeSamplerArgs, optional, default=TpeSamplerArgs()
        Optuna TPE sampler configuration.
    pruner : MedianPrunerArgs, optional, default=MedianPrunerArgs()
        Optuna median-pruner configuration.
    search_space : dict[str, SearchDistribution], optional, default={}
        Mapping from dotted model-configuration paths to distributions. Only
        listed leaves are sampled; every unlisted resolved value stays fixed.
        Search paths must begin with ``data.``, ``model.``, or ``training.``
        and cannot target protected campaign fields.
    """

    enabled: bool = False
    seed: int = Field(default=0, ge=0)
    n_trials: Optional[int] = Field(default=None, ge=1)
    max_consecutive_failed_trials: int = Field(
        default=5,
        ge=1,
    )
    objective: HpoObjectiveArgs = Field(default_factory=HpoObjectiveArgs)
    sampler: TpeSamplerArgs = Field(default_factory=TpeSamplerArgs)
    pruner: MedianPrunerArgs = Field(default_factory=MedianPrunerArgs)
    search_space: Dict[str, SearchDistribution] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_search_space(self) -> "HpoConfig":
        """Validate enabled studies and protected configuration paths."""
        if self.enabled and self.n_trials is None:
            raise ValueError(
                "hpo.n_trials is required when HPO is enabled."
            )
        if self.enabled and not self.search_space:
            raise ValueError(
                "hpo.search_space must not be empty when HPO is enabled."
            )
        for path in self.search_space:
            if path in PROTECTED_SEARCH_PATHS or path.startswith(
                PROTECTED_SEARCH_PREFIXES
            ):
                raise ValueError(
                    f"Search path '{path}' targets a protected workflow field."
                )
            if not path.startswith(("data.", "model.", "training.")):
                raise ValueError(
                    f"Search path '{path}' must start with data., model., "
                    "or training."
                )
        return self
