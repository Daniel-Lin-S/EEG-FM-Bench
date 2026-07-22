"""Search-space application and validation helpers.

Inputs are resolved model configuration dictionaries and validated HpoConfig
objects. Outputs are independent sampled dictionaries; the input mapping is
never mutated.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Mapping, Protocol, Type

from pydantic import BaseModel

from baseline.hpo.config import HpoConfig, SearchDistribution


class TrialProtocol(Protocol):
    """Subset of the Optuna trial interface used by this module."""

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        """Sample a floating-point value."""

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
        log: bool = False,
    ) -> int:
        """Sample an integer value."""

    def suggest_categorical(
        self,
        name: str,
        choices: list[str],
    ) -> str:
        """Sample an encoded categorical choice."""


def get_dotted_value(config: Mapping[str, Any], path: str) -> Any:
    """Return a leaf addressed by a dotted mapping path."""
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(
                f"Search path '{path}' does not exist in the model config."
            )
        current = current[part]
    return current


def set_dotted_value(config: Dict[str, Any], path: str, value: Any) -> None:
    """Replace one existing leaf addressed by a dotted mapping path."""
    parts = path.split(".")
    current: Dict[str, Any] = config
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise ValueError(
                f"Search path '{path}' does not address a mapping leaf."
            )
        current = child
    if parts[-1] not in current:
        raise ValueError(
            f"Search path '{path}' does not exist in the model config."
        )
    current[parts[-1]] = copy.deepcopy(value)


def _sample_distribution(
    trial: TrialProtocol,
    path: str,
    distribution: SearchDistribution,
) -> Any:
    """Sample and decode one configured distribution."""
    if distribution.distribution == "float":
        return trial.suggest_float(
            path,
            float(distribution.low),
            float(distribution.high),
            step=distribution.step,
            log=distribution.log,
        )
    if distribution.distribution == "int":
        step = 1 if distribution.step is None else int(distribution.step)
        return trial.suggest_int(
            path,
            int(distribution.low),
            int(distribution.high),
            step=step,
            log=distribution.log,
        )

    encoded_choices = [
        f"choice_{index:04d}"
        for index in range(len(distribution.choices or []))
    ]
    encoded = trial.suggest_categorical(path, encoded_choices)
    choice_index = encoded_choices.index(encoded)
    return copy.deepcopy(distribution.choices[choice_index])


def sample_config(
    base_config: Mapping[str, Any],
    hpo_config: HpoConfig,
    trial: TrialProtocol,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Return a sampled configuration and decoded parameter mapping."""
    sampled = copy.deepcopy(dict(base_config))
    decoded: Dict[str, Any] = {}
    for path, distribution in hpo_config.search_space.items():
        value = _sample_distribution(trial, path, distribution)
        set_dotted_value(sampled, path, value)
        decoded[path] = copy.deepcopy(value)
    return sampled, decoded


def _search_bounds(
    base_config: Mapping[str, Any],
    hpo_config: HpoConfig,
    path: str,
) -> tuple[float, float]:
    """Return the inclusive numeric bounds for one fixed or searched path."""
    distribution = hpo_config.search_space.get(path)
    if distribution is None:
        value = get_dotted_value(base_config, path)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Expected numeric value at '{path}'.")
        return float(value), float(value)
    if distribution.distribution == "categorical":
        choices = distribution.choices or []
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            for value in choices
        ):
            raise ValueError(
                f"Categorical learning-rate path '{path}' requires only "
                "numeric choices."
            )
        values = [float(value) for value in choices]
        return min(values), max(values)
    if distribution.low is None or distribution.high is None:
        raise ValueError(f"Numeric search path '{path}' has no bounds.")
    return float(distribution.low), float(distribution.high)


def validate_search_space(
    base_config: Mapping[str, Any],
    hpo_config: HpoConfig,
    config_class: Type[BaseModel],
) -> None:
    """Validate paths, representative values, and LR range compatibility."""
    for path, distribution in hpo_config.search_space.items():
        original = get_dotted_value(base_config, path)
        candidates: list[Any]
        if distribution.distribution == "categorical":
            candidates = list(distribution.choices or [])
        else:
            candidates = [distribution.low, distribution.high]

        for value in candidates:
            candidate = copy.deepcopy(dict(base_config))
            set_dotted_value(candidate, path, value)
            try:
                config_class.model_validate(candidate)
            except Exception as exc:
                raise ValueError(
                    f"Search value {value!r} for '{path}' is incompatible "
                    "with the model configuration."
                ) from exc

        if distribution.distribution == "categorical":
            continue
        if isinstance(original, bool):
            raise ValueError(
                f"Numeric search path '{path}' targets a boolean value."
            )

    _, min_lr_high = _search_bounds(
        base_config, hpo_config, "training.min_lr"
    )
    max_lr_low, _ = _search_bounds(
        base_config, hpo_config, "training.max_lr"
    )
    if min_lr_high > max_lr_low:
        raise ValueError(
            "training.min_lr may exceed training.max_lr: expected "
            "min_lr.high <= max_lr.low."
        )


def objective_values(
    metrics_by_dataset: Mapping[str, Mapping[str, float]],
    metric: str,
) -> Dict[str, float]:
    """Extract one finite validation metric for every dataset."""
    values: Dict[str, float] = {}
    suffix = f"/eval/{metric}"
    for dataset_name, metrics in metrics_by_dataset.items():
        matches = [
            value
            for key, value in metrics.items()
            if key == f"{dataset_name}{suffix}"
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected validation metric '{dataset_name}{suffix}', but "
                f"found {len(matches)} matches."
            )
        value = float(matches[0])
        if not math.isfinite(value):
            raise ValueError(
                f"Validation objective for {dataset_name} is not finite: "
                f"{value}."
            )
        values[dataset_name] = value
    if not values:
        raise ValueError("Validation produced no dataset objective values.")
    return values


def reduce_objective(
    values: Mapping[str, float],
    reduction: str,
    train_sizes: Mapping[str, int],
) -> float:
    """Combine per-dataset validation objectives."""
    if reduction == "macro_mean":
        return sum(values.values()) / len(values)
    if reduction != "log_train_size_weighted":
        raise ValueError(f"Unknown multitask objective reduction: {reduction}.")

    weights: Dict[str, float] = {}
    for dataset_name in values:
        size = train_sizes.get(dataset_name)
        if size is None or size <= 1:
            raise ValueError(
                f"Expected training size > 1 for {dataset_name}, but got "
                f"{size}."
            )
        weights[dataset_name] = math.log(size)
    total_weight = sum(weights.values())
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError(
            f"Expected positive finite objective weight, got {total_weight}."
        )
    return sum(
        values[name] * weights[name]
        for name in values
    ) / total_weight
