"""Assess whether an Optuna study should stop or receive another block.

Inputs are completed trial objectives and their epoch-level histories.
Outputs distinguish flat, separated, and responsive-unresolved studies.
No campaign artifacts or Optuna state are modified by this module.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from baseline.hpo.config import ProgressiveHpoArgs


FLAT_OUTCOME = "flat"
SEPARATED_OUTCOME = "separated"
UNRESOLVED_OUTCOME = "responsive_unresolved"
INSUFFICIENT_OUTCOME = "insufficient_completed_trials"


@dataclass(frozen=True)
class ProgressiveAssessment:
    """One progressive-allocation decision and its diagnostics."""

    outcome: str
    should_expand: bool
    completed_trials: int
    resolution_threshold: float
    between_trial_sd: float | None
    winner_gap: float | None
    incumbent_stable: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible diagnostic mapping."""
        return asdict(self)


def _linear_residuals(values: Sequence[float]) -> tuple[list[float], float]:
    """Return linear-fit residuals and slope for one objective history."""
    if len(values) < 2:
        return [0.0 for _ in values], 0.0
    center = (len(values) - 1) / 2.0
    mean_value = statistics.fmean(values)
    denominator = sum(
        (index - center) ** 2
        for index in range(len(values))
    )
    if denominator <= 0.0:
        return [0.0 for _ in values], 0.0
    slope = sum(
        (index - center) * (value - mean_value)
        for index, value in enumerate(values)
    ) / denominator
    intercept = mean_value - slope * center
    residuals = [
        value - (intercept + slope * index)
        for index, value in enumerate(values)
    ]
    return residuals, slope


def _history_values(
    trial: Mapping[str, Any],
    residual_epochs: int,
) -> list[float]:
    """Return finite final objective values from one completed trial."""
    raw_history = trial.get("objective_history", [])
    if not isinstance(raw_history, Sequence):
        return []
    values: list[float] = []
    for item in raw_history:
        if not isinstance(item, Mapping):
            continue
        value = item.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            values.append(numeric)
    return values[-residual_epochs:]


def has_complete_progressive_evidence(
    completed_trials: Sequence[Mapping[str, Any]],
    args: ProgressiveHpoArgs,
) -> bool:
    """Return whether every budgeted trial has usable saved evidence.

    Parameters
    ----------
    completed_trials : sequence of mappings
        Completed trials in chronological order.
    args : ProgressiveHpoArgs
        Progressive history settings.

    Returns
    -------
    bool
        Whether every trial has a finite objective and at least one saved
        epoch objective.
    """
    if not completed_trials:
        return True
    for trial in completed_trials:
        objective = trial.get("objective")
        if (
            isinstance(objective, bool)
            or not isinstance(objective, (int, float))
            or not math.isfinite(float(objective))
        ):
            return False
        if not _history_values(trial, args.residual_epochs):
            return False
    return True


def progressive_assessment_block(
    completed_trials: Sequence[Mapping[str, Any]],
    completed_count: int,
    args: ProgressiveHpoArgs,
) -> list[Mapping[str, Any]] | None:
    """Return the latest completed allocation block at a decision boundary.

    Parameters
    ----------
    completed_trials : sequence of mappings
        Completed trials in chronological order.
    completed_count : int
        Number of selectable completed trials in the study.
    args : ProgressiveHpoArgs
        Initial and incremental allocation sizes.

    Returns
    -------
    list of mappings or None
        First block at ``initial_trials`` or the latest incremental block at
        a later completed-trial boundary; otherwise ``None``.

    Raises
    ------
    ValueError
        If ``completed_count`` is invalid for ``completed_trials``.
    """
    if completed_count < 0:
        raise ValueError(
            "Expected a non-negative completed-trial count, got "
            f"{completed_count}."
        )
    if completed_count != len(completed_trials):
        raise ValueError(
            "Completed-trial count does not match collected trials: "
            f"{completed_count} != {len(completed_trials)}."
        )
    if completed_count < args.initial_trials:
        return None
    if completed_count == args.initial_trials:
        start = 0
    else:
        completed_after_initial = completed_count - args.initial_trials
        if completed_after_initial % args.increment_trials:
            return None
        start = completed_count - args.increment_trials
    return list(completed_trials[start:completed_count])


def _resolution_threshold(
    trials: Sequence[Mapping[str, Any]],
    args: ProgressiveHpoArgs,
) -> float:
    """Estimate the practical objective resolution after linear detrending."""
    residuals: list[float] = []
    for trial in trials:
        values = _history_values(trial, args.residual_epochs)
        trial_residuals, _ = _linear_residuals(values)
        residuals.extend(trial_residuals)
    residual_sd = statistics.stdev(residuals) if len(residuals) >= 2 else 0.0
    return max(
        args.minimum_resolution,
        args.noise_multiplier * residual_sd,
    )


def _incumbent_is_stable(
    trial: Mapping[str, Any],
    args: ProgressiveHpoArgs,
    threshold: float,
) -> bool:
    """Return whether fitted late-objective drift stays within resolution."""
    values = _history_values(trial, args.residual_epochs)
    if len(values) < args.residual_epochs:
        return False
    _, slope = _linear_residuals(values)
    total_drift = abs(slope) * (len(values) - 1)
    return total_drift <= threshold


def assess_progressive_study(
    completed_trials: Sequence[Mapping[str, Any]],
    direction: str,
    args: ProgressiveHpoArgs,
) -> ProgressiveAssessment:
    """Classify a completed-trial set for progressive allocation.

    Parameters
    ----------
    completed_trials : sequence of mappings
        Each mapping contains finite ``objective`` and ``objective_history``.
    direction : {"minimize", "maximize"}
        Objective ordering used to select the incumbent.
    args : ProgressiveHpoArgs
        Resolution and history settings.

    Returns
    -------
    ProgressiveAssessment
        Stop/expand decision with finite diagnostics where estimable.
    """
    if direction not in {"minimize", "maximize"}:
        raise ValueError(f"Unsupported objective direction: {direction}.")
    clean_trials = [
        trial
        for trial in completed_trials
        if isinstance(trial.get("objective"), (int, float))
        and not isinstance(trial.get("objective"), bool)
        and math.isfinite(float(trial["objective"]))
    ]
    threshold = _resolution_threshold(clean_trials, args)
    if len(clean_trials) < 2:
        return ProgressiveAssessment(
            outcome=INSUFFICIENT_OUTCOME,
            should_expand=True,
            completed_trials=len(clean_trials),
            resolution_threshold=threshold,
            between_trial_sd=None,
            winner_gap=None,
            incumbent_stable=False,
        )

    objectives = [float(trial["objective"]) for trial in clean_trials]
    between_sd = statistics.stdev(objectives)
    ordered = sorted(
        clean_trials,
        key=lambda trial: float(trial["objective"]),
        reverse=direction == "maximize",
    )
    winner_gap = abs(
        float(ordered[0]["objective"])
        - float(ordered[1]["objective"])
    )
    stable = _incumbent_is_stable(ordered[0], args, threshold)
    if between_sd <= threshold:
        outcome = FLAT_OUTCOME
        should_expand = False
    elif winner_gap > threshold and stable:
        outcome = SEPARATED_OUTCOME
        should_expand = False
    else:
        outcome = UNRESOLVED_OUTCOME
        should_expand = True
    return ProgressiveAssessment(
        outcome=outcome,
        should_expand=should_expand,
        completed_trials=len(clean_trials),
        resolution_threshold=threshold,
        between_trial_sd=between_sd,
        winner_gap=winner_gap,
        incumbent_stable=stable,
    )
