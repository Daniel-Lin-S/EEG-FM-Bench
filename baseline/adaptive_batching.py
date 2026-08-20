"""Select and record CUDA micro-batches for neural baseline training.

Inputs are a requested global update batch, distributed world size, CUDA
memory observations, and an optional previous OOM candidate. Outputs are an
exact per-rank micro-batch, accumulation count, and serializable memory
profile. This module does not create datasets or training artifacts.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn


BYTES_PER_OPTIMIZER_STATE_VALUE = 4
ADAMW_STATE_VALUES_PER_PARAMETER = 2
MEASURED_MEMORY_MODEL_VERSION = 2
FULL_MEMORY_MODE = "full"
HEAD_ONLY_MEMORY_MODE = "head_only"
LORA_MEMORY_MODE = "lora"


@dataclass(frozen=True)
class CudaMemoryLimit:
    """Resolved CUDA allocation ceiling for one process.

    Parameters
    ----------
    free_bytes : int
        Device memory free before the training model is allocated.
    total_bytes : int
        Total physical memory of the selected CUDA device.
    reserve_bytes : int
        Headroom retained from ``total_bytes``.
    external_bytes : int
        Approximate memory occupied outside the current PyTorch allocator.
    process_limit_bytes : int
        Maximum memory the current process may reserve.
    process_limit_fraction : float
        ``process_limit_bytes`` divided by ``total_bytes``.
    """

    free_bytes: int
    total_bytes: int
    reserve_bytes: int
    external_bytes: int
    process_limit_bytes: int
    process_limit_fraction: float

    def as_dict(self) -> dict[str, int | float]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def exact_divisors(value: int) -> list[int]:
    """Return positive divisors in descending order.

    Parameters
    ----------
    value : int
        Positive per-rank update batch.

    Returns
    -------
    list[int]
        Every exact micro-batch candidate, largest first.
    """
    if value <= 0:
        raise ValueError(
            f"Expected a positive update batch, but got {value}."
        )
    lower: list[int] = []
    upper: list[int] = []
    for candidate in range(1, math.isqrt(value) + 1):
        if value % candidate != 0:
            continue
        lower.append(candidate)
        paired = value // candidate
        if paired != candidate:
            upper.append(paired)
    return sorted(lower + upper, reverse=True)


def derive_batch_candidates(
    global_batch_size: int,
    world_size: int,
    adaptive: bool,
) -> list[int]:
    """Return exact per-rank micro-batch candidates.

    Parameters
    ----------
    global_batch_size : int
        Requested global samples per optimizer update.
    world_size : int
        Number of distributed ranks participating in training.
    adaptive : bool
        Whether smaller exact divisors may be attempted.

    Returns
    -------
    list[int]
        Per-rank candidates ordered from fastest to smallest.
    """
    if world_size <= 0:
        raise ValueError(
            f"Expected a positive world size, but got {world_size}."
        )
    if global_batch_size % world_size != 0:
        raise ValueError(
            "data.batch_size must be divisible by the distributed world "
            f"size, but got {global_batch_size} and {world_size}."
        )
    per_rank = global_batch_size // world_size
    if not adaptive:
        return [per_rank]
    return exact_divisors(per_rank)


def select_safe_micro_batch(
    candidates: list[int],
    fixed_bytes: int,
    calibration_peak_bytes: int,
    calibration_batch_size: int,
    process_limit_bytes: int,
    uncertainty_factor: float = 1.0,
) -> tuple[int, int]:
    """Select the largest candidate predicted to fit the process ceiling.

    Parameters
    ----------
    candidates : list[int]
        Exact per-rank batch divisors in descending order.
    fixed_bytes : int
        Analytical parameter, gradient, and optimizer-state memory.
    calibration_peak_bytes : int
        Reserved-memory peak measured by a disposable calibration update.
    calibration_batch_size : int
        Per-rank micro-batch used for calibration.
    process_limit_bytes : int
        Current allocator ceiling after external occupancy and headroom.
    uncertainty_factor : float, optional, default=1.0
        Conservative multiplier applied to sample-scaled memory.

    Returns
    -------
    tuple[int, int]
        Selected micro-batch and its predicted peak in bytes.
    """
    if not candidates:
        raise ValueError("Expected at least one micro-batch candidate.")
    if calibration_batch_size <= 0:
        raise ValueError(
            "Expected a positive calibration batch size, but got "
            f"{calibration_batch_size}."
        )
    if not math.isfinite(uncertainty_factor) or uncertainty_factor < 1.0:
        raise ValueError(
            "Expected uncertainty_factor >= 1, but got "
            f"{uncertainty_factor}."
        )
    variable_bytes = max(calibration_peak_bytes - fixed_bytes, 0)
    bytes_per_sample = (
        variable_bytes
        * uncertainty_factor
        / calibration_batch_size
    )
    for candidate in candidates:
        predicted = math.ceil(fixed_bytes + bytes_per_sample * candidate)
        if predicted <= process_limit_bytes:
            return candidate, predicted
    predicted_one = math.ceil(fixed_bytes + bytes_per_sample)
    return candidates[-1], predicted_one


def select_measured_micro_batch(
    candidates: list[int],
    observations: Sequence[Mapping[str, Any]],
    fixed_bytes: int,
    process_limit_bytes: int,
    uncertainty_factor: float = 1.0,
) -> tuple[int, int] | None:
    """Select a candidate from distinct measured reserved-memory peaks.

    Parameters
    ----------
    candidates : list[int]
        Exact per-rank batch divisors in descending order.
    observations : sequence of mappings
        Successful measurements containing ``micro_batch_size`` and
        ``measured_peak_reserved_bytes``.
    fixed_bytes : int
        Analytical lower bound for persistent training memory.
    process_limit_bytes : int
        Current allocator ceiling after occupancy and reserve handling.
    uncertainty_factor : float, optional, default=1.0
        Multiplier applied to the fitted sample-scaled memory slope.

    Returns
    -------
    tuple[int, int] or None
        Selected candidate and predicted reserved peak, or ``None`` when
        fewer than two distinct usable observations are available.
    """
    if not candidates:
        raise ValueError("Expected at least one micro-batch candidate.")
    if fixed_bytes < 0 or process_limit_bytes < 0:
        raise ValueError("CUDA memory byte counts cannot be negative.")
    if not math.isfinite(uncertainty_factor) or uncertainty_factor < 1.0:
        raise ValueError(
            "Expected uncertainty_factor >= 1, but got "
            f"{uncertainty_factor}."
        )

    peaks_by_batch: dict[int, int] = {}
    for index, observation in enumerate(observations):
        micro_batch = observation.get("micro_batch_size")
        peak_bytes = observation.get("measured_peak_reserved_bytes")
        if (
            isinstance(micro_batch, bool)
            or not isinstance(micro_batch, int)
            or micro_batch <= 0
            or isinstance(peak_bytes, bool)
            or not isinstance(peak_bytes, int)
            or peak_bytes <= 0
        ):
            raise ValueError(
                "Expected positive integer memory observation values at "
                f"index {index}, but got batch={micro_batch!r}, "
                f"peak={peak_bytes!r}."
            )
        peaks_by_batch[micro_batch] = max(
            peak_bytes,
            peaks_by_batch.get(micro_batch, 0),
        )
    if len(peaks_by_batch) < 2:
        return None

    points = sorted(peaks_by_batch.items())
    slopes = [
        (right_peak - left_peak) / (right_batch - left_batch)
        for left_index, (left_batch, left_peak) in enumerate(points)
        for right_batch, right_peak in points[left_index + 1:]
        if right_peak > left_peak
    ]
    if not slopes:
        return None
    bytes_per_sample = max(slopes)
    intercept = max(
        float(fixed_bytes),
        max(
            peak_bytes - bytes_per_sample * micro_batch
            for micro_batch, peak_bytes in points
        ),
    )
    for candidate in candidates:
        predicted = math.ceil(
            intercept
            + uncertainty_factor * bytes_per_sample * candidate
        )
        if predicted <= process_limit_bytes:
            return candidate, predicted
    predicted_one = math.ceil(
        intercept + uncertainty_factor * bytes_per_sample
    )
    return candidates[-1], predicted_one


def merge_memory_observation(
    observations: Sequence[Mapping[str, Any]],
    observation: Mapping[str, Any],
) -> list[dict[str, int]]:
    """Return conservative successful peaks indexed by micro-batch size."""
    merged: dict[int, dict[str, int]] = {}
    for index, candidate in enumerate([*observations, observation]):
        micro_batch = candidate.get("micro_batch_size")
        reserved = candidate.get("measured_peak_reserved_bytes")
        allocated = candidate.get("measured_peak_allocated_bytes")
        if (
            isinstance(micro_batch, bool)
            or not isinstance(micro_batch, int)
            or micro_batch <= 0
            or isinstance(reserved, bool)
            or not isinstance(reserved, int)
            or reserved <= 0
        ):
            raise ValueError(
                "Expected positive integer memory observation values at "
                f"index {index}, but got batch={micro_batch!r}, "
                f"peak={reserved!r}."
            )
        payload = {
            "micro_batch_size": micro_batch,
            "measured_peak_reserved_bytes": reserved,
        }
        if (
            not isinstance(allocated, bool)
            and isinstance(allocated, int)
            and allocated > 0
        ):
            payload["measured_peak_allocated_bytes"] = allocated
        elif allocated is not None:
            raise ValueError(
                "Expected a positive allocated-memory observation at "
                f"index {index}, but got {allocated!r}."
            )
        previous = merged.get(micro_batch)
        if previous is None or reserved > previous[
            "measured_peak_reserved_bytes"
        ]:
            merged[micro_batch] = payload
    return [merged[key] for key in sorted(merged)]


def remove_memory_observation(
    observations: Sequence[Mapping[str, Any]],
    micro_batch_size: int,
) -> list[dict[str, int]]:
    """Remove a candidate invalidated by an actual training OOM."""
    if (
        isinstance(micro_batch_size, bool)
        or not isinstance(micro_batch_size, int)
        or micro_batch_size <= 0
    ):
        raise ValueError(
            "Expected a positive micro_batch_size, but got "
            f"{micro_batch_size}."
        )
    retained = [
        observation
        for observation in observations
        if observation.get("micro_batch_size") != micro_batch_size
    ]
    if not retained:
        return []
    first, *remaining = retained
    merged = merge_memory_observation([], first)
    for observation in remaining:
        merged = merge_memory_observation(merged, observation)
    return merged


def resolve_training_memory_mode(
    freeze_encoder: bool,
    use_lora: bool,
) -> str:
    """Return the activation-memory policy after optimizer freezing."""
    if use_lora:
        return LORA_MEMORY_MODE
    if freeze_encoder:
        return HEAD_ONLY_MEMORY_MODE
    return FULL_MEMORY_MODE


def resolve_cuda_memory_limit(
    free_bytes: int,
    total_bytes: int,
    current_reserved_bytes: int,
    reserve_fraction: float,
) -> CudaMemoryLimit:
    """Resolve a process ceiling while preserving total-device headroom."""
    if total_bytes <= 0:
        raise ValueError(
            f"Expected positive total CUDA memory, but got {total_bytes}."
        )
    if free_bytes < 0 or current_reserved_bytes < 0:
        raise ValueError("CUDA memory observations cannot be negative.")
    if not 0.0 <= reserve_fraction < 1.0:
        raise ValueError(
            "memory_reserve_fraction must be in [0, 1), but got "
            f"{reserve_fraction}."
        )
    reserve_bytes = int(total_bytes * reserve_fraction)
    used_bytes = max(total_bytes - free_bytes, 0)
    external_bytes = max(used_bytes - current_reserved_bytes, 0)
    process_limit = total_bytes - reserve_bytes - external_bytes
    process_limit = max(process_limit, 0)
    fraction = process_limit / total_bytes
    return CudaMemoryLimit(
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        reserve_bytes=reserve_bytes,
        external_bytes=external_bytes,
        process_limit_bytes=process_limit,
        process_limit_fraction=fraction,
    )


def configure_cuda_allocator(
    device: torch.device,
    reserve_fraction: float,
) -> Optional[CudaMemoryLimit]:
    """Apply the resolved CUDA process ceiling on ``device``."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    current_reserved = torch.cuda.memory_reserved(device)
    limit = resolve_cuda_memory_limit(
        int(free_bytes),
        int(total_bytes),
        int(current_reserved),
        reserve_fraction,
    )
    if limit.process_limit_bytes <= current_reserved:
        raise torch.cuda.OutOfMemoryError(
            "CUDA memory outside the training process leaves no capacity "
            "while preserving the configured reserve."
        )
    torch.cuda.set_per_process_memory_fraction(
        limit.process_limit_fraction,
        device=device,
    )
    torch.cuda.reset_peak_memory_stats(device)
    return limit


def estimate_fixed_training_bytes(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, int]:
    """Estimate parameter, gradient, and optimizer-state memory."""
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    trainable_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    frozen_parameter_bytes = parameter_bytes - trainable_parameter_bytes
    trainable_elements = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    gradient_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    state_values = ADAMW_STATE_VALUES_PER_PARAMETER
    if optimizer.__class__.__name__.lower() not in {
        "adamw",
        "stableadamw",
    }:
        state_values = ADAMW_STATE_VALUES_PER_PARAMETER
    optimizer_bytes = (
        trainable_elements
        * state_values
        * BYTES_PER_OPTIMIZER_STATE_VALUE
    )
    return {
        "parameter_bytes": parameter_bytes,
        "trainable_parameter_bytes": trainable_parameter_bytes,
        "frozen_parameter_bytes": frozen_parameter_bytes,
        "gradient_bytes": gradient_bytes,
        "optimizer_state_bytes": optimizer_bytes,
        "estimated_fixed_bytes": (
            parameter_bytes + gradient_bytes + optimizer_bytes
        ),
    }


def is_cuda_oom(error: BaseException) -> bool:
    """Return whether ``error`` represents CUDA allocation exhaustion."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    if not isinstance(error, RuntimeError):
        return False
    message = str(error).lower()
    return (
        "out of memory" in message
        or "cuda error: memory allocation" in message
    )


def profile_payload(
    selection: Mapping[str, Any],
    fixed_memory: Optional[Mapping[str, int]] = None,
) -> dict[str, Any]:
    """Combine runtime selection and optional fixed-memory estimates."""
    payload = dict(selection)
    if fixed_memory is not None:
        payload.update(fixed_memory)
    return payload
