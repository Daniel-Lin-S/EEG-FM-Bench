"""Tests for adaptive CUDA batching and gradient accumulation.

Inputs are synthetic memory observations, batch sizes, and tiny CPU tensors.
Outputs verify reserve calculation, exact global batching, and one normalized
optimizer update without creating datasets or campaign artifacts.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from baseline.abstract.trainer import AbstractTrainer
from baseline.adaptive_batching import (
    derive_batch_candidates,
    exact_divisors,
    resolve_cuda_memory_limit,
    select_safe_micro_batch,
)
from baseline.brainomni.brainomni_config import BrainOmniConfig
from baseline.hpo.config import HpoConfig


class DummyTrainer(AbstractTrainer):
    """Minimal CPU trainer exposing the shared accumulation engine."""

    def setup_model(self) -> nn.Module:
        """Return the model assigned by the test."""
        return self.model

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Reject checkpoint loading in this artifact-free test trainer."""
        raise RuntimeError(
            f"DummyTrainer cannot load checkpoint {checkpoint_path}."
        )

    def train_step(
        self,
        batch: dict[str, Any],
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return linear predictions and mean squared error."""
        predictions = self.model(batch["data"])
        return predictions, self.loss_fn(predictions, labels)


class DummySampler:
    """Record the epoch supplied by the shared training loop."""

    def __init__(self) -> None:
        self.epoch: int | None = None

    def set_epoch(self, epoch: int) -> None:
        """Record ``epoch`` without changing batch order."""
        self.epoch = epoch


class DummyScheduler:
    """Count optimizer-update scheduler steps."""

    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        """Record one optimizer update."""
        self.steps += 1


def test_exact_global_batch_candidates() -> None:
    """Candidates preserve the requested global optimizer batch."""
    assert exact_divisors(8) == [8, 4, 2, 1]
    assert derive_batch_candidates(16, 2, True) == [8, 4, 2, 1]
    assert derive_batch_candidates(16, 2, False) == [8]
    with pytest.raises(ValueError, match="divisible"):
        derive_batch_candidates(15, 2, True)


@pytest.mark.parametrize(
    (
        "free_bytes",
        "total_bytes",
        "current_reserved_bytes",
        "expected_reserve",
        "expected_limit",
    ),
    [
        (600, 1000, 100, 150, 550),
        (1200, 2000, 200, 300, 1100),
    ],
)
def test_total_memory_reserve_accounts_for_external_usage(
    free_bytes: int,
    total_bytes: int,
    current_reserved_bytes: int,
    expected_reserve: int,
    expected_limit: int,
) -> None:
    """The ceiling preserves headroom without an 85-percent-free gate."""
    limit = resolve_cuda_memory_limit(
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        current_reserved_bytes=current_reserved_bytes,
        reserve_fraction=0.15,
    )

    assert limit.reserve_bytes == expected_reserve
    assert limit.process_limit_bytes == expected_limit
    assert limit.process_limit_fraction == pytest.approx(
        expected_limit / total_bytes
    )


def test_calibrated_selector_uses_largest_predicted_divisor() -> None:
    """Micro-batch prediction combines fixed and sample-scaled memory."""
    selected, predicted = select_safe_micro_batch(
        candidates=[8, 4, 2, 1],
        fixed_bytes=400,
        calibration_peak_bytes=500,
        calibration_batch_size=1,
        process_limit_bytes=800,
    )

    assert selected == 4
    assert predicted == 800


def test_adaptive_configuration_validation_and_defaults() -> None:
    """Training and HPO expose the selected reserve and failure defaults."""
    config = BrainOmniConfig()
    assert config.training.adaptive_batching.enabled is True
    assert (
        config.training.adaptive_batching.memory_reserve_fraction
        == pytest.approx(0.15)
    )
    assert HpoConfig().max_consecutive_failed_trials == 5
    with pytest.raises(ValidationError, match="less than 1"):
        BrainOmniConfig(
            training={
                "adaptive_batching": {
                    "memory_reserve_fraction": 1.0,
                },
            },
        )

    with pytest.raises(ValidationError, match="less than or equal to 300"):
        BrainOmniConfig(
            training={
                "adaptive_batching": {
                    "contention_wait_seconds": 301,
                },
            },
        )


def test_accumulation_normalizes_by_samples() -> None:
    """A short final window receives its own sample-normalized update."""
    config = BrainOmniConfig(
        seeds=[0],
        data={"batch_size": 4},
        training={"use_amp": False, "max_grad_norm": 100.0},
        logging={"use_cloud": False},
    )
    trainer = DummyTrainer(config)
    trainer.device = torch.device("cpu")
    trainer.world_size = 1
    trainer.model = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(trainer.model.weight)
    trainer.loss_fn = nn.MSELoss()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.scaler = torch.amp.GradScaler(enabled=False)
    trainer.scheduler = DummyScheduler()
    trainer.micro_batch_size = 2
    trainer.accumulation_steps = 2
    batches = [
        {
            "data": torch.ones(2, 1),
            "label": torch.ones(2, 1),
            "montage": ["demo/default", "demo/default"],
        },
        {
            "data": torch.ones(2, 1),
            "label": torch.ones(2, 1),
            "montage": ["demo/default", "demo/default"],
        },
        {
            "data": torch.ones(1, 1),
            "label": torch.ones(1, 1),
            "montage": ["demo/default"],
        },
    ]
    sampler = DummySampler()
    trainer._log_accumulated_update = lambda *args, **kwargs: None

    trainer._run_accumulated_epoch(
        batches,
        sampler,
        trainer.scheduler.step,
    )

    assert sampler.epoch == 0
    assert trainer.current_step == 2
    assert trainer.scheduler.steps == 2
    assert trainer.model.weight.detach().item() == pytest.approx(0.36)
