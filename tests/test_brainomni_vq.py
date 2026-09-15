"""Tests for bounded-memory BrainOmni vector-quantizer initialization.

Inputs are small synthetic latent-vector matrices. Outputs verify the
pairwise-distance and k-means results without loading datasets or models.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from baseline.brainomni import model as brainomni_model


brainomni_model.import_brainomni_class()
vq_module = importlib.import_module("model_utils.vq")
kmeans = vq_module.kmeans
squared_euclidean_distances = (
    vq_module.squared_euclidean_distances
)


def _broadcast_distances(
    samples: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    """Return the former broadcast calculation for a small test matrix."""
    return (
        (samples.unsqueeze(1) - centers).square().sum(dim=-1)
    )


def test_squared_distances_match_broadcast_reference() -> None:
    """The bounded calculation retains the original distance values."""
    samples = torch.tensor(
        [[-2.0, 1.0], [0.0, 0.0], [3.0, 4.0]],
    )
    centers = torch.tensor([[-1.0, 1.0], [2.0, 3.0]])

    actual = squared_euclidean_distances(samples, centers)
    expected = _broadcast_distances(samples, centers)

    assert actual.shape == (3, 2)
    torch.testing.assert_close(actual, expected)


def test_kmeans_matches_broadcast_reference_assignments() -> None:
    """K-means retains assignments for well-separated sample groups."""
    samples = torch.tensor(
        [
            [-5.0, -5.0],
            [-4.5, -5.5],
            [-5.5, -4.5],
            [5.0, 5.0],
            [4.5, 5.5],
            [5.5, 4.5],
        ],
    )
    torch.manual_seed(7)
    centers, counts = kmeans(samples, 2, 4)
    distances = _broadcast_distances(samples, centers)

    assert sorted(counts.tolist()) == [3, 3]
    assert torch.equal(
        torch.bincount(distances.argmin(dim=-1), minlength=2),
        counts,
    )


@pytest.mark.parametrize(
    ("samples", "centers", "message"),
    [
        (torch.ones(2, 3, 1), torch.ones(2, 3), "two-dimensional"),
        (torch.ones(2, 3), torch.ones(2, 4), "dimensions must match"),
        (
            torch.tensor([[float("nan"), 0.0]]),
            torch.ones(2, 2),
            "non-finite",
        ),
    ],
)
def test_squared_distances_reject_invalid_inputs(
    samples: torch.Tensor,
    centers: torch.Tensor,
    message: str,
) -> None:
    """Invalid shapes and values fail before distance construction."""
    with pytest.raises(ValueError, match=message):
        squared_euclidean_distances(samples, centers)


@pytest.mark.parametrize(
    ("clusters", "iterations", "message"),
    [
        (0, 1, "positive number of k-means clusters"),
        (1, 0, "positive number of k-means iterations"),
    ],
)
def test_kmeans_rejects_invalid_controls(
    clusters: int,
    iterations: int,
    message: str,
) -> None:
    """K-means control values must be positive."""
    with pytest.raises(ValueError, match=message):
        kmeans(torch.ones(2, 2), clusters, iterations)


def test_kmeans_rejects_empty_or_nonfinite_samples() -> None:
    """K-means fails clearly when initialization evidence is invalid."""
    with pytest.raises(ValueError, match="at least one sample"):
        kmeans(torch.empty(0, 2), 1, 1)
    with pytest.raises(ValueError, match="non-finite"):
        kmeans(torch.tensor([[float("inf"), 0.0]]), 1, 1)
