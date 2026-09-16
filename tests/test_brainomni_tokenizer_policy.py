"""Check downstream tokenizer policy and RVQ state with synthetic tensors.

No datasets or experiment artifacts are used. Temporary directories exercise
configuration rejection before campaign creation; tiny models exercise the
actual tokenisation and downstream gradient paths.
"""

import importlib
from copy import deepcopy
from pathlib import Path
import sys
from unittest import mock

import pytest
import torch
from torch import nn

from baseline.brainomni import model as model_utils
from baseline.brainomni.brainomni_config import BrainOmniConfig
from baseline.brainomni.brainomni_trainer import BrainOmniTrainer


model_utils.import_brainomni_class()
EuclideanCodebook = importlib.import_module(
    "model_utils.vq"
).EuclideanCodebook
SMALL_MODEL = {
    "window_length": 32, "n_filters": 4, "ratios": [2, 2],
    "kernel_size": 5, "last_kernel_size": 5, "n_dim": 8,
    "n_neuro": 4, "n_head": 2, "dropout": 0.0,
    "codebook_dim": 8, "codebook_size": 16, "num_quantizers": 2,
    "rotation_trick": True, "quantize_optimize_method": "ema",
    "overlap_ratio": 0.0, "lm_dim": 8, "lm_head": 2,
    "lm_depth": 2, "lm_dropout": 0.0, "mask_ratio": 0.5,
    "num_quantizers_used": 2,
}


def test_downstream_rejects_unfrozen_tokenizer(
    caplog: pytest.LogCaptureFixture, tmp_path: Path,
) -> None:
    """Reject configuration and direct API bypasses before accessing files."""
    config = BrainOmniConfig(fs=256)
    assert config.model.freeze_tokenizer is True
    assert config.validate_config()
    config.model.freeze_tokenizer = False
    assert not config.validate_config()
    assert "freeze_tokenizer=False is invalid" in caplog.text
    with pytest.raises(ValueError, match="freeze_tokenizer=True"):
        model_utils.load_brainomni_from_pretrained(
            str(tmp_path / "missing"), freeze_tokenizer=False,
        )
    with pytest.raises(ValueError, match="freeze_tokenizer=True"):
        BrainOmniTrainer(config)
    with pytest.raises(ValueError, match="freeze_tokenizer=True"):
        model_utils.build_brainomni_from_cfg({"freeze_tokenizer": False})
    assert list(tmp_path.iterdir()) == []


def test_downstream_gradients_and_buffers() -> None:
    """Classification trains downstream layers without changing tokenizer."""
    torch.manual_seed(42)
    model = model_utils.build_brainomni_from_cfg(SMALL_MODEL)
    classifier = nn.Linear(SMALL_MODEL["lm_dim"], 2)
    inputs = {
        "x": torch.randn(2, 4, 64),
        "pos": torch.randn(2, 4, 6),
        "sensor_type": torch.zeros(2, 4, dtype=torch.long),
    }
    model.train()
    features = model.encode(**inputs)
    before = deepcopy(model.tokenizer.state_dict())
    loss = nn.functional.cross_entropy(
        classifier(features.mean(dim=(1, 2))), torch.tensor([0, 1]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(not p.requires_grad for p in model.tokenizer.parameters())
    assert all(p.grad is None for p in model.tokenizer.parameters())
    assert any(p.grad is not None for p in model.blocks.parameters())
    assert classifier.weight.grad is not None
    model.train()
    model.encode(**inputs)
    assert not model.tokenizer.training
    for key, value in model.tokenizer.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ema_fixed_point(dtype: torch.dtype) -> None:
    """Exact k-means centroids survive subsequent training EMA updates."""
    centers = torch.eye(2)
    counts = torch.tensor([30.0, 10.0])
    data = torch.repeat_interleave(centers, counts.long(), dim=0).to(dtype)
    codebook = EuclideanCodebook(
        dim=2, codebook_size=2, kmeans_init=True,
        threshold_ema_dead_code=0,
    ).to(dtype)
    with mock.patch("model_utils.vq.kmeans", return_value=(centers, counts)):
        codebook.init_embed_(data)
    torch.testing.assert_close(
        codebook.embed_avg.float(), centers * counts[:, None],
    )
    for _ in range(3):
        with torch.autocast(
            "cpu", dtype=dtype, enabled=dtype != torch.float32,
        ):
            output, indices = codebook(data)
        assert torch.isfinite(output).all()
        assert indices.unique().numel() == 2
        torch.testing.assert_close(
            codebook.embed.float(), centers, atol=0.01, rtol=0,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_revival_survives_update(dtype: torch.dtype) -> None:
    """Replacement resets all state and is selectable on the next batch."""
    codebook = EuclideanCodebook(dim=1, codebook_size=2).to(dtype)
    codebook.embed.copy_(torch.tensor([[1.0], [99.0]]))
    codebook.embed_avg.copy_(torch.tensor([[10.0], [99.0]]))
    codebook.cluster_size.copy_(torch.tensor([10.0, 0.0]))
    data = torch.tensor([[1.0], [2.0]], dtype=dtype)
    with mock.patch(
        "model_utils.vq.sample_vectors",
        return_value=torch.full((2, 1), 2.0, dtype=dtype),
    ), mock.patch("model_utils.vq.broadcast_tensors") as sync:
        codebook(data)
    assert sync.call_count == 1
    assert codebook.embed[1].item() == 2
    assert codebook.cluster_size[1].item() == 2
    assert codebook.embed_avg[1].item() == 4
    codebook.eval()
    _, indices = codebook(data[1:])
    assert indices.item() == 1


def test_loaded_codebook_inference_unchanged() -> None:
    """Historical EMA sums do not affect evaluation centroids or state."""
    codebook = EuclideanCodebook(dim=2, codebook_size=4).eval()
    state = deepcopy(codebook.state_dict())
    state["embed_avg"].fill_(99)
    state["cluster_size"].zero_()
    codebook.load_state_dict(state, strict=True)
    data = torch.randn(7, 2)
    expected_indices = torch.cdist(data, state["embed"]).argmin(dim=-1)
    output, indices = codebook(data)
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(output, state["embed"][expected_indices])
    for key, value in codebook.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


def test_initial_pseudocounts_and_invalid_values() -> None:
    """Consistent random state and clear rejection of undefined inputs."""
    codebook = EuclideanCodebook(dim=2, codebook_size=4)
    torch.testing.assert_close(
        codebook.embed_avg,
        codebook.embed * codebook.cluster_size[:, None],
    )
    for data in (
        torch.empty(0, 2), torch.ones(3, 4),
        torch.full((2, 2), float("nan")),
    ):
        with pytest.raises(ValueError):
            codebook(data)
    codebook.eval()
    codebook.embed.fill_(float("nan"))
    with pytest.raises(ValueError, match="output must be finite"):
        codebook(torch.ones(2, 2))


def test_invalid_cli_creates_no_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The public entry point rejects tokenizer training before artifacts."""
    import baseline_main
    import yaml

    run_dir = tmp_path / "runs"
    config = tmp_path / "invalid.yaml"
    config.write_text(yaml.safe_dump({
        "model_type": "brainomni", "fs": 256,
        "model": {"freeze_tokenizer": False},
        "logging": {"run_dir": str(run_dir)},
    }))
    monkeypatch.setattr(
        sys, "argv", ["baseline_main.py", f"conf_file={config}"],
    )
    with pytest.raises(ValueError, match="Invalid configuration"):
        baseline_main.main()
    assert "freeze_tokenizer=False is invalid" in caplog.text
    assert not run_dir.exists()
