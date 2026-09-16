"""Exercise attention train/eval behaviour using synthetic tensor inputs.

No datasets or checkpoints are needed. Tests cover self-attention and both
sensor/neural attention projections, including finite training gradients.
"""

import pytest
import torch

from baseline.brainomni.model import import_brainomni_class

import_brainomni_class()

from model_utils.attn import SelfAttention
from model_utils.module import BackWardSolution, ForwardSolution


@pytest.mark.parametrize('kind', ['self', 'forward', 'backward'])
def test_attention_respects_eval_mode(kind: str) -> None:
    """Evaluation is deterministic while training retains attention dropout."""
    torch.manual_seed(42)
    if kind == 'self':
        module = SelfAttention(16, 4, 0.5)
        inputs = (torch.randn(2, 7, 16),)
    elif kind == 'forward':
        module = ForwardSolution(16, 4, 0.5)
        inputs = (torch.randn(2, 5, 16), torch.randn(2, 7, 16))
    else:
        module = BackWardSolution(16, 4, 0.5)
        inputs = (
            torch.randn(2, 5, 16), torch.randn(2, 7, 16),
            torch.randn(2, 7, 16),
        )
    module.eval()
    with torch.no_grad():
        first = module(*inputs)
        second = module(*inputs)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.isfinite(first).all()
    module.train()
    first = module(*inputs)
    second = module(*inputs)
    assert not torch.equal(first, second)
    first.square().mean().backward()
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_corrected_policy_has_distinct_identity() -> None:
    """Historical dropout results cannot satisfy corrected-run identity."""
    from baseline.brainomni.brainomni_config import BrainOmniConfig
    from baseline.utils.identity import get_campaign_identity

    corrected = BrainOmniConfig(fs=256).model_dump(mode='json')
    historic = BrainOmniConfig(fs=256).model_dump(mode='json')
    del historic['model']['attention_dropout_policy']
    assert get_campaign_identity(corrected, None) != get_campaign_identity(
        historic, None,
    )
