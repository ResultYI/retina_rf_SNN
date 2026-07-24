from __future__ import annotations

import torch

from data.rgc_response import ResponseTargetKind
from loss.rgc_response import response_nll


def test_masked_macro_bernoulli_likelihood_is_differentiable() -> None:
    logits = torch.zeros(1, 2, 2, requires_grad=True)
    targets = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    mask = torch.tensor([[[True, True], [True, False]]])

    loss = response_nll(
        logits,
        targets,
        mask,
        ResponseTargetKind.BERNOULLI,
    )
    loss.backward()

    assert torch.isclose(loss, torch.tensor(0.6931472))
    assert logits.grad is not None
    assert logits.grad[0, 1, 1] == 0


def test_poisson_likelihood_accepts_counts() -> None:
    logits = torch.zeros(1, 2, 1)
    counts = torch.tensor([[[0.0], [2.0]]])
    mask = torch.ones_like(counts, dtype=torch.bool)

    loss = response_nll(logits, counts, mask, ResponseTargetKind.POISSON)

    assert torch.isfinite(loss)

