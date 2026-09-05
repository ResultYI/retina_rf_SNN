from __future__ import annotations

import importlib

import pytest
import torch

from models.mechanistic_retina.state import decay_from_tau, fixed_one_bin_history_state
from training.mechanistic_retina.losses import expected_bernoulli_nll


def make_model():
    module = importlib.import_module("baselines.center_surround_ln")
    return module.CenterSurroundLN(1000 / 150, 30, 61001)


def test_given_requested_model_when_resolved_then_it_exists() -> None:
    assert importlib.util.find_spec("baselines.center_surround_ln") is not None


def test_given_fresh_model_when_counted_then_gaussian_contract_holds() -> None:
    model = make_model()
    assert sum(p.numel() for p in model.parameters()) == 128
    assert all(p.requires_grad for p in model.parameters())
    torch.testing.assert_close(model.sigmas(), torch.tensor([1.5, 3.0]))
    torch.testing.assert_close(model.amplitudes(), torch.ones(2))
    torch.testing.assert_close(model.gaussians().sum(dim=(1, 2)), torch.ones(2))
    torch.testing.assert_close(model.temporal_kernels().norm(dim=1), torch.ones(2))
    assert not torch.equal(model.temporal_kernels()[0], model.temporal_kernels()[1])


def test_given_learned_geometry_when_projected_then_shared_isotropic_gaussians() -> None:
    model = make_model().double()
    with torch.no_grad():
        model.center_xy.copy_(torch.tensor([1.2, -0.6]))
        model.raw_widths.copy_(torch.tensor([-2.0, 0.3]))
    sigma = model.sigmas()
    squared = (model.grid_xy - model.center_xy).square().sum(dim=-1)
    expected = torch.stack([torch.exp(-squared / (2 * s.square())) for s in sigma])
    expected = expected / expected.sum(dim=(1, 2), keepdim=True)
    torch.testing.assert_close(model.gaussians(), expected)
    assert sigma[1] > sigma[0] > 0
    assert bool((model.amplitudes() > 0).all())


def test_given_arbitrary_input_when_filtered_then_exact_center_minus_surround_sum() -> None:
    model = make_model().double()
    cones = torch.randn(2, 75, 289, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    events = torch.zeros(2, 75, 1, dtype=torch.float64)
    expected = torch.zeros_like(events)
    kernels = model.pathway_kernels().flatten(2)
    for lag in range(60):
        expected[:, lag:, 0] += cones[:, :75-lag] @ (kernels[0, lag] - kernels[1, lag])
    torch.testing.assert_close(model(cones, events), expected)


def test_given_impulse_when_filtered_then_exact_60_bin_causal_support() -> None:
    model = make_model()
    with torch.no_grad():
        model.raw_temporal.fill_(1)
    cones = torch.zeros(1, 62, 289)
    cones[:, 0, 144] = 1
    logits = model(cones, torch.zeros(1, 62, 1))
    assert bool((logits[:, :60] > 0).all())
    assert torch.equal(logits[:, 60:], torch.zeros(1, 2, 1))


def test_given_same_cell_spikes_when_history_evaluated_then_canonical_shift_and_reset() -> None:
    model = make_model()
    with torch.no_grad():
        model.history_weight.fill_(1)
    events = torch.zeros(2, 10, 1)
    events[0, 4] = 1
    expected = fixed_one_bin_history_state(events, decay_from_tau(1000 / 150, 30))
    logits = model(torch.zeros(2, 10, 289), events)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    assert torch.equal(logits[0, :5], torch.zeros(5, 1))
    assert logits[0, 5, 0] > 0
    assert torch.equal(logits[1], torch.zeros(10, 1))


def test_given_future_stimulus_when_changed_then_past_logits_are_unchanged() -> None:
    model = make_model()
    cones = torch.zeros(1, 12, 289)
    events = torch.zeros(1, 12, 1)
    before = model(cones, events)
    cones[:, 7:] = 3
    after = model(cones, events)
    torch.testing.assert_close(before[:, :7], after[:, :7], rtol=0, atol=0)


def test_given_effective_filters_when_regularized_then_three_mean_terms_only() -> None:
    model = make_model().double()
    spatial = model.spatial_components()
    expected = spatial.square().mean() + torch.cat((
        spatial.diff(dim=1).flatten(), spatial.diff(dim=2).flatten(),
    )).square().mean() + model.temporal_kernels().diff(n=2, dim=1).square().mean()
    with torch.no_grad():
        model.bias.fill_(100)
        model.history_weight.fill_(100)
        model.raw_temporal[0].mul_(7)
        model.raw_temporal[1].mul_(3)
    torch.testing.assert_close(model.regularizer(), expected)


def test_given_spikes_when_nll_backward_then_all_parameters_have_finite_gradients() -> None:
    model = make_model()
    generator = torch.Generator().manual_seed(5)
    cones = torch.randn(2, 70, 289, generator=generator)
    events = (torch.rand(2, 70, 1, generator=generator) < 0.4).float()
    logits = model(cones, events)
    loss = expected_bernoulli_nll(logits, events, torch.ones_like(events, dtype=torch.bool))
    torch.testing.assert_close(loss, torch.nn.functional.binary_cross_entropy_with_logits(logits, events))
    loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool((p.grad != 0).all()) for p in model.parameters())


def test_given_wrong_grid_when_forward_then_rejected() -> None:
    with pytest.raises(ValueError, match="289"):
        make_model()(torch.zeros(1, 3, 288), torch.zeros(1, 3, 1))
