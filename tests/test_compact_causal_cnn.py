from __future__ import annotations

import importlib

import torch

from baselines.center_surround_ln import CenterSurroundLN


def test_given_frozen_spec_when_resolved_then_cnn_exists() -> None:
    assert importlib.util.find_spec("baselines.compact_causal_cnn") is not None


def test_given_future_stimulus_when_changed_then_past_logits_are_identical() -> None:
    module = importlib.import_module("baselines.compact_causal_cnn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = module.CompactCausalCNN(1000 / 150, 30, 61001).to(device)
    generator = torch.Generator().manual_seed(42)
    cones = torch.randn(2, 90, 289, generator=generator).to(device)
    events = (torch.rand(2, 90, 1, generator=generator) < 0.3).float().to(device)
    original = model(cones, events)
    changed = cones.clone()
    changed[:, 61:] = (torch.randn(2, 29, 289, generator=generator) * 9).to(device)
    actual = model(changed, events)
    torch.testing.assert_close(actual[:, :61], original[:, :61], rtol=0, atol=0)
    assert not torch.equal(actual[:, 61:], original[:, 61:])


def test_given_frozen_kernels_when_counted_then_2990_parameters_and_60_bins() -> None:
    module = importlib.import_module("baselines.compact_causal_cnn")
    model = module.CompactCausalCNN(1000 / 150, 30, 61001)
    assert model.conv1.weight.shape == (4, 1, 12, 5, 5)
    assert model.conv2.weight.shape == (4, 4, 9, 3, 3)
    assert model.conv2.dilation == (6, 1, 1)
    assert model.spatial_readout.shape == (4, 11, 11)
    assert sum(p.numel() for p in model.parameters()) == 2990
    assert all(p.requires_grad for p in model.parameters())
    with torch.no_grad():
        model.conv1.weight.fill_(0.01)
        model.conv2.weight.fill_(0.01)
        model.conv1.bias.zero_()
        model.conv2.bias.zero_()
        model.spatial_readout.fill_(0.01)
    cones = torch.ones(1, 75, 289, requires_grad=True)
    logit = model(cones, torch.zeros(1, 75, 1))[0, 70, 0]
    gradient = torch.autograd.grad(logit, cones)[0].abs().sum(dim=-1)
    assert torch.equal(gradient[:, :11], torch.zeros(1, 11))
    assert bool((gradient[:, 11:71] > 0).all())
    assert torch.equal(gradient[:, 71:], torch.zeros(1, 4))


def test_given_observed_spikes_when_head_evaluated_then_same_ln_history_and_bias() -> None:
    module = importlib.import_module("baselines.compact_causal_cnn")
    cnn = module.CompactCausalCNN(1000 / 150, 30, 61001)
    ln = CenterSurroundLN(1000 / 150, 30, 61001)
    with torch.no_grad():
        cnn.spatial_readout.zero_()
        for model in (cnn, ln):
            model.history_weight.fill_(1.7)
            model.bias.fill_(-0.2)
    events = torch.zeros(2, 75, 1)
    events[0, 60] = 1
    cones = torch.zeros(2, 75, 289)
    actual = cnn(cones, events)
    torch.testing.assert_close(cnn.history_feature(events), ln.history_feature(events), rtol=0, atol=0)
    torch.testing.assert_close(actual, ln(cones, events), rtol=0, atol=0)
    torch.testing.assert_close(actual[0, :61], actual[1, :61], rtol=0, atol=0)
    assert actual[0, 61, 0] > actual[1, 61, 0]


def test_given_same_seed_when_reinitialized_then_fresh_identical_parameters() -> None:
    module = importlib.import_module("baselines.compact_causal_cnn")
    first = module.CompactCausalCNN(1000 / 150, 30, 61001)
    second = module.CompactCausalCNN(1000 / 150, 30, 61001)
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert left.data_ptr() != right.data_ptr()
        assert torch.equal(left, right)


def test_given_two_step_refit_when_fitted_then_finite_gradients_and_updates() -> None:
    from data.retinal_recording import RealSequenceSplit
    from training.mechanistic_retina.compact_causal_cnn import CNNRun, LNHistory, fit_cnn

    generator = torch.Generator().manual_seed(3)
    events = (torch.rand(2, 12, 1, generator=generator) < 0.4).float()
    train = RealSequenceSplit(torch.randn(2, 12, 289, generator=generator), events.long(), events,
                              torch.ones_like(events, dtype=torch.bool), ("a", "b"), (0, 1))
    fit = fit_cnn(train, LNHistory(1000 / 150, 30), CNNRun(1e-3, 2))
    assert fit.stop_step == fit.best_step == 2
    assert fit.development_curve == ()
    assert fit.parameter_counts.total == fit.parameter_counts.optimizer_listed == 2990
    assert fit.parameter_counts.nonzero_gradient > 0
    assert fit.parameter_counts.actually_updated > 0


def test_given_selection_entrypoint_when_inspected_then_validation_is_not_an_input() -> None:
    import inspect
    from training.mechanistic_retina.compact_causal_cnn import LEARNING_RATES, select_and_refit_cnn

    assert LEARNING_RATES == (1e-3, 3e-4)
    assert tuple(inspect.signature(select_and_refit_cnn).parameters) == ("train", "history")


def test_given_gpu_runner_when_imported_then_no_movie_decoder_is_required() -> None:
    import runpy
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / ".omo/evidence/compact_causal_cnn_baseline/train.py"
    namespace = runpy.run_path(str(path))
    assert callable(namespace["main"])
