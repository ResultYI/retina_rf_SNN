from __future__ import annotations

import importlib
import inspect

import pytest
import torch

from data.retinal_recording import RealSequenceSplit


def small_train() -> RealSequenceSplit:
    generator = torch.Generator().manual_seed(3)
    events = (torch.rand(2, 12, 1, generator=generator) < 0.4).float()
    return RealSequenceSplit(
        torch.randn(2, 12, 289, generator=generator), events.long(), events,
        torch.ones_like(events, dtype=torch.bool), ("a", "b"), (0, 1),
    )


def test_given_contract_when_loading_then_training_entrypoint_exists() -> None:
    assert importlib.util.find_spec("training.mechanistic_retina.center_surround_ln") is not None


def test_given_plateau_when_observed_then_stops_after_exact_patience() -> None:
    module = importlib.import_module("training.mechanistic_retina.center_surround_ln")
    status = module.DevelopmentStop(0.5, 0, 0.5, 0)
    for step in range(1, 200):
        status = status.observe(0.6, step)
        assert not status.stopped
    status = status.observe(0.6, 200)
    assert status.stopped
    assert status.best_step == 0
    assert status.best_nll == 0.5


def test_given_subthreshold_improvement_when_observed_then_best_saved_without_patience_reset() -> None:
    module = importlib.import_module("training.mechanistic_retina.center_surround_ln")
    status = module.DevelopmentStop(0.5, 0, 0.5, 199).observe(0.5 - 5e-8, 200)
    assert status.stopped
    assert status.best_step == 200
    reset = status.observe(0.49, 201)
    assert reset.stale_steps == 0
    assert not reset.stopped


def test_given_fixed_refit_budget_when_fitting_then_exact_steps_and_all_parameters_update() -> None:
    module = importlib.import_module("training.mechanistic_retina.center_surround_ln")
    result = module.fit_center_surround_ln(
        small_train(), module.LNHistory(1000 / 150, 30), module.LNRun(1e-4, 3),
    )
    assert result.stop_step == result.best_step == 3
    assert result.development_curve == ()
    assert result.best_dev_nll is None
    assert result.parameter_counts.total == 128
    assert result.parameter_counts.optimizer_listed == 128
    assert result.parameter_counts.nonzero_gradient == 128
    assert result.parameter_counts.actually_updated == 128
    assert result.gradients_finite
    assert module.REGULARIZATIONS == (1e-5, 1e-4, 1e-3, 1e-2)
    assert (module.LEARNING_RATE, module.BATCH_SIZE, module.SEED) == (0.01, 8, 61001)
    assert (module.MAX_STEPS, module.PATIENCE, module.MIN_DELTA) == (1000, 200, 1e-7)
    assert list(inspect.signature(module.select_and_refit_ln).parameters) == ["train", "history"]


def test_given_zero_best_step_when_refitting_then_raw_model_is_preserved() -> None:
    module = importlib.import_module("training.mechanistic_retina.center_surround_ln")
    result = module.fit_center_surround_ln(
        small_train(), module.LNHistory(1000 / 150, 30), module.LNRun(1e-4, 0),
    )
    assert result.parameter_counts.actually_updated == 0
    assert result.stop_step == 0
    assert all(torch.equal(result.initial_state[n], p) for n, p in result.model.state_dict().items())


def test_given_changed_model_when_created_again_then_initialization_is_fresh() -> None:
    module = importlib.import_module("training.mechanistic_retina.center_surround_ln")
    history = module.LNHistory(1000 / 150, 30)
    source = small_train()
    first = module.fresh_ln(source, history)
    initial = {n: p.detach().clone() for n, p in first.named_parameters()}
    with torch.no_grad():
        first.bias.add_(10)
    second = module.fresh_ln(source, history)
    for name, parameter in second.named_parameters():
        assert parameter.data_ptr() != dict(first.named_parameters())[name].data_ptr()
        assert torch.equal(parameter, initial[name])


def test_given_undeclared_lambda_when_fitting_then_rejected_before_training() -> None:
    module = importlib.import_module("training.mechanistic_retina.center_surround_ln")
    with pytest.raises(ValueError, match="frozen"):
        module.fit_center_surround_ln(small_train(), module.LNHistory(1000 / 150, 30), module.LNRun(0.3, 3))
