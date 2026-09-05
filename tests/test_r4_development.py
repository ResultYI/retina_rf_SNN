from __future__ import annotations

import copy
from dataclasses import replace
import importlib
import inspect

import pytest
import torch

from data.retinal_recording import RealSequenceSplit
from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig
from models.mechanistic_retina.model import build_mechanistic_retina
from training.mechanistic_retina.real_sampled import RealSpikeTrainingRequest, fit_real_spike_model


def fixture():
    torch.manual_seed(17)
    model = build_mechanistic_retina(
        MechanisticRetinaConfig(architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE,
                               cell_specific_gains=True, dt_ms=1000 / 150),
        torch.tensor([[0., 0.], [0.05, 0.], [0.1, 0.], [0.15, 0.]]),
        torch.tensor([[0.04, 0.]]), ("midget",), ("ON",),
    )
    events = (torch.rand(3, 8, 1) < 0.3).float()
    split = RealSequenceSplit(torch.randn(3, 8, 4), events.long(), events,
                              torch.ones_like(events, dtype=torch.bool), ("a", "b", "c"), (0, 1, 2))
    return model, split


def test_given_r4_contract_when_loading_then_development_entrypoint_exists() -> None:
    assert importlib.util.find_spec("training.mechanistic_retina.r4_development") is not None


def test_given_same_seed_when_fixed_steps_then_exact_legacy_update_parity() -> None:
    module = importlib.import_module("training.mechanistic_retina.r4_development")
    model, split = fixture()
    legacy = copy.deepcopy(model)
    fit_real_spike_model(RealSpikeTrainingRequest(legacy, split, split, 3, 0.03, 4, 17))
    fitted = module.fit_r4(model, split, module.R4Run(seed=17, steps=3))
    assert fitted.best_step == fitted.stop_step == 3
    assert all(torch.equal(value, fitted.model.state_dict()[name]) for name, value in legacy.state_dict().items())
    assert all(step == 3 for step in fitted.optimizer_steps)
    assert (module.LEARNING_RATE, module.BATCH_SIZE, module.MAX_STEPS) == (0.03, 4, 1000)


def test_given_development_curve_when_fitting_then_restore_exact_best(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("training.mechanistic_retina.r4_development")
    model, train = fixture()
    development = copy.deepcopy(train)
    evaluate = module.evaluate_retinal_model
    values = iter((0.5, 0.4, 0.3, 0.31, 0.32))
    snapshots = []

    def score(current, split):
        metrics, logits = evaluate(current, split)
        if split is development:
            snapshots.append({k: v.detach().clone() for k, v in current.state_dict().items()})
            metrics = replace(metrics, population_nll=next(values))
        return metrics, logits

    monkeypatch.setattr(module, "MAX_STEPS", 4)
    monkeypatch.setattr(module, "evaluate_retinal_model", score)
    result = module.fit_r4(model, train, module.R4Run(seed=17, steps=4, development=development))
    assert (result.best_step, result.stop_step, result.best_dev_nll) == (2, 4, 0.3)
    assert tuple(row.inner_dev_nll for row in result.trajectory) == (0.5, 0.4, 0.3, 0.31, 0.32)
    assert all(torch.equal(value, result.model.state_dict()[name]) for name, value in snapshots[2].items())


def test_given_zero_best_step_when_refitting_then_preserves_raw_state() -> None:
    module = importlib.import_module("training.mechanistic_retina.r4_development")
    model, train = fixture()
    result = module.fit_r4(model, train, module.R4Run(seed=17, steps=0))
    assert result.stop_step == result.best_step == 0
    assert result.actually_updated == ()
    assert all(torch.equal(value, result.model.state_dict()[name]) for name, value in result.initial_state.items())


def test_given_selection_when_inspected_then_same_ln_split_and_no_validation_argument() -> None:
    module = importlib.import_module("training.mechanistic_retina.r4_development")
    from evaluation.mechanistic_retina.factorized_ln_split import make_inner_dev
    from training.mechanistic_retina.center_surround_ln import DevelopmentStop

    assert module.make_inner_dev is make_inner_dev
    assert module.DevelopmentStop is DevelopmentStop
    assert list(inspect.signature(module.select_and_refit_r4).parameters) == ["train", "factory", "seed"]


def test_given_changed_inner_model_when_refitting_then_fresh_factory_called_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("training.mechanistic_retina.r4_development")
    from evaluation.mechanistic_retina.factorized_ln_split import InnerDevSplit

    model, train = fixture()
    created = []

    def factory():
        value = copy.deepcopy(model)
        created.append(value)
        return value

    monkeypatch.setattr(module, "MAX_STEPS", 2)
    monkeypatch.setattr(module, "make_inner_dev", lambda source: InnerDevSplit(source, copy.deepcopy(source), ()))
    selected = module.select_and_refit_r4(train, factory, 17)
    assert len(created) == 2 and created[0] is not created[1]
    assert selected.refit.stop_step == selected.inner.best_step
    assert all(torch.equal(value, selected.refit.initial_state[name]) for name, value in model.state_dict().items())


def test_given_raw_development_best_when_selecting_then_fresh_refit_has_zero_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("training.mechanistic_retina.r4_development")
    from evaluation.mechanistic_retina.factorized_ln_split import InnerDevSplit

    model, train = fixture()
    development = copy.deepcopy(train)
    evaluate = module.evaluate_retinal_model
    values = iter((0.3, 0.4, 0.5))

    def score(current, split):
        metrics, logits = evaluate(current, split)
        if split is development:
            metrics = replace(metrics, population_nll=next(values))
        return metrics, logits

    monkeypatch.setattr(module, "MAX_STEPS", 2)
    monkeypatch.setattr(module, "make_inner_dev", lambda source: InnerDevSplit(source, development, ()))
    monkeypatch.setattr(module, "evaluate_retinal_model", score)
    result = module.select_and_refit_r4(train, lambda: copy.deepcopy(model), 17)
    assert result.inner.best_step == result.refit.stop_step == 0
    assert result.refit.actually_updated == ()
    assert result.refit.optimizer_steps and set(result.refit.optimizer_steps) == {0}
    assert result.refit.trajectory == ()
    assert all(torch.equal(value, result.refit.model.state_dict()[name]) for name, value in model.state_dict().items())
