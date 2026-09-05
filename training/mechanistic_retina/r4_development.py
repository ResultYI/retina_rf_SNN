from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import torch

from data.retinal_recording import RealSequenceSplit
from evaluation.mechanistic_retina.factorized_ln_split import InnerDevSplit, make_inner_dev
from evaluation.mechanistic_retina.karamanlis_prediction_baselines import evaluate_retinal_model
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina
from training.mechanistic_retina.center_surround_ln import DevelopmentStop
from training.mechanistic_retina.losses import expected_bernoulli_nll
from training.mechanistic_retina.optimizer import build_phase1_optimizer, phase1_parameters
from training.mechanistic_retina.real_sampled import RealSpikeTrainingError


LEARNING_RATE: Final = 0.03
BATCH_SIZE: Final = 4
MAX_STEPS: Final = 1000


@dataclass(frozen=True, slots=True)
class R4Run:
    seed: int
    steps: int
    development: RealSequenceSplit | None = None


@dataclass(frozen=True, slots=True)
class R4Step:
    step: int
    sampled_train_batch_nll: float | None
    inner_dev_nll: float | None


@dataclass(frozen=True, slots=True)
class R4Fit:
    model: MechanisticGraphTemporalRetina
    initial_state: dict[str, torch.Tensor]
    best_step: int
    stop_step: int
    best_dev_nll: float | None
    train_nll_raw: float
    train_nll_trained: float
    trajectory: tuple[R4Step, ...]
    actually_updated: tuple[str, ...]
    optimizer_steps: tuple[int, ...]
    gradients_finite: bool


@dataclass(frozen=True, slots=True)
class R4Selection:
    split: InnerDevSplit
    inner: R4Fit
    refit: R4Fit


def fit_r4(model: MechanisticGraphTemporalRetina, train: RealSequenceSplit, run: R4Run) -> R4Fit:
    """Preserve R4 updates; optionally select a checkpoint using inner development NLL."""
    if not 0 <= run.steps <= MAX_STEPS:
        raise RealSpikeTrainingError("R4 run exceeds the frozen step contract")
    if run.development is not None and run.steps != MAX_STEPS:
        raise RealSpikeTrainingError("development fitting must use the full maximum budget")
    if not bool(train.valid_mask.any()) or not bool(((train.spike_events == 0) | (train.spike_events == 1)).all()):
        raise RealSpikeTrainingError("R4 fitting requires valid binary spike events")
    torch.manual_seed(run.seed)
    optimizer = build_phase1_optimizer(model, learning_rate=LEARNING_RATE)
    parameters = phase1_parameters(model)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    raw_metrics, _ = evaluate_retinal_model(model, train)
    generator = torch.Generator().manual_seed(run.seed + 1_000_003)
    status: DevelopmentStop | None = None
    best_state = initial
    trajectory = []
    if run.development is not None:
        metrics, _ = evaluate_retinal_model(model, run.development)
        status = DevelopmentStop(metrics.population_nll, 0, metrics.population_nll, 0)
        trajectory.append(R4Step(0, None, metrics.population_nll))
    stop_step = 0
    for step in range(1, run.steps + 1):
        model.train()
        indices = torch.randint(train.cone_drive.shape[0], (BATCH_SIZE,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_sequence(train.cone_drive[indices], observed_counts=train.spike_events[indices]).logits
        loss = expected_bernoulli_nll(logits, train.spike_events[indices], train.valid_mask[indices])
        loss.backward()
        if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) for parameter in parameters):
            raise RealSpikeTrainingError("R4 gradient is absent or non-finite")
        optimizer.step()
        model.project_mechanism_parameters()
        stop_step = step
        dev_nll = None
        if run.development is not None and status is not None:
            metrics, _ = evaluate_retinal_model(model, run.development)
            dev_nll = metrics.population_nll
            status = status.observe(dev_nll, step)
            if status.best_step == step:
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        trajectory.append(R4Step(step, float(loss.detach()), dev_nll))
        if step % 50 == 0:
            print(f"R4 step={step} batch-NLL={float(loss.detach()):.9f} inner-dev={dev_nll}", flush=True)
        if status is not None and status.stopped:
            break
    if status is not None:
        model.load_state_dict(best_state, strict=True)
    trained_metrics, _ = evaluate_retinal_model(model, train)
    updated = tuple(name for name, value in model.named_parameters() if not torch.equal(initial[name], value.detach()))
    optimizer_steps = tuple(int(optimizer.state[p].get("step", 0)) for p in parameters)
    return R4Fit(model, initial, status.best_step if status is not None else stop_step,
                 stop_step, status.best_nll if status is not None else None,
                 raw_metrics.population_nll, trained_metrics.population_nll,
                 tuple(trajectory), updated, optimizer_steps, True)


def select_and_refit_r4(
    train: RealSequenceSplit, factory: Callable[[], MechanisticGraphTemporalRetina], seed: int,
) -> R4Selection:
    """Select on the LN inner split, then discard inner weights and optimizer for refit."""
    split = make_inner_dev(train)
    print("R4 inner-development fit", flush=True)
    inner = fit_r4(factory(), split.train, R4Run(seed, MAX_STEPS, split.development))
    print(f"R4 selected best={inner.best_step} stop={inner.stop_step}; fresh full-train refit", flush=True)
    refit = fit_r4(factory(), train, R4Run(seed, inner.best_step))
    return R4Selection(split, inner, refit)
