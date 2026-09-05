from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

from baselines.center_surround_ln import LNError
from baselines.compact_causal_cnn import CompactCausalCNN
from data.retinal_recording import RealSequenceSplit
from evaluation.mechanistic_retina.factorized_ln_split import InnerDevSplit, make_inner_dev
from training.mechanistic_retina.center_surround_ln import (
    BATCH_SIZE, MAX_STEPS, SEED, DevelopmentStop, LNHistory, LNParameterCounts,
)
from training.mechanistic_retina.losses import expected_bernoulli_nll
from training.mechanistic_retina.real_sampled import SpikePredictionMetrics, spike_prediction_metrics


LEARNING_RATES: Final = (1e-3, 3e-4)


@dataclass(frozen=True, slots=True)
class CNNRun:
    learning_rate: float
    steps: int
    development: RealSequenceSplit | None = None


@dataclass(frozen=True, slots=True)
class CNNFit:
    model: CompactCausalCNN
    initial_state: dict[str, torch.Tensor]
    learning_rate: float
    train_nll_raw: float
    train_nll_trained: float
    best_step: int
    stop_step: int
    best_dev_nll: float | None
    development_curve: tuple[float, ...]
    parameter_counts: LNParameterCounts


@dataclass(frozen=True, slots=True)
class CNNSelection:
    inner_split: InnerDevSplit
    candidates: tuple[CNNFit, ...]
    selected: CNNFit
    refit: CNNFit


def fresh_cnn(train: RealSequenceSplit, history: LNHistory) -> CompactCausalCNN:
    if train.spike_events.shape != train.valid_mask.shape or train.spike_events.shape[-1] != 1:
        raise LNError("CNN fitting requires aligned single-cell targets and masks")
    if not bool(train.valid_mask.any()) or not bool(((train.spike_events == 0) | (train.spike_events == 1)).all()):
        raise LNError("CNN fitting requires binary targets and valid scoring bins")
    model = CompactCausalCNN(history.dt_ms, history.tau_ms, SEED).to(train.cone_drive)
    rate = train.spike_events[train.valid_mask].mean().clamp(1e-6, 1 - 1e-6)
    with torch.no_grad():
        model.bias.copy_(torch.logit(rate).reshape(1))
    return model


def evaluate_cnn(model: CompactCausalCNN, split: RealSequenceSplit) -> tuple[SpikePredictionMetrics, torch.Tensor]:
    return _evaluate(model, split, model.history_feature(split.spike_events).detach())


def _evaluate(
    model: CompactCausalCNN, split: RealSequenceSplit, history: torch.Tensor,
) -> tuple[SpikePredictionMetrics, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        logits = torch.cat([
            model.forward_with_history(split.cone_drive[start:start+BATCH_SIZE], history[start:start+BATCH_SIZE])
            for start in range(0, split.cone_drive.shape[0], BATCH_SIZE)
        ])
    if not bool(torch.isfinite(logits).all()):
        raise LNError("CNN produced nonfinite logits")
    return spike_prediction_metrics(logits, split.spike_events, split.valid_mask), logits


def fit_cnn(train: RealSequenceSplit, history: LNHistory, run: CNNRun) -> CNNFit:
    if run.learning_rate not in LEARNING_RATES or not 0 <= run.steps <= MAX_STEPS:
        raise LNError("CNN must use the frozen learning rates and step limit")
    if run.development is not None and run.steps != MAX_STEPS:
        raise LNError("inner-development fitting requires the frozen maximum budget")
    model = fresh_cnn(train, history)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    history_feature = model.history_feature(train.spike_events).detach()
    raw_metrics, _ = _evaluate(model, train, history_feature)
    optimizer = torch.optim.Adam(model.parameters(), lr=run.learning_rate)
    generator = torch.Generator().manual_seed(SEED)
    parameters = tuple(model.parameters())
    seen = [torch.zeros_like(p, dtype=torch.bool) for p in parameters]
    curve: list[float] = []
    best_state = initial
    status: DevelopmentStop | None = None
    dev_history: torch.Tensor | None = None
    if run.development is not None:
        dev_history = model.history_feature(run.development.spike_events).detach()
        metrics, _ = _evaluate(model, run.development, dev_history)
        curve.append(metrics.population_nll)
        status = DevelopmentStop(curve[0], 0, curve[0], 0)
    stop_step = 0
    for step in range(1, run.steps + 1):
        model.train()
        indices = torch.randint(train.cone_drive.shape[0], (BATCH_SIZE,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_with_history(train.cone_drive[indices], history_feature[indices])
        loss = expected_bernoulli_nll(logits, train.spike_events[indices], train.valid_mask[indices])
        loss.backward()
        for parameter, flags in zip(parameters, seen, strict=True):
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise LNError("CNN gradient is absent or nonfinite")
            flags |= parameter.grad != 0
        optimizer.step()
        stop_step = step
        if run.development is not None and status is not None and dev_history is not None:
            metrics, _ = _evaluate(model, run.development, dev_history)
            curve.append(metrics.population_nll)
            status = status.observe(curve[-1], step)
            if status.best_step == step:
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            if status.stopped:
                break
        if step % 200 == 0:
            print(f"step={step} lr={run.learning_rate:g} train-batch={float(loss):.9f}", flush=True)
    if status is not None:
        model.load_state_dict(best_state, strict=True)
    trained, _ = _evaluate(model, train, history_feature)
    counts = LNParameterCounts(
        total=sum(p.numel() for p in parameters), requires_grad=sum(p.numel() for p in parameters if p.requires_grad),
        optimizer_listed=sum(p.numel() for group in optimizer.param_groups for p in group["params"]),
        nonzero_gradient=sum(int(flags.sum()) for flags in seen),
        actually_updated=sum(int((p.detach() != initial[name]).sum()) for name, p in model.named_parameters()),
    )
    return CNNFit(model, initial, run.learning_rate, raw_metrics.population_nll, trained.population_nll,
                  status.best_step if status is not None else stop_step, stop_step,
                  status.best_nll if status is not None else None, tuple(curve), counts)


def select_and_refit_cnn(train: RealSequenceSplit, history: LNHistory) -> CNNSelection:
    inner = make_inner_dev(train)
    candidates = tuple(fit_cnn(inner.train, history, CNNRun(lr, MAX_STEPS, inner.development)) for lr in LEARNING_RATES)
    scores = []
    for fit in candidates:
        if fit.best_dev_nll is None:
            raise LNError("inner fit did not record development NLL")
        scores.append(fit.best_dev_nll)
        print(f"lr={fit.learning_rate:g} best-dev={fit.best_dev_nll:.9f} best-step={fit.best_step} stop-step={fit.stop_step}", flush=True)
    selected = candidates[min(range(len(scores)), key=scores.__getitem__)]
    print(f"fresh full-train refit: lr={selected.learning_rate:g}, steps={selected.best_step}", flush=True)
    refit = fit_cnn(train, history, CNNRun(selected.learning_rate, selected.best_step))
    return CNNSelection(inner, candidates, selected, refit)
