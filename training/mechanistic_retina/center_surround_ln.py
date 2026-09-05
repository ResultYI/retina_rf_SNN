from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

from baselines.center_surround_ln import CenterSurroundLN, LNError
from data.retinal_recording import RealSequenceSplit
from evaluation.mechanistic_retina.factorized_ln_split import InnerDevSplit, make_inner_dev
from training.mechanistic_retina.losses import expected_bernoulli_nll
from training.mechanistic_retina.real_sampled import SpikePredictionMetrics, spike_prediction_metrics


REGULARIZATIONS: Final = (1e-5, 1e-4, 1e-3, 1e-2)
LEARNING_RATE: Final = 0.01
BATCH_SIZE: Final = 8
MAX_STEPS: Final = 1000
SEED: Final = 61001
PATIENCE: Final = 200
MIN_DELTA: Final = 1e-7


@dataclass(frozen=True, slots=True)
class LNHistory:
    dt_ms: float
    tau_ms: float


@dataclass(frozen=True, slots=True)
class LNParameterCounts:
    total: int
    requires_grad: int
    optimizer_listed: int
    nonzero_gradient: int
    actually_updated: int


@dataclass(frozen=True, slots=True)
class LNRun:
    regularization: float
    steps: int
    development: RealSequenceSplit | None = None


@dataclass(frozen=True, slots=True)
class DevelopmentStop:
    best_nll: float
    best_step: int
    plateau_reference: float
    stale_steps: int

    def observe(self, nll: float, step: int) -> DevelopmentStop:
        improved = nll < self.best_nll
        significant = nll < self.plateau_reference - MIN_DELTA
        return DevelopmentStop(
            nll if improved else self.best_nll, step if improved else self.best_step,
            nll if significant else self.plateau_reference,
            0 if significant else self.stale_steps + 1,
        )

    @property
    def stopped(self) -> bool:
        return self.stale_steps >= PATIENCE


@dataclass(frozen=True, slots=True)
class LNFit:
    model: CenterSurroundLN
    initial_state: dict[str, torch.Tensor]
    regularization: float
    train_nll_raw: float
    train_nll_trained: float
    best_step: int
    stop_step: int
    best_dev_nll: float | None
    development_curve: tuple[float, ...]
    gradients_finite: bool
    parameter_counts: LNParameterCounts


@dataclass(frozen=True, slots=True)
class LNSelection:
    inner_split: InnerDevSplit
    candidates: tuple[LNFit, ...]
    inner_dev_nll: tuple[float, ...]
    selected_lambda: float
    selected_best_step: int
    selected_stop_step: int
    refit: LNFit


def fresh_ln(train: RealSequenceSplit, history: LNHistory) -> CenterSurroundLN:
    if train.spike_events.shape != train.valid_mask.shape or train.spike_events.shape[-1] != 1:
        raise LNError("LN fitting requires single-cell aligned targets and masks")
    if not bool(train.valid_mask.any()) or not bool(((train.spike_events == 0) | (train.spike_events == 1)).all()):
        raise LNError("LN fitting requires binary targets and valid scoring bins")
    model = CenterSurroundLN(history.dt_ms, history.tau_ms, SEED).to(train.cone_drive)
    rate = train.spike_events[train.valid_mask].mean().clamp(1e-6, 1 - 1e-6)
    with torch.no_grad():
        model.bias.copy_(torch.logit(rate).reshape(1))
    return model


def evaluate_center_surround_ln(
    model: CenterSurroundLN, split: RealSequenceSplit,
) -> tuple[SpikePredictionMetrics, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        logits = torch.cat([
            model(split.cone_drive[start:start+BATCH_SIZE], split.spike_events[start:start+BATCH_SIZE])
            for start in range(0, split.cone_drive.shape[0], BATCH_SIZE)
        ])
    return spike_prediction_metrics(logits, split.spike_events, split.valid_mask), logits


def fit_center_surround_ln(train: RealSequenceSplit, history: LNHistory, run: LNRun) -> LNFit:
    if run.regularization not in REGULARIZATIONS or not 0 <= run.steps <= MAX_STEPS:
        raise LNError("training must respect the frozen lambda set and step limit")
    if run.development is not None and run.steps != MAX_STEPS:
        raise LNError("inner-development fitting must use the frozen maximum budget")
    model = fresh_ln(train, history)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    raw_metrics, _ = evaluate_center_surround_ln(model, train)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(SEED)
    parameters = tuple(model.parameters())
    gradient_seen = [torch.zeros_like(value, dtype=torch.bool) for value in parameters]
    history_feature = model.history_feature(train.spike_events).detach()
    curve: list[float] = []
    best_state = initial
    status: DevelopmentStop | None = None
    if run.development is not None:
        initial_dev, _ = evaluate_center_surround_ln(model, run.development)
        curve.append(initial_dev.population_nll)
        status = DevelopmentStop(curve[0], 0, curve[0], 0)
    stop_step = 0
    for step in range(1, run.steps + 1):
        model.train()
        indices = torch.randint(train.cone_drive.shape[0], (BATCH_SIZE,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_with_history(train.cone_drive[indices], history_feature[indices])
        loss = expected_bernoulli_nll(logits, train.spike_events[indices], train.valid_mask[indices])
        objective = loss + run.regularization * model.regularizer()
        objective.backward()
        for parameter, seen in zip(parameters, gradient_seen, strict=True):
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise LNError("LN gradient is absent or non-finite")
            seen |= parameter.grad != 0
        optimizer.step()
        stop_step = step
        if run.development is not None and status is not None:
            metrics, _ = evaluate_center_surround_ln(model, run.development)
            curve.append(metrics.population_nll)
            status = status.observe(curve[-1], step)
            if status.best_step == step:
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            if status.stopped:
                break
    if status is not None:
        model.load_state_dict(best_state, strict=True)
    trained_metrics, _ = evaluate_center_surround_ln(model, train)
    counts = LNParameterCounts(
        total=sum(p.numel() for p in parameters),
        requires_grad=sum(p.numel() for p in parameters if p.requires_grad),
        optimizer_listed=sum(p.numel() for group in optimizer.param_groups for p in group["params"]),
        nonzero_gradient=sum(int(seen.sum()) for seen in gradient_seen),
        actually_updated=sum(int((p.detach() != initial[name]).sum()) for name, p in model.named_parameters()),
    )
    return LNFit(model, initial, run.regularization, raw_metrics.population_nll,
                 trained_metrics.population_nll, status.best_step if status is not None else stop_step,
                 stop_step, status.best_nll if status is not None else None, tuple(curve), True, counts)


def select_and_refit_ln(train: RealSequenceSplit, history: LNHistory) -> LNSelection:
    inner = make_inner_dev(train)
    candidates = []
    scores = []
    for regularization in REGULARIZATIONS:
        fitted = fit_center_surround_ln(inner.train, history, LNRun(regularization, MAX_STEPS, inner.development))
        if fitted.best_dev_nll is None:
            raise LNError("inner fit did not record development NLL")
        candidates.append(fitted)
        scores.append(fitted.best_dev_nll)
        print(f"lambda={regularization:g} best-dev={fitted.best_dev_nll:.9f} best-step={fitted.best_step} stop-step={fitted.stop_step}", flush=True)
    selected = candidates[min(range(len(scores)), key=scores.__getitem__)]
    print(f"fresh full-train refit: lambda={selected.regularization:g}, steps={selected.best_step}", flush=True)
    refit = fit_center_surround_ln(train, history, LNRun(selected.regularization, selected.best_step))
    return LNSelection(inner, tuple(candidates), tuple(scores), selected.regularization,
                       selected.best_step, selected.stop_step, refit)
