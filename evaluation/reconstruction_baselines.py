from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain

import torch
from torch.nn import functional as F

from training.hybrid import RetinaTargets, RetinaTrainingBatch


class ReconstructionBaselineError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GlobalMeanBaseline:
    current_mean: torch.Tensor


@dataclass(frozen=True, slots=True)
class BaselineMSE:
    zero_current: torch.Tensor
    global_current: torch.Tensor


@dataclass(frozen=True, slots=True)
class LocalLinearSupport:
    current: torch.Tensor


@dataclass(frozen=True, slots=True)
class LocalLinearTargetModel:
    source_indices: torch.Tensor
    coefficients: torch.Tensor


@dataclass(frozen=True, slots=True)
class LocalLinearBaseline:
    current: tuple[LocalLinearTargetModel, ...]
    input_steps: int
    cone_count: int


@dataclass(frozen=True, slots=True)
class LocalLinearOutput:
    target_current: torch.Tensor


@dataclass(frozen=True, slots=True)
class _Accumulator:
    source_indices: torch.Tensor
    gram: torch.Tensor
    rhs: torch.Tensor
    weight_sum: torch.Tensor


def fit_global_mean_baseline(
    batches: Iterable[RetinaTrainingBatch],
    *,
    fit_split: str = "train",
) -> GlobalMeanBaseline:
    _validate_fit_split(fit_split)
    total: torch.Tensor | None = None
    weight: torch.Tensor | None = None
    for batch in batches:
        target = batch.targets.target_current
        _validate_target(target)
        batch_total = target.sum()
        batch_weight = target.new_tensor(target.numel())
        total = batch_total if total is None else total + batch_total
        weight = batch_weight if weight is None else weight + batch_weight
    if total is None or weight is None:
        raise ReconstructionBaselineError("Cannot fit a baseline from no batches")
    if float(weight) <= 0:
        raise ReconstructionBaselineError("Baseline targets need positive weight")
    return GlobalMeanBaseline(total / weight)


def baseline_mse(
    baseline: GlobalMeanBaseline,
    targets: RetinaTargets,
) -> BaselineMSE:
    return BaselineMSE(
        zero_current=F.mse_loss(
            torch.zeros_like(targets.target_current),
            targets.target_current,
        ),
        global_current=F.mse_loss(
            torch.full_like(targets.target_current, baseline.current_mean),
            targets.target_current,
        ),
    )


def fit_local_linear_baseline(
    batches: Iterable[RetinaTrainingBatch],
    support: LocalLinearSupport,
    *,
    ridge: float = 1e-3,
    fit_split: str = "train",
) -> LocalLinearBaseline:
    _validate_fit_split(fit_split)
    if not math.isfinite(ridge) or ridge <= 0:
        raise ReconstructionBaselineError("Local linear ridge must be positive and finite")
    iterator = iter(batches)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ReconstructionBaselineError("Cannot fit local linear from no batches") from exc
    if first.x_cone.ndim != 3:
        raise ReconstructionBaselineError("Local linear input must be [batch,time,cone]")
    current = _make_accumulators(
        first.x_cone,
        first.targets.target_current,
        support.current,
    )
    expected = (first.x_cone.shape[1:], first.targets.target_current.shape[1])
    for batch in chain((first,), iterator):
        if (
            batch.x_cone.shape[1:] != expected[0]
            or batch.targets.target_current.shape[1] != expected[1]
        ):
            raise ReconstructionBaselineError("Local linear batch shapes are inconsistent")
        _accumulate(
            current,
            batch.x_cone,
            batch.targets.target_current,
        )
    return LocalLinearBaseline(
        current=_finish(current, ridge),
        input_steps=first.x_cone.shape[1],
        cone_count=first.x_cone.shape[2],
    )


def predict_local_linear(
    baseline: LocalLinearBaseline,
    x_cone: torch.Tensor,
) -> LocalLinearOutput:
    if x_cone.shape[1:] != (baseline.input_steps, baseline.cone_count):
        raise ReconstructionBaselineError("Local linear input shape does not match fit data")
    return LocalLinearOutput(target_current=_predict_scale(x_cone, baseline.current))


def _make_accumulators(
    x_cone: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
) -> tuple[_Accumulator, ...]:
    if target.ndim != 2 or target.shape[0] != x_cone.shape[0]:
        raise ReconstructionBaselineError("Local linear targets must be [batch,target]")
    support = _validated_support(support, target.shape[1], x_cone.shape[2])
    indices = support.indices()
    accumulators = []
    for target_index in range(target.shape[1]):
        source_indices = indices[1, indices[0] == target_index]
        feature_count = 1 + x_cone.shape[1] * source_indices.numel()
        accumulators.append(
            _Accumulator(
                source_indices,
                x_cone.new_zeros(feature_count, feature_count),
                x_cone.new_zeros(feature_count),
                x_cone.new_zeros(()),
            )
        )
    return tuple(accumulators)


def _accumulate(
    accumulators: tuple[_Accumulator, ...],
    x_cone: torch.Tensor,
    target: torch.Tensor,
) -> None:
    _validate_target(target)
    for target_index, accumulator in enumerate(accumulators):
        features = x_cone.index_select(2, accumulator.source_indices).flatten(start_dim=1)
        design = torch.cat((torch.ones_like(features[:, :1]), features), dim=1)
        accumulator.gram.add_(design.T @ design)
        accumulator.rhs.add_(design.T @ target[:, target_index])
        accumulator.weight_sum.add_(design.shape[0])


def _finish(
    accumulators: tuple[_Accumulator, ...],
    ridge: float,
) -> tuple[LocalLinearTargetModel, ...]:
    models = []
    for accumulator in accumulators:
        if accumulator.weight_sum <= 0:
            models.append(
                LocalLinearTargetModel(
                    accumulator.source_indices,
                    accumulator.rhs.new_zeros(accumulator.rhs.shape),
                )
            )
            continue
        penalty = torch.eye(
            accumulator.gram.shape[0],
            dtype=accumulator.gram.dtype,
            device=accumulator.gram.device,
        ) * ridge
        penalty[0, 0] = 0
        models.append(
            LocalLinearTargetModel(
                accumulator.source_indices,
                torch.linalg.solve(accumulator.gram + penalty, accumulator.rhs),
            )
        )
    return tuple(models)


def _predict_scale(
    x_cone: torch.Tensor,
    models: tuple[LocalLinearTargetModel, ...],
) -> torch.Tensor:
    outputs = []
    for model in models:
        features = x_cone.index_select(2, model.source_indices).flatten(start_dim=1)
        design = torch.cat((torch.ones_like(features[:, :1]), features), dim=1)
        outputs.append(design @ model.coefficients)
    return torch.stack(outputs, dim=1)


def _validate_target(target: torch.Tensor) -> None:
    if target.ndim != 2:
        raise ReconstructionBaselineError("Targets must have shape [batch,target]")
    if not torch.isfinite(target).all():
        raise ReconstructionBaselineError("Targets must be finite")


def _validated_support(
    support: torch.Tensor,
    target_count: int,
    cone_count: int,
) -> torch.Tensor:
    if support.layout != torch.sparse_coo or support.shape != (target_count, cone_count):
        raise ReconstructionBaselineError("Local linear support shape is invalid")
    support = support.coalesce()
    if not torch.isfinite(support.values()).all() or torch.any(support.values() < 0):
        raise ReconstructionBaselineError("Local linear support values must be finite and non-negative")
    row_counts = torch.bincount(support.indices()[0], minlength=target_count)
    if torch.any(row_counts == 0):
        raise ReconstructionBaselineError("Local linear support has an empty target row")
    return support


def _validate_fit_split(fit_split: str) -> None:
    if fit_split != "train":
        raise ReconstructionBaselineError("Baselines may only be fit on the training split")
