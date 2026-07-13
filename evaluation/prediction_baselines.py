from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain

import torch

from training.hybrid import RetinaTargets, RetinaTrainingBatch


class PredictionBaselineError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GlobalChangeBaseline:
    fine_mean: torch.Tensor
    coarse_mean: torch.Tensor


@dataclass(frozen=True, slots=True)
class BaselineMSE:
    zero_fine: torch.Tensor
    zero_coarse: torch.Tensor
    global_fine: torch.Tensor
    global_coarse: torch.Tensor


@dataclass(frozen=True, slots=True)
class LocalARSupports:
    fine: torch.Tensor
    coarse: torch.Tensor


@dataclass(frozen=True, slots=True)
class LocalARTargetModel:
    source_indices: torch.Tensor
    coefficients: torch.Tensor


@dataclass(frozen=True, slots=True)
class LocalARBaseline:
    fine: tuple[LocalARTargetModel, ...]
    coarse: tuple[LocalARTargetModel, ...]
    input_steps: int
    cone_count: int


@dataclass(slots=True)
class _LocalARAccumulator:
    source_indices: torch.Tensor
    gram: torch.Tensor
    rhs: torch.Tensor


def fit_global_change_baseline(
    batches: Iterable[RetinaTrainingBatch],
) -> GlobalChangeBaseline:
    fine_total: torch.Tensor | None = None
    coarse_total: torch.Tensor | None = None
    fine_count = 0
    coarse_count = 0
    for batch in batches:
        fine = batch.targets.fine
        coarse = batch.targets.coarse
        if fine.ndim != 3 or coarse.ndim != 3 or fine.shape[1] != coarse.shape[1]:
            raise PredictionBaselineError("Targets must be [batch,horizon,target]")
        fine_sum = fine.sum(dim=(0, 2))
        coarse_sum = coarse.sum(dim=(0, 2))
        fine_total = fine_sum if fine_total is None else fine_total + fine_sum
        coarse_total = coarse_sum if coarse_total is None else coarse_total + coarse_sum
        fine_count += fine.shape[0] * fine.shape[2]
        coarse_count += coarse.shape[0] * coarse.shape[2]
    if fine_total is None or coarse_total is None:
        raise PredictionBaselineError("Cannot fit a baseline from no batches")
    return GlobalChangeBaseline(
        fine_mean=fine_total / fine_count,
        coarse_mean=coarse_total / coarse_count,
    )


def baseline_mse(
    baseline: GlobalChangeBaseline,
    targets: RetinaTargets,
) -> BaselineMSE:
    if (
        targets.fine.ndim != 3
        or targets.coarse.ndim != 3
        or baseline.fine_mean.shape != (targets.fine.shape[1],)
        or baseline.coarse_mean.shape != (targets.coarse.shape[1],)
    ):
        raise PredictionBaselineError("Baseline horizon counts do not match targets")
    global_fine = baseline.fine_mean.view(1, -1, 1)
    global_coarse = baseline.coarse_mean.view(1, -1, 1)
    return BaselineMSE(
        zero_fine=targets.fine.square().mean(),
        zero_coarse=targets.coarse.square().mean(),
        global_fine=(targets.fine - global_fine).square().mean(),
        global_coarse=(targets.coarse - global_coarse).square().mean(),
    )


def fit_local_ar_baseline(
    batches: Iterable[RetinaTrainingBatch],
    supports: LocalARSupports,
    ridge: float = 1e-3,
) -> LocalARBaseline:
    if not math.isfinite(ridge) or ridge <= 0:
        raise PredictionBaselineError("Local AR ridge must be positive and finite")
    iterator = iter(batches)
    try:
        first = next(iterator)
    except StopIteration:
        raise PredictionBaselineError("Cannot fit local AR from no batches")
    if first.x_cone.ndim != 3:
        raise PredictionBaselineError("Local AR input must be [batch,time,cone]")
    fine_accumulators = _make_accumulators(
        first.x_cone,
        first.targets.fine,
        supports.fine,
    )
    coarse_accumulators = _make_accumulators(
        first.x_cone,
        first.targets.coarse,
        supports.coarse,
    )
    expected_input_shape = first.x_cone.shape[1:]
    expected_horizons = first.targets.fine.shape[1]
    expected_target_counts = (
        first.targets.fine.shape[2],
        first.targets.coarse.shape[2],
    )
    for batch in chain((first,), iterator):
        if (
            batch.x_cone.shape[1:] != expected_input_shape
            or batch.targets.fine.shape[1] != expected_horizons
            or batch.targets.coarse.shape[1] != expected_horizons
            or batch.targets.fine.shape[2] != expected_target_counts[0]
            or batch.targets.coarse.shape[2] != expected_target_counts[1]
        ):
            raise PredictionBaselineError("Local AR batch shapes are inconsistent")
        _accumulate_scale(fine_accumulators, batch.x_cone, batch.targets.fine)
        _accumulate_scale(coarse_accumulators, batch.x_cone, batch.targets.coarse)
    return LocalARBaseline(
        fine=_finish_accumulators(fine_accumulators, ridge),
        coarse=_finish_accumulators(coarse_accumulators, ridge),
        input_steps=first.x_cone.shape[1],
        cone_count=first.x_cone.shape[2],
    )


def predict_local_ar(
    baseline: LocalARBaseline,
    x_cone: torch.Tensor,
) -> RetinaTargets:
    if x_cone.shape[1:] != (baseline.input_steps, baseline.cone_count):
        raise PredictionBaselineError("Local AR input shape does not match fit data")
    return RetinaTargets(
        fine=_predict_local_scale(x_cone, baseline.fine),
        coarse=_predict_local_scale(x_cone, baseline.coarse),
    )


def _make_accumulators(
    x_cone: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
) -> tuple[_LocalARAccumulator, ...]:
    if target.ndim != 3 or target.shape[0] != x_cone.shape[0]:
        raise PredictionBaselineError("Local AR targets must be [batch,horizon,target]")
    support = _validated_support(support, target.shape[2], x_cone.shape[2])
    indices = support.indices()
    accumulators: list[_LocalARAccumulator] = []
    for target_index in range(target.shape[2]):
        source_indices = indices[1, indices[0] == target_index]
        feature_count = 1 + x_cone.shape[1] * source_indices.numel()
        accumulators.append(
            _LocalARAccumulator(
                source_indices=source_indices,
                gram=x_cone.new_zeros(feature_count, feature_count),
                rhs=x_cone.new_zeros(feature_count, target.shape[1]),
            )
        )
    return tuple(accumulators)


def _accumulate_scale(
    accumulators: tuple[_LocalARAccumulator, ...],
    x_cone: torch.Tensor,
    target: torch.Tensor,
) -> None:
    x_cone = x_cone.detach()
    target = target.detach()
    for target_index, accumulator in enumerate(accumulators):
        features = x_cone.index_select(
            2,
            accumulator.source_indices,
        ).flatten(start_dim=1)
        design = torch.cat(
            (
                torch.ones(
                    features.shape[0],
                    1,
                    dtype=features.dtype,
                    device=features.device,
                ),
                features,
            ),
            dim=1,
        )
        accumulator.gram.add_(design.T @ design)
        accumulator.rhs.add_(design.T @ target[:, :, target_index])


def _finish_accumulators(
    accumulators: tuple[_LocalARAccumulator, ...],
    ridge: float,
) -> tuple[LocalARTargetModel, ...]:
    models = []
    for accumulator in accumulators:
        feature_count = accumulator.gram.shape[0]
        penalty = torch.diag(
            torch.cat(
                (
                    torch.zeros(
                        1,
                        dtype=accumulator.gram.dtype,
                        device=accumulator.gram.device,
                    ),
                    torch.full(
                        (feature_count - 1,),
                        ridge,
                        dtype=accumulator.gram.dtype,
                        device=accumulator.gram.device,
                    ),
                )
            )
        )
        coefficients = torch.linalg.solve(
            accumulator.gram + penalty,
            accumulator.rhs,
        )
        models.append(
            LocalARTargetModel(accumulator.source_indices, coefficients)
        )
    return tuple(models)


def _predict_local_scale(
    x_cone: torch.Tensor,
    models: tuple[LocalARTargetModel, ...],
) -> torch.Tensor:
    predictions = []
    for model in models:
        features = x_cone.index_select(2, model.source_indices).flatten(start_dim=1)
        design = torch.cat(
            (
                torch.ones(
                    features.shape[0],
                    1,
                    dtype=features.dtype,
                    device=features.device,
                ),
                features,
            ),
            dim=1,
        )
        predictions.append(design @ model.coefficients)
    return torch.stack(predictions, dim=2)


def _validated_support(
    support: torch.Tensor,
    target_count: int,
    cone_count: int,
) -> torch.Tensor:
    if support.layout != torch.sparse_coo or support.shape != (
        target_count,
        cone_count,
    ):
        raise PredictionBaselineError("Local AR support shape is invalid")
    support = support.coalesce()
    if not torch.isfinite(support.values()).all() or torch.any(
        support.values() < 0
    ):
        raise PredictionBaselineError(
            "Local AR support values must be finite and non-negative"
        )
    row_counts = torch.bincount(support.indices()[0], minlength=target_count)
    if torch.any(row_counts == 0):
        raise PredictionBaselineError("Local AR support has an empty target row")
    return support
