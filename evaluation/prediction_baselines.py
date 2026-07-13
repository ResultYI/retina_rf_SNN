from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

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
