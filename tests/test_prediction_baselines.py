from __future__ import annotations

import torch

from evaluation.prediction_baselines import baseline_mse, fit_global_change_baseline
from training.hybrid import RetinaTargets, RetinaTrainingBatch


def _batch(fine: torch.Tensor, coarse: torch.Tensor) -> RetinaTrainingBatch:
    return RetinaTrainingBatch(
        x_cone=torch.zeros(fine.shape[0], 1, 1),
        targets=RetinaTargets(fine=fine, coarse=coarse),
    )


def test_global_change_baseline_uses_train_targets_without_sample_future_access() -> None:
    train = _batch(
        torch.tensor([[[1.0, 3.0]], [[3.0, 5.0]]]),
        torch.tensor([[[2.0]], [[6.0]]]),
    )
    baseline = fit_global_change_baseline((train,))

    metrics = baseline_mse(baseline, train.targets)

    torch.testing.assert_close(baseline.fine_mean, torch.tensor([3.0]))
    torch.testing.assert_close(baseline.coarse_mean, torch.tensor([4.0]))
    assert metrics.global_fine < metrics.zero_fine
    assert metrics.global_coarse < metrics.zero_coarse
