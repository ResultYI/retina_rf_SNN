from __future__ import annotations

import pytest
import torch

from evaluation.prediction_baselines import (
    LocalARSupports,
    PredictionBaselineError,
    baseline_mse,
    fit_global_change_baseline,
    fit_local_ar_baseline,
    predict_local_ar,
)
from training.hybrid import RetinaTargets, RetinaTrainingBatch


def _batch(fine: torch.Tensor, coarse: torch.Tensor) -> RetinaTrainingBatch:
    return RetinaTrainingBatch(
        x_cone=torch.zeros(fine.shape[0], 1, 1),
        targets=RetinaTargets(fine=fine, coarse=coarse),
    )


def test_local_ar_uses_only_declared_spatial_support() -> None:
    # Given
    generator = torch.Generator().manual_seed(7)
    x_cone = torch.randn((32, 3, 2), generator=generator)
    targets = RetinaTargets(
        fine=(2.0 * x_cone[:, -1]).unsqueeze(1),
        coarse=x_cone[:, :, :1].sum(dim=1).unsqueeze(1),
    )
    batch = RetinaTrainingBatch(x_cone=x_cone, targets=targets)
    supports = LocalARSupports(
        fine=torch.eye(2).to_sparse(),
        coarse=torch.tensor([[1.0, 0.0]]).to_sparse(),
    )

    # When
    baseline = fit_local_ar_baseline((batch,), supports, ridge=1e-5)
    prediction = predict_local_ar(baseline, x_cone)

    # Then
    torch.testing.assert_close(prediction.fine, targets.fine, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(prediction.coarse, targets.coarse, atol=1e-3, rtol=1e-3)
    assert tuple(baseline.fine[0].source_indices.tolist()) == (0,)
    assert tuple(baseline.fine[1].source_indices.tolist()) == (1,)


def test_global_change_baseline_uses_train_targets_without_sample_future_access(
) -> None:
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


def test_local_ar_rejects_negative_support_weights() -> None:
    # Given
    batch = RetinaTrainingBatch(
        x_cone=torch.ones(2, 2, 1),
        targets=RetinaTargets(
            fine=torch.ones(2, 1, 1),
            coarse=torch.ones(2, 1, 1),
        ),
    )
    negative = torch.tensor([[-1.0]]).to_sparse()

    # When / Then
    with pytest.raises(PredictionBaselineError, match="non-negative"):
        fit_local_ar_baseline(
            (batch,),
            LocalARSupports(fine=negative, coarse=negative),
        )
