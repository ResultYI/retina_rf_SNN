from __future__ import annotations

import numpy as np
import torch

from baselines.point_process_glm import fit_point_process_glm
from data.input_identity import synthetic_input_identity
from data.rgc_response import CellMetadata, ResponseTargetKind
from training.response_data import PreparedResponseData, ResponseSplit


def test_glm_skips_test_metrics_by_default() -> None:
    result = fit_point_process_glm(
        _prepared_response_data(), device=torch.device("cpu"), steps=2
    )

    assert result.model.kernel.shape[1] == 16
    assert np.isfinite(result.validation_metrics.nll)
    assert result.test_metrics is None
    assert result.best_step >= 1


def test_glm_reports_test_metrics_when_explicitly_requested() -> None:
    result = fit_point_process_glm(
        _prepared_response_data(),
        device=torch.device("cpu"),
        steps=2,
        evaluate_test=True,
    )

    assert result.test_metrics is not None
    assert np.isfinite(result.test_metrics.nll)


def _prepared_response_data() -> PreparedResponseData:
    cells = CellMetadata(
        ids=("cell",),
        type_ids=("midget",),
        polarities=np.asarray([0], dtype=np.int64),
        positions_degs=np.zeros((1, 2), dtype=np.float32),
        eccentricities_deg=np.asarray([4.0], dtype=np.float32),
    )
    train = _response_split(0.0, "train")
    validation = _response_split(1.0, "validation")
    test = _response_split(0.0, "test")
    return PreparedResponseData(
        train=train,
        validation=validation,
        test=test,
        cells=cells,
        cone_positions_degs=np.zeros((1, 2), dtype=np.float32),
        time_axis_seconds=np.arange(20, dtype=np.float64) * 0.005,
        target_kind=ResponseTargetKind.BERNOULLI,
        normalization_mean=np.zeros(1, dtype=np.float32),
        normalization_std=np.ones(1, dtype=np.float32),
        fingerprint="test",
        input_identity=synthetic_input_identity(1, ("train", "validation", "test")),
    )


def _response_split(value: float, source: str) -> ResponseSplit:
    counts = torch.full((1, 2, 20, 1), value)
    return ResponseSplit(
        cone_response=torch.linspace(-1, 1, 20).view(1, 20, 1),
        spike_counts=counts,
        valid_mask=torch.ones_like(counts, dtype=torch.bool),
        source_ids=(source,),
        context_ids=("stationary",),
    )
