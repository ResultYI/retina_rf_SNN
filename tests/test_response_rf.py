from __future__ import annotations

import torch

from baselines.point_process_glm import PointProcessGLM
from evaluation.rf_dynamic import classify_dynamic_rf


def test_static_glm_uses_only_past_spike_history() -> None:
    model = PointProcessGLM(cone_count=2, cell_count=1, temporal_lags=2)
    cones = torch.zeros(1, 4, 2)
    baseline = torch.zeros(1, 4, 1)
    changed = baseline.clone()
    changed[:, 1] = 1

    before = model(cones, baseline)
    after = model(cones, changed)

    assert torch.equal(before[:, :2], after[:, :2])


def test_dynamic_rf_needs_three_independent_context_pairs() -> None:
    assert classify_dynamic_rf(1, 0.2, 0.2) == "not_identifiable"
    assert classify_dynamic_rf(3, 0.2, 0.2) == "supported"
