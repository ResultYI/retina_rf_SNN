from __future__ import annotations

import numpy as np
import torch

from configs.physiology_profiles import human_macaque
from configs.rgc_type_priors import (
    ParameterPrior,
    RGCTypePrior,
    RGCTypePriors,
)
from data.rgc_response import CellMetadata
from models.response_snn import build_response_retina_model
from training.response_unroll import ResponseUnrollRequest, unroll_response


def _parameter(mean: float, lower: float, upper: float) -> ParameterPrior:
    return ParameterPrior(mean, lower, upper)


def _priors() -> RGCTypePriors:
    parameters = {
        "spatial_sigma": _parameter(0.08, 0.02, 0.2),
        "sustained_mix": _parameter(0.5, 0.0, 1.0),
        "membrane_tau_ms": _parameter(20.0, 5.0, 250.0),
        "adaptation_tau_ms": _parameter(80.0, 5.0, 250.0),
        "adaptation_gain": _parameter(0.1, 0.0, 1.0),
        "amacrine_gain": _parameter(0.05, 0.0, 1.0),
        "threshold": _parameter(0.2, 0.02, 2.0),
        "subunit_tau_ms": _parameter(50.0, 5.0, 250.0),
        "subunit_gain": _parameter(0.5, 0.0, 2.0),
    }
    return RGCTypePriors(
        0.25,
        0.01,
        0.02,
        (
            RGCTypePrior("midget", **parameters),
            RGCTypePrior("parasol", **parameters),
        ),
    )


def test_recorded_cells_have_one_output_and_causal_observed_history() -> None:
    positions = np.asarray([[0.0, 0.0], [0.05, 0.0], [0.1, 0.0]], dtype=np.float32)
    cells = CellMetadata(
        ids=("on", "off"),
        type_ids=("midget", "parasol"),
        polarities=np.asarray([0, 1]),
        positions_degs=positions[:2],
        eccentricities_deg=np.asarray([4.0, 4.0]),
    )
    profile = human_macaque(dt_ms=5.0, cone_spacing_deg=0.05, eccentricity_deg=4.0)
    model = build_response_retina_model(
        torch.as_tensor(positions),
        cells,
        profile,
        _priors(),
        support_radius_degs=0.2,
        readout_rate_tau_ms=50.0,
        surrogate_slope=5.0,
    )
    sequence = torch.rand(1, 4, 3)
    no_spike = torch.zeros(1, 4, 2)
    one_spike = no_spike.clone()
    one_spike[:, 1, 0] = 1

    baseline, _ = model.forward_sequence(sequence, observed_counts=no_spike)
    changed, _ = model.forward_sequence(sequence, observed_counts=one_spike)

    assert baseline.spike_logits.shape == (1, 4, 2)
    assert torch.equal(baseline.spike_logits[:, :2], changed.spike_logits[:, :2])
    assert not torch.equal(baseline.spike_logits[:, 2:], changed.spike_logits[:, 2:])
    assert torch.allclose(model.rgc.compute_spatial_weights().sum(dim=1), torch.ones(2))


def test_type_prior_penalty_tracks_type_base_drift() -> None:
    positions = np.asarray([[0.0, 0.0], [0.05, 0.0]], dtype=np.float32)
    cells = CellMetadata(
        ids=("on", "off"),
        type_ids=("midget", "parasol"),
        polarities=np.asarray([0, 1]),
        positions_degs=positions,
        eccentricities_deg=np.asarray([4.0, 4.0]),
    )
    profile = human_macaque(dt_ms=5.0, cone_spacing_deg=0.05, eccentricity_deg=4.0)
    model = build_response_retina_model(
        torch.as_tensor(positions),
        cells,
        profile,
        _priors(),
        support_radius_degs=0.2,
        readout_rate_tau_ms=50.0,
        surrogate_slope=5.0,
    )

    baseline = model.rgc.physiology_prior_penalty()
    with torch.no_grad():
        model.rgc.threshold.type_base_raw.add_(0.5)

    assert baseline == 0
    assert model.rgc.physiology_prior_penalty() > 0


def test_checkpointed_response_unroll_matches_plain_unroll() -> None:
    positions = np.asarray([[0.0, 0.0], [0.05, 0.0], [0.1, 0.0]], dtype=np.float32)
    cells = CellMetadata(
        ids=("on", "off"),
        type_ids=("midget", "parasol"),
        polarities=np.asarray([0, 1]),
        positions_degs=positions[:2],
        eccentricities_deg=np.asarray([4.0, 4.0]),
    )
    profile = human_macaque(dt_ms=5.0, cone_spacing_deg=0.05, eccentricity_deg=4.0)
    model = build_response_retina_model(
        torch.as_tensor(positions),
        cells,
        profile,
        _priors(),
        support_radius_degs=0.2,
        readout_rate_tau_ms=50.0,
        surrogate_slope=5.0,
    )
    sequence = torch.rand(1, 8, 3)
    counts = torch.zeros(1, 8, 2)
    base = dict(
        model=model,
        cone_response=sequence,
        observed_counts=counts,
        burn_in_steps=2,
        differentiable_steps=6,
        checkpoint_block_steps=2,
    )

    plain, _ = unroll_response(ResponseUnrollRequest(**base, checkpointed=False))
    checkpointed, _ = unroll_response(
        ResponseUnrollRequest(**base, checkpointed=True)
    )

    assert torch.allclose(plain.spike_logits, checkpointed.spike_logits)
