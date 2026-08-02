from __future__ import annotations

# noqa: SIZE_OK - integration checks intentionally share one scientific fixture.

import numpy as np
import pytest
import torch

from configs.physiology_profiles import macaque_photopic
from configs.rgc_type_priors import (
    ParameterPrior,
    RGCTypePrior,
    RGCTypePriors,
)
from data.rgc_response import CellMetadata
from models.cells.parameter_sharing import (
    ParameterSharingError,
    parameter_sharing_groups,
)
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
            RGCTypePrior(
                "parasol",
                **(parameters | {"threshold": _parameter(0.4, 0.02, 2.0)}),
            ),
        ),
    )


def _cells(
    *,
    type_ids: tuple[str, ...] = ("midget", "parasol", "midget", "parasol"),
) -> CellMetadata:
    return CellMetadata(
        ids=tuple(f"cell-{index}" for index in range(len(type_ids))),
        type_ids=type_ids,
        polarities=np.asarray([index % 2 for index in range(len(type_ids))]),
        positions_degs=np.asarray(
            [[0.05 * index, 0.0] for index in range(len(type_ids))],
            dtype=np.float32,
        ),
        eccentricities_deg=np.full(len(type_ids), 4.0, dtype=np.float32),
    )


def _model(
    *,
    type_ids: tuple[str, ...] = ("midget", "parasol", "midget", "parasol"),
    parameter_sharing_mode: str = "type_aware",
    matched_initialization: bool = False,
):
    cells = _cells(type_ids=type_ids)
    cone_positions = np.asarray(
        [[0.05 * index, 0.0] for index in range(len(type_ids) + 1)],
        dtype=np.float32,
    )
    profile = macaque_photopic(dt_ms=5.0, cone_spacing_deg=0.05, eccentricity_deg=4.0)
    return build_response_retina_model(
        torch.as_tensor(cone_positions),
        cells,
        profile,
        _priors(),
        support_radius_degs=0.2,
        readout_rate_tau_ms=50.0,
        surrogate_slope=5.0,
        parameter_sharing_mode=parameter_sharing_mode,
        parameter_sharing_seed=11,
        matched_initialization=matched_initialization,
    )


def test_default_type_aware_parameterization_preserves_current_shapes() -> None:
    # Given / When
    model = build_response_retina_model(
        torch.as_tensor(np.asarray([[0.0, 0.0], [0.05, 0.0]], dtype=np.float32)),
        CellMetadata(
            ids=("on", "off"),
            type_ids=("midget", "parasol"),
            polarities=np.asarray([0, 1]),
            positions_degs=np.asarray([[0.0, 0.0], [0.05, 0.0]], dtype=np.float32),
            eccentricities_deg=np.asarray([4.0, 4.0]),
        ),
        macaque_photopic(dt_ms=5.0, cone_spacing_deg=0.05, eccentricity_deg=4.0),
        _priors(),
        support_radius_degs=0.2,
        readout_rate_tau_ms=50.0,
        surrogate_slope=5.0,
    )

    # Then
    assert model.rgc.threshold.type_base_raw.shape == (2,)
    assert model.rgc.threshold.cell_residual_raw.shape == (2,)


def test_parameter_sharing_modes_record_effective_groups_and_parameter_counts() -> None:
    # Given / When
    type_aware = _model(parameter_sharing_mode="type_aware")
    type_blind = _model(parameter_sharing_mode="type_blind")
    cell_only = _model(parameter_sharing_mode="cell_only")

    # Then
    assert type_aware.rgc.parameter_sharing_mode == "type_aware"
    assert type_aware.rgc.effective_type_labels == (
        "midget",
        "parasol",
        "midget",
        "parasol",
    )
    assert type_aware.rgc.parameter_group_labels == ("midget", "parasol")
    assert type_aware.rgc.threshold.type_base_raw.numel() == 2
    assert type_aware.rgc.threshold.cell_residual_raw.numel() == 4

    assert type_blind.rgc.effective_type_labels == ("pooled",) * 4
    assert type_blind.rgc.parameter_group_labels == ("pooled",)
    assert type_blind.rgc.threshold.type_base_raw.numel() == 1
    assert type_blind.rgc.threshold.cell_residual_raw.numel() == 4

    assert cell_only.rgc.effective_type_labels == (
        "cell-0",
        "cell-1",
        "cell-2",
        "cell-3",
    )
    assert cell_only.rgc.parameter_group_labels == (
        "cell-0",
        "cell-1",
        "cell-2",
        "cell-3",
    )
    assert cell_only.rgc.threshold.type_base_raw.numel() == 4
    assert not hasattr(cell_only.rgc.threshold, "cell_residual_raw")


def test_shuffled_type_mode_is_seeded_count_preserving_and_keeps_polarity() -> None:
    # Given / When
    first = _model(parameter_sharing_mode="shuffled_type")
    second = _model(parameter_sharing_mode="shuffled_type")

    # Then
    assert first.rgc.effective_type_labels == second.rgc.effective_type_labels
    assert first.rgc.effective_type_labels != ("midget", "parasol", "midget", "parasol")
    assert sorted(first.rgc.effective_type_labels) == ["midget", "midget", "parasol", "parasol"]
    assert torch.equal(first.rgc.cell_polarities, second.rgc.cell_polarities)
    assert first.rgc.parameter_group_labels == ("midget", "parasol")


def test_shuffled_type_mode_rejects_single_observed_type() -> None:
    # Given / When / Then
    with pytest.raises(ValueError, match="at least two"):
        _model(type_ids=("midget", "midget"), parameter_sharing_mode="shuffled_type")


def test_balanced_shuffled_type_preserves_type_counts_within_each_polarity() -> None:
    # Given
    cells = _cells(type_ids=("midget", "midget", "parasol", "parasol"))

    # When
    groups = parameter_sharing_groups(
        cells,
        _priors(),
        "balanced_shuffled_type",
        11,
    )

    # Then
    assert groups.effective_type_labels != cells.type_ids
    for polarity in (0, 1):
        indices = np.flatnonzero(cells.polarities == polarity)
        assert sorted(groups.effective_type_labels[index] for index in indices) == sorted(
            cells.type_ids[index] for index in indices
        )


def test_matched_initialization_aligns_effective_parameters_and_outputs() -> None:
    # Given
    type_ids = ("midget", "midget", "parasol", "parasol")
    models = tuple(
        _model(
            type_ids=type_ids,
            parameter_sharing_mode=mode,
            matched_initialization=True,
        )
        for mode in ("type_aware", "type_blind", "balanced_shuffled_type")
    )
    sequence = torch.linspace(0.0, 1.0, 20).reshape(1, 4, 5)
    counts = torch.zeros(1, 4, 4)

    # When
    effective = tuple(
        tuple(getattr(model.rgc, name)() for name in model.rgc.parameter_names)
        for model in models
    )
    outputs = tuple(
        model.forward_sequence(sequence, observed_counts=counts)[0].spike_logits
        for model in models
    )

    # Then
    for values in zip(*effective, strict=True):
        torch.testing.assert_close(values[0], values[1], atol=1e-7, rtol=0.0)
        torch.testing.assert_close(values[0], values[2], atol=1e-7, rtol=0.0)
    torch.testing.assert_close(outputs[0], outputs[1], atol=1e-7, rtol=0.0)
    torch.testing.assert_close(outputs[0], outputs[2], atol=1e-7, rtol=0.0)


def test_parameter_sharing_groups_rejects_invalid_public_mode() -> None:
    # Given / When / Then
    with pytest.raises(ParameterSharingError, match="parameter_sharing_mode"):
        parameter_sharing_groups(_cells(), _priors(), "permutation_factory", 11)


def test_recorded_cells_have_one_output_and_causal_observed_history() -> None:
    positions = np.asarray([[0.0, 0.0], [0.05, 0.0], [0.1, 0.0]], dtype=np.float32)
    cells = CellMetadata(
        ids=("on", "off"),
        type_ids=("midget", "parasol"),
        polarities=np.asarray([0, 1]),
        positions_degs=positions[:2],
        eccentricities_deg=np.asarray([4.0, 4.0]),
    )
    profile = macaque_photopic(dt_ms=5.0, cone_spacing_deg=0.05, eccentricity_deg=4.0)
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
    profile = macaque_photopic(dt_ms=5.0, cone_spacing_deg=0.05, eccentricity_deg=4.0)
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
    profile = macaque_photopic(dt_ms=5.0, cone_spacing_deg=0.05, eccentricity_deg=4.0)
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
