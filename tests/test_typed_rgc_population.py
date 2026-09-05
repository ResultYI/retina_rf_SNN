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
from models.response_snn import build_response_retina_model, response_state_to_tensors
from evaluation.direct_readout_paths import forward_sequence_readout_paths
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
    enable_response_bias: bool = False,
    enable_synaptic_gain: bool = False,
    enable_direct_readout: bool = False,
    readout_mode: str = "v2_direct_logit",
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
        enable_response_bias=enable_response_bias,
        enable_synaptic_gain=enable_synaptic_gain,
        enable_direct_readout=enable_direct_readout,
        synaptic_gain_min=0.1,
        synaptic_gain_max=4.0,
        synaptic_gain_init=1.0,
        readout_mode=readout_mode,
    )


def test_v3_step_zero_preserves_v2_forward_and_state() -> None:
    # Given
    torch.manual_seed(17)
    v2 = _model(
        type_ids=("midget", "parasol"),
        enable_response_bias=True,
        enable_synaptic_gain=True,
        enable_direct_readout=True,
    )
    torch.manual_seed(17)
    v3 = _model(
        type_ids=("midget", "parasol"),
        enable_response_bias=True,
        readout_mode="v3_mechanism_preserving",
    )
    sequence = torch.randn(3, 24, 3)
    observed = torch.zeros(3, 24, 2)
    observed[:, (7, 15), :] = 1.0

    # When
    v2_output, v2_state = v2.forward_sequence(sequence, observed_counts=observed)
    v3_output, v3_state = v3.forward_sequence(sequence, observed_counts=observed)

    # Then
    torch.testing.assert_close(v3_output.spike_logits, v2_output.spike_logits, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(v3_output.generator_potential, v2_output.generator_potential, atol=2e-6, rtol=2e-6)
    for v3_value, v2_value in zip(
        response_state_to_tensors(v3_state),
        response_state_to_tensors(v2_state),
        strict=True,
    ):
        torch.testing.assert_close(v3_value, v2_value, atol=2e-6, rtol=2e-6)


def test_v3_has_nonnegative_current_gains_and_no_direct_logit_parameters() -> None:
    # Given / When
    model = _model(
        type_ids=("midget", "parasol"),
        enable_response_bias=True,
        readout_mode="v3_mechanism_preserving",
    )

    # Then
    assert model.rgc.bipolar_current_gain_raw.shape == (2, 2)
    assert model.rgc.amacrine_current_gain_raw.shape == (2, 2)
    assert bool((model.rgc.bipolar_current_gain() >= 0).all())
    assert bool((model.rgc.amacrine_current_gain() >= 0).all())
    assert not hasattr(model.rgc, "synaptic_gain_raw")
    assert not hasattr(model.rgc, "bipolar_readout_gain")
    assert not hasattr(model.rgc, "amacrine_readout_gain")


def test_v3_total_rf_equals_physiological_core_rf() -> None:
    # Given
    model = _model(
        type_ids=("midget", "parasol"),
        enable_response_bias=True,
        readout_mode="v3_mechanism_preserving",
    )
    sequence = torch.randn(2, 20, 3, requires_grad=True)
    observed = torch.zeros(2, 20, 2)

    # When
    paths, _ = forward_sequence_readout_paths(
        model,
        sequence,
        observed_counts=observed,
    )
    total_rf = torch.autograd.grad(paths.total[:, -1].sum(), sequence, retain_graph=True)[0]
    core_rf = torch.autograd.grad(paths.core[:, -1].sum(), sequence)[0]

    # Then
    torch.testing.assert_close(total_rf, core_rf, atol=2e-6, rtol=0.0)
    assert int(torch.count_nonzero(paths.bipolar_direct)) == 0
    assert int(torch.count_nonzero(paths.amacrine_direct)) == 0


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
        enable_response_bias=False,
        enable_synaptic_gain=False,
        synaptic_gain_min=0.1,
        synaptic_gain_max=4.0,
        synaptic_gain_init=1.0,
    )

    # Then
    assert model.rgc.threshold.type_base_raw.shape == (2,)
    assert model.rgc.threshold.cell_residual_raw.shape == (2,)
    assert not hasattr(model.rgc, "response_bias")
    assert not hasattr(model.rgc, "synaptic_gain_raw")
    assert not hasattr(model.rgc, "bipolar_readout_gain")
    assert not hasattr(model.rgc, "amacrine_readout_gain")


def test_v2_readout_parameters_are_per_cell_when_type_blind() -> None:
    # Given / When
    model = _model(
        parameter_sharing_mode="type_blind",
        enable_response_bias=True,
        enable_synaptic_gain=True,
    )

    # Then
    assert model.rgc.response_bias.shape == (4,)
    assert model.rgc.synaptic_gain_raw.shape == (4,)
    torch.testing.assert_close(model.rgc.response_bias, torch.zeros(4))
    torch.testing.assert_close(model.rgc.synaptic_gain(), torch.ones(4))
    assert model.rgc.threshold.type_base_raw.numel() == 1


def test_direct_readout_parameters_are_zero_initialized_per_channel_and_cell() -> None:
    # Given / When
    model = _model(
        parameter_sharing_mode="type_blind",
        enable_direct_readout=True,
    )

    # Then
    assert model.rgc.bipolar_readout_gain.shape == (2, 4)
    assert model.rgc.amacrine_readout_gain.shape == (2, 4)
    torch.testing.assert_close(
        model.rgc.bipolar_readout_gain,
        torch.zeros(2, 4),
    )
    torch.testing.assert_close(
        model.rgc.amacrine_readout_gain,
        torch.zeros(2, 4),
    )


def test_direct_readout_adds_current_bipolar_and_amacrine_features_to_logits() -> None:
    # Given
    model = _model(
        type_ids=("midget", "parasol"),
        enable_direct_readout=True,
    )
    rgc = model.rgc
    previous = rgc.initial_state(1, torch.device("cpu"), torch.float32)
    spatial_weights = torch.eye(2)
    bipolar = torch.tensor(
        [[[[1.0, 2.0], [0.5, 0.25]], [[3.0, 4.0], [0.1, 0.2]]]]
    )
    amacrine = torch.tensor(
        [[[[0.3, 0.4], [0.2, 0.1]], [[0.6, 0.8], [0.05, 0.15]]]]
    )
    with torch.no_grad():
        rgc.bipolar_readout_gain.copy_(torch.tensor([[0.2, -0.1], [0.3, 0.4]]))
        rgc.amacrine_readout_gain.copy_(torch.tensor([[-0.5, 0.6], [0.7, -0.8]]))

    # When
    output, _ = rgc(bipolar, amacrine, previous, spatial_weights)

    # Then
    pooled_bipolar = torch.einsum("uc,bpkc->bpku", spatial_weights, bipolar)
    pooled_amacrine = torch.einsum("uc,bpkc->bpku", spatial_weights, amacrine)
    polarity_index = rgc.cell_polarities.view(1, 1, 1, -1).expand(1, 1, 2, -1)
    selected_bipolar = pooled_bipolar.gather(1, polarity_index).squeeze(1)
    selected_amacrine = pooled_amacrine.gather(1, polarity_index).squeeze(1)
    direct_logits = (
        rgc.bipolar_readout_gain.unsqueeze(0) * selected_bipolar
        + rgc.amacrine_readout_gain.unsqueeze(0) * selected_amacrine
    ).sum(dim=1)
    expected = rgc.logits_from_generator(output.generator_potential) + direct_logits
    torch.testing.assert_close(output.spike_logits, expected)


def test_response_bias_shifts_logits_without_changing_threshold() -> None:
    # Given
    model = _model(enable_response_bias=True)
    generator = torch.tensor([[0.2, 0.3, 0.4, 0.5]])
    with torch.no_grad():
        model.rgc.response_bias.copy_(torch.tensor([0.25, -0.5, 0.75, -1.0]))

    # When
    logits = model.rgc.logits_from_generator(generator)

    # Then
    expected = model.rgc.response_bias + 5.0 * (
        generator - model.rgc.threshold().view(1, -1)
    )
    torch.testing.assert_close(logits, expected)


def test_synaptic_gain_scales_excitatory_drive_before_inhibition() -> None:
    # Given
    model = _model(type_ids=("midget", "parasol"), enable_synaptic_gain=True)
    rgc = model.rgc
    previous = rgc.initial_state(1, torch.device("cpu"), torch.float32)
    spatial_weights = torch.eye(2)
    bipolar = torch.tensor([[[[1.0, 2.0], [0.5, 0.25]], [[3.0, 4.0], [0.1, 0.2]]]])
    amacrine = torch.zeros_like(bipolar)
    with torch.no_grad():
        fraction = (2.0 - 0.1) / (4.0 - 0.1)
        rgc.synaptic_gain_raw.fill_(torch.logit(torch.tensor(fraction)))

    # When
    output, _ = rgc(bipolar, amacrine, previous, spatial_weights)

    # Then
    pooled = torch.einsum("uc,bpkc->bpku", spatial_weights, bipolar)
    polarity_index = rgc.cell_polarities.view(1, 1, 1, -1).expand(1, 1, 2, -1)
    selected = pooled.gather(1, polarity_index).squeeze(1)
    subunit_leak = torch.exp(-rgc._dt_ms / rgc.subunit_tau_ms()).view(1, 1, -1)
    energy = subunit_leak * previous.subunit_energy + (1 - subunit_leak) * selected.square()
    adapted = selected / (1 + rgc.subunit_gain().view(1, 1, -1) * energy)
    mix = rgc.sustained_mix().view(1, -1)
    drive = mix * adapted[:, 0] + (1 - mix) * adapted[:, 1]
    membrane_leak = torch.exp(-rgc._dt_ms / rgc.membrane_tau_ms()).view(1, -1)
    expected = (1 - membrane_leak) * (2.0 * drive)
    torch.testing.assert_close(output.generator_potential, expected)


def test_bias_and_synaptic_gain_do_not_change_physiology_prior_penalty() -> None:
    # Given
    model = _model(enable_response_bias=True, enable_synaptic_gain=True)
    baseline = model.rgc.physiology_prior_penalty()

    # When
    with torch.no_grad():
        model.rgc.response_bias.add_(3.0)
        model.rgc.synaptic_gain_raw.add_(1.0)

    # Then
    torch.testing.assert_close(model.rgc.physiology_prior_penalty(), baseline)


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
        enable_response_bias=False,
        enable_synaptic_gain=False,
        synaptic_gain_min=0.1,
        synaptic_gain_max=4.0,
        synaptic_gain_init=1.0,
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
        enable_response_bias=False,
        enable_synaptic_gain=False,
        synaptic_gain_min=0.1,
        synaptic_gain_max=4.0,
        synaptic_gain_init=1.0,
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
        enable_response_bias=False,
        enable_synaptic_gain=False,
        synaptic_gain_min=0.1,
        synaptic_gain_max=4.0,
        synaptic_gain_init=1.0,
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
