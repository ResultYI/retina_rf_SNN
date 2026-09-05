from __future__ import annotations

import math

import torch

from data.retinal_recording import RealSequenceSplit
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MechanisticRetinaConfig,
)
from models.mechanistic_retina.pathway_spatial_geometry import PathwaySpatialGeometry
from models.mechanistic_retina.model import (
    MechanisticGraphTemporalRetina,
    build_mechanistic_retina,
)
from training.mechanistic_retina.optimizer import phase1_parameters
from training.mechanistic_retina.real_early_stopping import (
    EarlyStoppingConfig,
    EarlyStoppingTrainingRequest,
    fit_real_spike_model_early_stopping,
)


def _model(*, cell_specific_gains: bool) -> MechanisticGraphTemporalRetina:
    geometry = PathwaySpatialGeometry(
        spatial_basis=torch.tensor(
            [[[0.7, 0.2, 0.1], [0.5, 0.3, 0.2]], [[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]]]
        ),
        bc_support=torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
        ac_support=torch.ones(2, 3),
    )
    return build_mechanistic_retina(
        MechanisticRetinaConfig(
            architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE,
            lag_steps=3,
            h1_radius_deg=2.0,
            cell_specific_gains=cell_specific_gains,
        ),
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        torch.tensor([[0.0, 0.0], [2.0, 0.0]]),
        ("midget", "parasol"),
        ("ON", "OFF"),
        pathway_spatial_geometry=geometry,
    )


def _split(cones: torch.Tensor, events: torch.Tensor) -> RealSequenceSplit:
    return RealSequenceSplit(
        cone_drive=cones,
        spike_counts=events.to(torch.int64),
        spike_events=events,
        valid_mask=torch.ones_like(events, dtype=torch.bool),
        source_image_ids=tuple(f"image-{index}" for index in range(cones.shape[0])),
        trial_indices=tuple(range(cones.shape[0])),
    )


def test_cell_specific_gains_start_positive_at_one_and_enter_optimizer() -> None:
    # Given: a mechanism-identifiable model with the optional physiological gains enabled.
    model = _model(cell_specific_gains=True)

    # When: the gain values and canonical optimizer contract are inspected.
    parameters = phase1_parameters(model)

    # Then: two positive unit gains exist per cell and both raw vectors are optimized.
    assert model.cell_gains is not None
    assert torch.equal(model.cell_gains.bc, torch.ones(2))
    assert torch.equal(model.cell_gains.ac, torch.ones(2))
    assert any(parameter is model.cell_gains.log_bc for parameter in parameters)
    assert any(parameter is model.cell_gains.log_ac for parameter in parameters)
    assert model.cell_gains.log_bc.numel() + model.cell_gains.log_ac.numel() == 4


def test_disabled_cell_specific_gains_preserve_v1_state_dict_contract() -> None:
    # Given: the unchanged V1 configuration with no cell-specific gains requested.
    model = _model(cell_specific_gains=False)

    # When: the model state contract is inspected.
    state_keys = tuple(model.state_dict())

    # Then: no new gain tensors are present in V1 checkpoints.
    assert model.cell_gains is None
    assert not any(key.startswith("cell_gains.") for key in state_keys)


def test_cell_specific_gains_scale_bc_and_ac_without_sign_reversal() -> None:
    # Given: fixed stimuli and a gain-enabled model at the unit-gain initialization.
    torch.manual_seed(17)
    model = _model(cell_specific_gains=True)
    assert model.cell_gains is not None
    cones = torch.randn(3, 8, 3)
    history = torch.zeros(3, 8, 2)
    unit = model.forward_sequence(cones, observed_counts=history)

    # When: BC and AC gains are independently raised to two and three.
    with torch.no_grad():
        model.cell_gains.log_bc.fill_(math.log(2.0))
        model.cell_gains.log_ac.fill_(math.log(3.0))
    scaled = model.forward_sequence(cones, observed_counts=history)

    # Then: only the intended signed pathway currents receive those positive scales.
    assert torch.allclose(scaled.bc_sustained_current, 2 * unit.bc_sustained_current)
    assert torch.allclose(scaled.bc_transient_current, 2 * unit.bc_transient_current)
    assert torch.allclose(scaled.amacrine_local_current, 3 * unit.amacrine_local_current)
    assert torch.allclose(
        scaled.amacrine_transient_current,
        3 * unit.amacrine_transient_current,
    )
    assert bool(
        (scaled.amacrine_local_current * scaled.amacrine_local_state <= 0).all()
    )
    expected_total = (
        scaled.bc_sustained_current
        + scaled.bc_transient_current
        + scaled.amacrine_local_current
        + scaled.amacrine_transient_current
    )
    assert torch.allclose(scaled.total_current, expected_total)


def test_bc_gain_preserves_named_states_and_scales_public_currents() -> None:
    # Given: the gain-enabled model and its unit-gain named BC decomposition.
    torch.manual_seed(19)
    model = _model(cell_specific_gains=True)
    assert model.cell_gains is not None
    cones = torch.randn(3, 8, 3)
    history = torch.zeros(3, 8, 2)
    unit = model.forward_sequence(cones, observed_counts=history)

    # When: the cell-specific BC gain is doubled without changing pathway states.
    with torch.no_grad():
        model.cell_gains.log_bc.fill_(math.log(2.0))
    scaled = model.forward_sequence(cones, observed_counts=history)

    # Then: public states stay raw, while currents scale and recompose exactly.
    for state_name, current_name in (
        ("on_sustained_state", "on_sustained_current"),
        ("on_transient_state", "on_transient_current"),
        ("off_sustained_state", "off_sustained_current"),
        ("off_transient_state", "off_transient_current"),
    ):
        assert torch.equal(getattr(scaled, state_name), getattr(unit, state_name))
        assert torch.allclose(
            getattr(scaled, current_name), 2 * getattr(unit, current_name)
        )
    assert torch.allclose(
        scaled.bc_sustained_current,
        scaled.on_sustained_current + scaled.off_sustained_current,
    )
    assert torch.allclose(
        scaled.bc_transient_current,
        scaled.on_transient_current + scaled.off_transient_current,
    )


def test_early_stopping_audits_every_cell_specific_gain() -> None:
    # Given: sampled binary targets for a small gain-enabled population.
    torch.manual_seed(23)
    model = _model(cell_specific_gains=True)
    cones = torch.randn(12, 10, 3)
    events = torch.bernoulli(torch.full((12, 10, 2), 0.2))

    # When: the unchanged early-stopping trainer runs from fresh initialization.
    result = fit_real_spike_model_early_stopping(
        EarlyStoppingTrainingRequest(
            model=model,
            train=_split(cones[:8], events[:8]),
            validation=_split(cones[8:], events[8:]),
            learning_rate=0.01,
            batch_size=4,
            seed=29,
            stopping=EarlyStoppingConfig(
                max_steps=8,
                evaluation_interval=1,
                patience=4,
                min_delta=0.0,
            ),
        )
    )

    # Then: every BC and AC gain receives signal and differs in the restored best state.
    audit = result.cell_gain_audit
    assert audit is not None
    assert tuple(pathway.name for pathway in audit.pathways) == ("BC", "AC")
    assert all(pathway.all_gradient_nonzero for pathway in audit.pathways)
    assert all(pathway.all_best_updated for pathway in audit.pathways)
    assert all(pathway.min_peak_abs_gradient > 0 for pathway in audit.pathways)
    assert all(value > 0 for pathway in audit.pathways for value in pathway.best)
