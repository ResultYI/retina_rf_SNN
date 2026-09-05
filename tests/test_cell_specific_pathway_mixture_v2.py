from __future__ import annotations

import math

import torch

from data.retinal_recording import RealSequenceSplit
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MechanisticRetinaConfig,
)
from models.mechanistic_retina.model import (
    MechanisticGraphTemporalRetina,
    build_mechanistic_retina,
)
from models.mechanistic_retina.pathway_spatial_geometry import (
    PathwaySpatialGeometry,
)
from training.mechanistic_retina.optimizer import phase1_parameters
from training.mechanistic_retina.real_early_stopping import (
    EarlyStoppingConfig,
    EarlyStoppingTrainingRequest,
    fit_real_spike_model_early_stopping,
)


def _model() -> MechanisticGraphTemporalRetina:
    geometry = PathwaySpatialGeometry(
        spatial_basis=torch.tensor(
            [
                [[0.7, 0.2, 0.1], [0.5, 0.3, 0.2]],
                [[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]],
            ]
        ),
        bc_support=torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
        ac_support=torch.ones(2, 3),
    )
    return build_mechanistic_retina(
        MechanisticRetinaConfig(
            architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE,
            lag_steps=5,
            h1_radius_deg=2.0,
            cell_specific_pathway_mixture=True,
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


def test_pathway_mixture_starts_at_unit_gain_and_enters_optimizer() -> None:
    # Given: the V2 pathway-mixture configuration for two RGCs.
    model = _model()

    # When: its gain values and optimizer coverage are inspected.
    gains = model.cell_gains
    parameters = phase1_parameters(model)

    # Then: four strictly positive unit gains per cell are independently optimized.
    assert gains is not None
    assert gains.pathway_names == (
        "BC_sustained",
        "BC_transient",
        "AC_local",
        "AC_transient",
    )
    torch.testing.assert_close(gains.pathway_values, torch.ones(2, 4))
    assert all(
        any(raw is listed for listed in parameters) for raw in gains.raw_parameters
    )
    assert sum(raw.numel() for raw in gains.raw_parameters) == 8


def test_pathway_mixture_scales_four_currents_without_changing_signs() -> None:
    # Given: a V2 model and its unit-gain pathway decomposition.
    torch.manual_seed(31)
    model = _model()
    gains = model.cell_gains
    assert gains is not None
    cones = torch.randn(3, 8, 3)
    history = torch.zeros(3, 8, 2)
    unit = model.forward_sequence(cones, observed_counts=history)

    # When: each cell-specific pathway gain receives a distinct positive value.
    with torch.no_grad():
        for raw, value in zip(gains.raw_parameters, (2.0, 3.0, 4.0, 5.0), strict=True):
            raw.fill_(math.log(value))
    scaled = model.forward_sequence(cones, observed_counts=history)

    # Then: only its named contribution scales, while E/I signs and states persist.
    for name, factor in (
        ("bc_sustained_current", 2.0),
        ("bc_transient_current", 3.0),
        ("amacrine_local_current", 4.0),
        ("amacrine_transient_current", 5.0),
    ):
        torch.testing.assert_close(getattr(scaled, name), factor * getattr(unit, name))
    for name in (
        "on_sustained_state",
        "on_transient_state",
        "off_sustained_state",
        "off_transient_state",
        "amacrine_local_state",
        "amacrine_transient_state",
    ):
        torch.testing.assert_close(getattr(scaled, name), getattr(unit, name))
    assert bool((scaled.amacrine_local_current * scaled.amacrine_local_state <= 0).all())
    assert bool(
        (scaled.amacrine_transient_current * scaled.amacrine_transient_state <= 0).all()
    )


def test_early_stopping_audits_every_pathway_mixture_gain() -> None:
    # Given: sampled binary targets for a small V2 population.
    torch.manual_seed(37)
    model = _model()
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
            seed=41,
            stopping=EarlyStoppingConfig(
                max_steps=8,
                evaluation_interval=1,
                patience=4,
                min_delta=0.0,
            ),
        )
    )

    # Then: every cell in every named pathway has signal and changes at best step.
    audit = result.cell_gain_audit
    assert audit is not None
    assert tuple(pathway.name for pathway in audit.pathways) == (
        "BC_sustained",
        "BC_transient",
        "AC_local",
        "AC_transient",
    )
    assert all(pathway.all_gradient_nonzero for pathway in audit.pathways)
    assert all(pathway.all_best_updated for pathway in audit.pathways)
    assert all(pathway.min_peak_abs_gradient > 0 for pathway in audit.pathways)
    assert all(value > 0 for pathway in audit.pathways for value in pathway.best)
