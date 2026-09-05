from __future__ import annotations

import torch

from evaluation.mechanistic_retina.rf_effective import effective_rf
from models.mechanistic_retina.contracts import MechanisticRetinaConfig, PathwayClamp
from models.mechanistic_retina.model import build_mechanistic_retina


def _model():
    cone_positions = torch.tensor(
        [[0.00, 0.00], [0.04, 0.00], [0.08, 0.00], [0.13, 0.00], [0.18, 0.00]]
    )
    cell_positions = torch.tensor([[0.00, 0.00], [0.04, 0.00]])
    config = MechanisticRetinaConfig(architecture_mode="mechanism_identifiable")
    return build_mechanistic_retina(
        config,
        cone_positions,
        cell_positions,
        ("midget", "midget"),
        ("ON", "ON"),
    )


def test_normal_mechanism_gates_start_inside_the_differentiable_interval() -> None:
    model = _model()
    cones = torch.randn(1, 20, 5)
    history = torch.zeros(1, 20, 2)

    output = model.forward_sequence(cones, observed_counts=history)

    torch.testing.assert_close(model.gates.h1, model.gates.h1.new_tensor(0.01))
    assert float(model.gates.ac_local) == 0.5
    assert float(model.gates.ac_transient) == 0.5
    assert float(model.gates.history) == 0.5
    assert all(torch.isfinite(value).all() for value in output.tensors())
    assert torch.count_nonzero(output.h1_surround_contribution) > 0
    assert torch.count_nonzero(output.amacrine_local_current) > 0
    assert torch.count_nonzero(output.amacrine_transient_current) > 0


def test_support_partition_and_shared_connectivity_are_structural() -> None:
    # Given: nested BC/AC disks and a farther H1 support.
    model = _model()
    supports = model.feature_bank.supports
    connection = model.shared_subunits.connection_matrix()

    # When: the fixed supports and shared graph are inspected.
    overlap = supports.bc * supports.ac

    # Then: AC contains BC, H1 extends farther, and one subunit is shared.
    assert torch.equal(overlap, supports.bc)
    assert bool((supports.ac.sum(dim=1) > supports.bc.sum(dim=1)).all())
    assert bool((supports.h1.sum(dim=1) > supports.ac.sum(dim=1)).all())
    assert bool((connection > 0).sum(dim=0).max() > 1)
    assert model.bipolar.raw_weights.shape[0] == 1


def test_h1_is_shared_ac_is_inhibitory_and_history_has_no_spatial_coupling() -> None:
    # Given: present H1/AC gates and a history impulse in only the first RGC.
    model = _model()
    with torch.no_grad():
        model.gates.set_h1_amplitude_(0.01)
        model.gates.ac_local.fill_(1.0)
        model.gates.ac_transient.fill_(1.0)
        model.gates.history.fill_(1.0)
    cones = torch.rand(1, 20, 5)
    history = torch.zeros(1, 20, 2)
    changed = history.clone()
    changed[:, 5, 0] = 1.0

    # When: both histories are evaluated.
    base = model.forward_sequence(cones, observed_counts=history)
    perturbed = model.forward_sequence(cones, observed_counts=changed)

    # Then: H1 has cone-shared state, AC current opposes its state, and history stays cell-local.
    assert base.h1_state.shape == cones.shape
    assert bool((base.amacrine_local_current * base.amacrine_local_state <= 1e-8).all())
    assert bool((base.amacrine_transient_current * base.amacrine_transient_state <= 1e-8).all())
    assert torch.equal(base.logits[..., 1], perturbed.logits[..., 1])


def test_current_and_rf_decompositions_close() -> None:
    # Given: an active mechanism-identifiable model.
    model = _model()
    with torch.no_grad():
        for parameter in model.gates.parameters():
            parameter.fill_(0.7)
    cones = torch.rand(1, 20, 5)
    history = torch.zeros(1, 20, 2)

    # When: currents and structural pathway RFs are decomposed.
    output = model.forward_sequence(cones, observed_counts=history)
    full_rf = effective_rf(model, cones, history)
    h1_off_rf = effective_rf(
        model,
        cones,
        history,
        clamps=frozenset({PathwayClamp.H1}),
    )
    bc_rf = effective_rf(
        model,
        cones,
        history,
        clamps=frozenset(
            {
                PathwayClamp.H1,
                PathwayClamp.AMACRINE_LOCAL,
                PathwayClamp.AMACRINE_TRANSIENT,
            }
        ),
    )
    pathways = {
        "BC": bc_rf,
        "AC": h1_off_rf - bc_rf,
        "H1": full_rf - h1_off_rf,
    }
    total_rf = full_rf

    # Then: both decompositions reconstruct their totals within the frozen tolerance.
    current_sum = (
        output.bc_sustained_current
        + output.bc_transient_current
        + output.amacrine_local_current
        + output.amacrine_transient_current
    )
    assert float((current_sum - output.total_current).abs().max()) <= 1e-8
    assert float((sum(pathways.values(), torch.zeros_like(total_rf)) - total_rf).abs().max()) <= 2e-6


def test_structural_clamps_remove_h1_and_ac_rf() -> None:
    # Given: an active model and fixed validation stimulus.
    model = _model()
    with torch.no_grad():
        model.gates.set_h1_amplitude_(0.01)
        model.gates.ac_local.fill_(1.0)
        model.gates.ac_transient.fill_(1.0)
    cones = torch.rand(1, 20, 5)
    history = torch.zeros(1, 20, 2)

    # When: H1 and both AC paths are structurally ablated.
    no_h1 = model.forward_sequence(
        cones, observed_counts=history, clamps=frozenset({PathwayClamp.H1})
    )
    no_ac = model.forward_sequence(
        cones,
        observed_counts=history,
        clamps=frozenset({PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}),
    )

    # Then: the corresponding observable contributions are exactly zero.
    assert float(no_h1.h1_surround_contribution.abs().max()) <= 1e-8
    assert float(no_ac.amacrine_local_current.abs().max()) <= 1e-8
    assert float(no_ac.amacrine_transient_current.abs().max()) <= 1e-8


def test_functional_call_reaches_canonical_model_forward() -> None:
    # Given: the exact stateless call used by the Fisher-Jacobian diagnostic.
    model = _model()
    cones = torch.rand(1, 20, 5)
    history = torch.zeros(1, 20, 2)
    from torch.func import functional_call

    # When: the raw H1 amplitude is functionally overridden.
    output = functional_call(
        model,
        {
            "gates.raw_h1_amplitude": model.gates.raw_h1_amplitude.detach()
            .clone()
            .requires_grad_(True)
        },
        (cones,),
        {"observed_counts": history},
        strict=False,
    )

    # Then: the call reaches the canonical model path and returns finite logits.
    assert bool(torch.isfinite(output.logits).all())


def test_fisher_subspace_diagnostic_runs_through_model_entrypoint() -> None:
    # Given: the same model/input boundary consumed by the P0 Fisher diagnostic.
    model = _model()
    cones = torch.rand(1, 20, 5)
    history = torch.zeros(1, 20, 2)
    from evaluation.mechanistic_retina.subspace_overlap import (
        SubspaceOverlapRequest,
        fisher_subspace_overlap,
    )

    # When: pathway Jacobians are measured through stateless calls.
    result = fisher_subspace_overlap(
        SubspaceOverlapRequest(model, cones, history, tail_steps=4)
    )

    # Then: all pathway pairs and requested outputs are present and finite.
    assert len(result.pairs) == 3
    assert result.output_count == 8
    assert all(
        torch.isfinite(torch.tensor(pair.maximum_canonical_correlation))
        for pair in result.pairs
    )
