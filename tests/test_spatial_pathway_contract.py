from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from evaluation.mechanistic_retina.pathway_decomposition import pathway_base_rf
from evaluation.mechanistic_retina.mechanism_teacher_support import preregistered_probe_inputs
from evaluation.mechanistic_retina.rf_base import base_rf
from evaluation.mechanistic_retina.rf_effective import effective_rf
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MechanisticRetinaConfig,
    PathwayClamp,
)
from models.mechanistic_retina.model import (
    MechanisticGraphTemporalRetina,
    build_mechanistic_retina,
)
from models.mechanistic_retina.support_partition import (
    SupportPartition,
    partition_spatial_basis,
)


def _model() -> MechanisticGraphTemporalRetina:
    return build_mechanistic_retina(
        MechanisticRetinaConfig(
            architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE,
            cell_specific_gains=True,
        ),
        torch.tensor(((0.0, 0.0), (0.04, 0.0), (0.08, 0.0),
                      (0.12, 0.0), (0.14, 0.0), (0.18, 0.0))),
        torch.zeros(2, 2),
        ("midget", "parasol"),
        ("ON", "OFF"),
    )


def test_bc_disk_is_strict_subset_of_full_ac_disk_at_unchanged_radii() -> None:
    # Given: both cell types sampled across the unchanged radii.
    bank = _model().feature_bank
    distances = torch.tensor((0.0, 0.04, 0.08, 0.12, 0.14, 0.18))

    # When: the actual pathway masks are inspected.
    bc, ac = bank.bc_support.bool(), bank.ac_support.bool()

    # Then: AC includes the center and every BC cone, plus an outer region.
    assert torch.equal(bc, distances[None] <= torch.tensor((0.06, 0.10))[:, None])
    assert torch.equal(ac, distances[None] <= torch.tensor((0.13, 0.15))[:, None])
    assert not bool((bc & ~ac).any())
    assert bool((bc & ac).any(dim=1).all())
    assert bool((ac & ~bc).any(dim=1).all())
    assert bool(ac[:, 0].all())


def test_pathway_bases_preserve_scales_and_normalize_on_separate_supports() -> None:
    # Given: the original two Gaussian scales for each cell type.
    bank = _model().feature_bank
    distance = torch.tensor((0.0, 0.04, 0.08, 0.12, 0.14, 0.18))
    sigma = torch.tensor(((0.05, 0.14), (0.09, 0.20)))
    gaussian = torch.exp(-0.5 * (distance[None, None] / sigma[:, :, None]) ** 2)

    # When: each pathway's effective spatial basis is reconstructed.
    bc = gaussian * bank.bc_support[:, None]
    ac = gaussian * bank.ac_support[:, None]
    expected = torch.stack((bc, bc, ac, ac), dim=1)
    expected = expected / expected.sum(dim=-1, keepdim=True)

    # Then: every basis has unit mass, with distinct BC/AC support and shape.
    torch.testing.assert_close(bank.path_spatial_basis, expected)
    torch.testing.assert_close(bank.path_spatial_basis.sum(-1), torch.ones(2, 4, 2))
    assert bool((bank.path_spatial_basis[:, 2:, :, 0] > 0).all())
    assert not torch.allclose(bank.path_spatial_basis[:, :2], bank.path_spatial_basis[:, 2:])


def test_only_ac_spatial_drive_changes_against_explicit_old_annulus_reference() -> None:
    # Given: identical parameters, with a test-only copy of the old annular mask.
    model = _model()
    old = deepcopy(model)
    old.feature_bank.ac_support.mul_(1 - old.feature_bank.bc_support)
    old.feature_bank.path_spatial_basis.copy_(partition_spatial_basis(
        old.feature_bank.spatial_basis,
        SupportPartition(old.feature_bank.bc_support, old.feature_bank.ac_support,
                         old.feature_bank.h1_support),
    ))
    cones = torch.randn(2, 22, 6)
    history = torch.zeros(2, 22, 2)
    ac_off = frozenset({PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT})

    # When: the only changed support is removed by structural AC clamp.
    normal = model.forward_sequence(cones, observed_counts=history)
    old_normal = old.forward_sequence(cones, observed_counts=history)
    clamped = model.forward_sequence(cones, observed_counts=history, clamps=ac_off)
    old_clamped = old.forward_sequence(cones, observed_counts=history, clamps=ac_off)

    # Then: BC/H1 and AC-off logits stay bit-identical; normal AC is allowed to differ.
    assert torch.equal(normal.bc_sustained_current, old_normal.bc_sustained_current)
    assert torch.equal(normal.bc_transient_current, old_normal.bc_transient_current)
    assert torch.equal(normal.h1_surround_contribution, old_normal.h1_surround_contribution)
    assert torch.equal(clamped.logits, old_clamped.logits)
    assert not torch.equal(normal.amacrine_local_current, old_normal.amacrine_local_current)


def test_forward_remains_causal_for_stimulus_and_strictly_past_history() -> None:
    # Given: two inputs identical through the queried time and history before it.
    model = _model()
    cones = torch.randn(2, 22, 6)
    history = torch.zeros(2, 22, 2)
    changed_cones, changed_history = cones.clone(), history.clone()
    changed_cones[:, 13:] += 10
    changed_history[:, 12:] = 1

    # When: future stimulus and current/future spike targets are changed.
    original = model.forward_sequence(cones, observed_counts=history)
    changed = model.forward_sequence(changed_cones, observed_counts=changed_history)

    # Then: no current target or future input reaches earlier logits.
    assert torch.equal(original.logits[:, :13], changed.logits[:, :13])


@pytest.mark.parametrize("clamps", (
    frozenset(),
    frozenset({PathwayClamp.H1}),
    frozenset({PathwayClamp.DIRECT_BC_SUSTAINED, PathwayClamp.DIRECT_BC_TRANSIENT}),
    frozenset({PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}),
))
def test_pathway_rf_helpers_match_forward_jacobian_under_structural_clamps(
    clamps: frozenset[PathwayClamp],
) -> None:
    # Given: a full lag window, unit gains, and one fixed structural condition.
    model = _model()
    cones = torch.randn(1, model.config.lag_steps, 6, requires_grad=True)
    history = torch.zeros(1, model.config.lag_steps, 2)

    # When: RF helpers and true current Jacobians use the identical spatial contract.
    output = model.forward_sequence(cones, observed_counts=history, clamps=clamps)
    currents = (output.bc_sustained_current, output.bc_transient_current,
                output.amacrine_local_current, output.amacrine_transient_current)
    helper = pathway_base_rf(model, clamps=clamps)
    actual = model.pathway_base_rfs(clamps=clamps)
    for current, kernel, named in zip(currents, actual, helper.values(), strict=True):
        expected = torch.stack(tuple(
            torch.autograd.grad(current[0, -1, cell], cones, retain_graph=True)[0][0]
            if current.requires_grad else torch.zeros_like(cones[0])
            for cell in range(2)
        ))
        torch.testing.assert_close(kernel, expected, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(named, kernel, atol=0, rtol=0)

    # Then: base RF is the current-sum Jacobian, and clamped contributions are exact zero.
    torch.testing.assert_close(base_rf(model, clamps=clamps), sum(actual))
    if PathwayClamp.H1 in clamps:
        assert torch.count_nonzero(output.h1_surround_contribution) == 0
    if PathwayClamp.DIRECT_BC_SUSTAINED in clamps:
        assert all(torch.count_nonzero(value) == 0 for value in currents[:2] + actual[:2])
    if PathwayClamp.AMACRINE_LOCAL in clamps:
        assert all(torch.count_nonzero(value) == 0 for value in currents[2:] + actual[2:])


def test_global_rf_matches_forward_finite_difference() -> None:
    # Given: a nonzero context and a direction covering both shared and outer support.
    model = _model().double()
    cones = torch.linspace(-0.2, 0.3, 16 * 6, dtype=torch.float64).reshape(1, 16, 6)
    history = torch.zeros(1, 16, 2, dtype=torch.float64)
    direction = torch.cos(torch.arange(cones.numel(), dtype=torch.float64)).reshape_as(cones)

    # When: the true logit RF and symmetric finite difference are evaluated.
    kernel = effective_rf(model, cones, history)
    epsilon = 1e-5
    plus = model.forward_sequence(cones + epsilon * direction, observed_counts=history)
    minus = model.forward_sequence(cones - epsilon * direction, observed_counts=history)
    finite_difference = (plus.logits[:, -1] - minus.logits[:, -1]) / (2 * epsilon)

    # Then: the logit Jacobian includes the changed AC support without a surrogate RF.
    torch.testing.assert_close((kernel * direction[:, None]).sum((-1, -2)),
                               finite_difference, rtol=1e-5, atol=1e-9)


@pytest.mark.parametrize("h1", (False, True))
def test_diagnostic_stimulus_geometry_and_cell_selection_preserve_old_contract(
    h1: bool,
) -> None:
    # Given: the same geometry with full-disk versus historical annular AC support.
    model = _model()
    old = deepcopy(model)
    old.feature_bank.ac_support.mul_(1 - old.feature_bank.bc_support)
    data = SimpleNamespace(cell_ids=("one", "two"))

    # When: existing probes are built, without running models or generating targets.
    actual = preregistered_probe_inputs(model, data, h1)
    reference = preregistered_probe_inputs(old, data, h1)

    # Then: pathway support changes do not redefine stimulus surround or select another cell.
    for new_probe, old_probe in zip(actual, reference, strict=True):
        assert new_probe.cell == old_probe.cell
        assert new_probe.preregistered_name == old_probe.preregistered_name
        assert torch.equal(new_probe.stimulus, old_probe.stimulus)
        assert torch.equal(new_probe.history, old_probe.history)
