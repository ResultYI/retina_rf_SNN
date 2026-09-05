from __future__ import annotations

import pytest
import torch

from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.h1_pathway import H1Output
from models.mechanistic_retina.model import (
    MechanisticGraphTemporalRetina,
    build_mechanistic_retina,
)
from models.mechanistic_retina.shared_subunits import SharedSubunitLayout, SharedSubunitMixer
from training.mechanistic_retina.optimizer import build_phase1_optimizer


def _audit_model() -> MechanisticGraphTemporalRetina:
    axis = torch.arange(-6, 7, dtype=torch.float64) * 0.04
    with torch.random.fork_rng():
        torch.manual_seed(831)
        return build_mechanistic_retina(
            MechanisticRetinaConfig(architecture_mode="mechanism_identifiable"),
            torch.cartesian_prod(axis, axis),
            torch.tensor(
                ((0, 0), (0.04, 0), (0, 0.04), (0, 0), (0.04, 0), (0, 0.04)),
                dtype=torch.float64,
            ),
            ("midget", "midget", "midget", "parasol", "parasol", "parasol"),
            ("ON", "ON", "OFF", "ON", "ON", "OFF"),
        ).eval()


@pytest.mark.parametrize("broad", (False, True), ids=("direct", "broad"))
def test_target_support_derivative_is_zero_when_audited_six_cells_mix(broad: bool) -> None:
    # Given: the original six-cell audit fixture and its actual H1 output.
    model = _audit_model()
    count = model.feature_bank.spatial_basis.shape[-1]
    cones = torch.sin(torch.arange(40 * count, dtype=torch.float64) * 0.017)
    cones = cones.reshape(1, 40, count).requires_grad_()
    captured: list[torch.Tensor] = []

    def capture_h1(
        module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: H1Output
    ) -> None:
        captured.append(output.modulated_cones)

    hook = model.h1.register_forward_hook(capture_h1)
    try:
        output = model.forward_sequence(cones, observed_counts=torch.zeros(1, 40, 6))
    finally:
        hook.remove()
    values = output.bc_broad_presynaptic if broad else output.bc_direct_presynaptic
    support = model.feature_bank.ac_support if broad else model.feature_bank.bc_support

    # When: each final-bin BC component is differentiated through production forward.
    for cell in range(6):
        for component in range(2):
            gradient = torch.autograd.grad(
                values[0, -1, cell, component], captured[0], retain_graph=True
            )[0]
            inside = support[cell].bool()

            # Then: the target disk, not merely every source disk, bounds dependency.
            assert torch.count_nonzero(gradient[..., ~inside]) == 0
            assert torch.count_nonzero(gradient[..., inside]) > 0


def test_direct_support_is_local_when_two_nonoverlapping_bc_disks_mix() -> None:
    # Given: the minimal original two-cone counterexample.
    positions = torch.tensor(((0.0, 0.0), (0.07, 0.0)), dtype=torch.float64)
    model = build_mechanistic_retina(
        MechanisticRetinaConfig(architecture_mode="mechanism_identifiable"),
        positions,
        positions,
        ("midget", "midget"),
        ("ON", "ON"),
    )
    cones = torch.ones(1, 24, 2, dtype=torch.float64, requires_grad=True)
    captured: list[torch.Tensor] = []

    def capture_h1(
        module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: H1Output
    ) -> None:
        captured.append(output.modulated_cones)

    hook = model.h1.register_forward_hook(capture_h1)
    try:
        output = model.forward_sequence(cones, observed_counts=torch.zeros(1, 24, 2))
    finally:
        hook.remove()

    # When: the target-zero direct state is differentiated at the other cone.
    gradient = torch.autograd.grad(output.bc_direct_presynaptic[0, -1, 0].sum(), captured[0])[0]

    # Then: the cross-cell edge remains, but cannot import outside-disk stimulus.
    assert model.shared_subunits.connection_matrix()[0, 1] > 0
    assert torch.count_nonzero(gradient[..., 1]) == 0
    assert torch.count_nonzero(gradient[..., 0]) > 0


def test_self_only_rows_are_not_optimizer_parameters_when_four_groups_are_isolated() -> None:
    # Given: the audit's four self-only class/polarity groups.
    axis = torch.arange(-4, 5, dtype=torch.float64) * 0.04
    model = build_mechanistic_retina(
        MechanisticRetinaConfig(architecture_mode="mechanism_identifiable"),
        torch.cartesian_prod(axis, axis),
        torch.zeros(4, 2, dtype=torch.float64),
        ("midget", "midget", "parasol", "parasol"),
        ("ON", "OFF", "ON", "OFF"),
    )
    cones = torch.sin(torch.arange(32 * 81, dtype=torch.float64) * 0.017).reshape(1, 32, 81)
    history = torch.zeros(1, 32, 4, dtype=torch.float64)
    before = model.forward_sequence(cones, observed_counts=history).logits

    # When: the unused compatibility buffer is perturbed without an optimizer step.
    with torch.no_grad():
        model.shared_subunits.raw_connections.fill_(-1e6)
    after = model.forward_sequence(cones, observed_counts=history).logits
    optimizer = build_phase1_optimizer(model, learning_rate=0.001)
    listed = tuple(value for group in optimizer.param_groups for value in group["params"])

    # Then: literal identity rows have no raw trainable or optimizer coordinates.
    assert not tuple(model.shared_subunits.parameters())
    assert all(value is not model.shared_subunits.raw_connections for value in listed)
    assert torch.equal(model.shared_subunits.connection_matrix(), torch.eye(4, dtype=torch.float64))
    assert torch.equal(before, after)
    assert sum(value.numel() for value in model.parameters() if value.requires_grad) == 76


def test_only_multi_neighbor_edges_train_when_one_row_is_self_only() -> None:
    # Given: two mixing rows and a third fixed identity row.
    edges = torch.tensor(((0, 0, 1, 1, 2), (0, 1, 0, 1, 2)))
    mixer = SharedSubunitMixer(
        SharedSubunitLayout(torch.zeros(3, 2), ("midget",) * 3, ("ON",) * 3, edges),
        radius_deg=0.08,
        trainable=True,
    ).double()

    # When: a genuine mixing coefficient is differentiated and perturbed.
    before = mixer.connection_matrix()
    gradient = torch.autograd.grad(before[0, 1], mixer.raw_connections)[0]
    with torch.no_grad():
        mixer.raw_connections[0].add_(0.2)
    after = mixer.connection_matrix()

    # Then: all four ordinary mixing edges remain trainable, with no fifth null edge.
    assert mixer.raw_connections.numel() == 4
    assert torch.count_nonzero(gradient) == 2
    assert not torch.equal(before[0], after[0])
    assert torch.equal(before[2], torch.tensor((0.0, 0.0, 1.0), dtype=torch.float64))
    assert torch.equal(before[2], after[2])


def test_single_cell_identity_is_fixed_when_compatibility_buffer_changes() -> None:
    # Given: the existing one-cell state_dict buffer contract.
    mixer = SharedSubunitMixer(
        SharedSubunitLayout(torch.zeros(1, 2), ("midget",), ("ON",)),
        radius_deg=0.08,
        trainable=True,
    )
    assert set(mixer.state_dict()) == {"raw_connections", "edge_index", "cell_order"}
    assert mixer.raw_connections.shape == (1,)
    assert not tuple(mixer.parameters())

    # When: a legacy-compatible but semantically unused buffer value is perturbed.
    with torch.no_grad():
        mixer.raw_connections.fill_(-1e6)

    # Then: normalization cannot turn the self-only identity into zero.
    assert torch.equal(mixer.connection_matrix(), torch.ones(1, 1))


def test_loaded_edge_order_preserves_identity_rows_and_mixing_parameters() -> None:
    # Given: the same mixed-degree graph constructed with different edge orderings.
    edges = torch.tensor(((0, 0, 1, 1, 2), (0, 1, 0, 1, 2)))
    mixers = tuple(
        SharedSubunitMixer(
            SharedSubunitLayout(torch.zeros(3, 2), ("midget",) * 3, ("ON",) * 3, order),
            radius_deg=0.08,
            trainable=True,
        )
        for order in (edges, edges[:, (4, 0, 1, 2, 3)])
    )

    # When: the checkpoint carries its own edge order and corresponding raw weights.
    mixers[1].load_state_dict(mixers[0].state_dict(), strict=True)

    # Then: derived identity rows follow loaded edges rather than construction order.
    assert torch.equal(mixers[0].connection_matrix(), mixers[1].connection_matrix())


def test_audit_parameter_count_excludes_two_self_only_coordinates() -> None:
    # Given: the original N=6, G=4, E=10 audit fixture.
    model = _audit_model()

    # When: trainable scalars are counted without training.
    counts = {name: value.numel() for name, value in model.named_parameters() if value.requires_grad}

    # Then: only the two structurally ineffective self-only coordinates disappear.
    assert model.shared_subunits.edge_index.shape == (2, 10)
    assert counts["shared_subunits.raw_connections"] == 8
    assert len(counts) == 13
    assert sum(counts.values()) == 86
