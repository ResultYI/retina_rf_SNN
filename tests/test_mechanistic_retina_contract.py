from __future__ import annotations

import torch

from evaluation.mechanistic_retina.pathway_decomposition import (
    effective_pathway_rf,
    pathway_base_rf,
    pathway_output_sensitivity,
)
from evaluation.mechanistic_retina.rf_base import base_rf
from evaluation.mechanistic_retina.rf_effective import effective_rf
from models.mechanistic_retina.contracts import MechanisticRetinaConfig, PathwayClamp
from models.mechanistic_retina.model import build_mechanistic_retina


def _model():
    # Given
    cone_positions = torch.tensor(
        [[0.00, 0.00], [0.05, 0.00], [0.10, 0.00], [0.15, 0.00]],
        dtype=torch.float32,
    )
    cell_positions = torch.tensor(
        [[0.04, 0.00], [0.11, 0.00]],
        dtype=torch.float32,
    )
    return build_mechanistic_retina(
        MechanisticRetinaConfig(),
        cone_positions,
        cell_positions,
        ("midget", "parasol"),
        ("ON", "OFF"),
    )


def test_graph_is_sparse_row_normalized_and_order_preserving() -> None:
    # Given
    model = _model()

    # When
    kernel = model.h1.graph.dense_kernel()

    # Then
    assert model.h1.graph.edge_count < kernel.numel()
    assert torch.all(kernel >= 0)
    assert torch.allclose(kernel.sum(dim=1), torch.ones(kernel.shape[0]))
    assert torch.equal(model.h1.graph.node_order, torch.arange(kernel.shape[0]))


def test_forward_exposes_finite_named_states_and_exact_current_sum() -> None:
    # Given
    model = _model()
    cones = torch.randn(2, 20, 4)
    history = torch.zeros(2, 20, 2)

    # When
    output = model.forward_sequence(cones, observed_counts=history)

    # Then
    assert output.logits.shape == (2, 20, 2)
    assert all(torch.isfinite(value).all() for value in output.tensors())
    expected = (
        output.bc_sustained_current
        + output.bc_transient_current
        + output.amacrine_local_current
        + output.amacrine_transient_current
    )
    assert torch.allclose(output.total_current, expected, atol=1e-7, rtol=0)
    assert torch.count_nonzero(output.on_sustained_state[..., 1]) == 0
    assert torch.count_nonzero(output.off_sustained_state[..., 0]) == 0


def test_observed_history_updates_only_future_logits() -> None:
    # Given
    model = _model()
    cones = torch.randn(1, 20, 4)
    first = torch.zeros(1, 20, 2)
    second = first.clone()
    second[:, 10] = 1

    # When
    first_logits = model.forward_sequence(cones, observed_counts=first).logits
    second_logits = model.forward_sequence(cones, observed_counts=second).logits

    # Then
    assert torch.equal(first_logits[:, :11], second_logits[:, :11])
    assert not torch.equal(first_logits[:, 11:], second_logits[:, 11:])


def test_all_synaptic_clamps_remove_every_stimulus_to_logit_path() -> None:
    # Given
    model = _model()
    clamps = frozenset(
        {
            PathwayClamp.DIRECT_BC_SUSTAINED,
            PathwayClamp.DIRECT_BC_TRANSIENT,
            PathwayClamp.AMACRINE_LOCAL,
            PathwayClamp.AMACRINE_TRANSIENT,
        }
    )
    history = torch.zeros(1, 20, 2)

    # When
    driven = model.forward_sequence(
        torch.randn(1, 20, 4), observed_counts=history, clamps=clamps
    )
    silent = model.forward_sequence(
        torch.zeros(1, 20, 4), observed_counts=history, clamps=clamps
    )

    # Then
    assert torch.equal(driven.logits, silent.logits)


def test_base_rf_is_exact_sum_of_named_pathway_rfs() -> None:
    # Given
    model = _model()

    # When
    pathways = pathway_base_rf(model)

    # Then
    assert torch.allclose(
        base_rf(model),
        sum(pathways.values(), torch.zeros_like(next(iter(pathways.values())))),
        atol=1e-7,
        rtol=0,
    )


def test_effective_rf_autodiff_is_finite_with_oldest_to_current_lags() -> None:
    # Given
    model = _model()
    cones = torch.randn(1, 20, 4)
    history = torch.zeros(1, 20, 2)

    # When
    rf = effective_rf(model, cones, history)

    # Then
    assert rf.shape == (1, 2, 16, 4)
    assert torch.isfinite(rf).all()


def test_effective_pathway_rfs_sum_to_total_autodiff_rf() -> None:
    # Given
    model = _model()
    cones = torch.randn(1, 20, 4)
    history = torch.zeros(1, 20, 2)

    # When
    total = effective_rf(model, cones, history)
    pathways = effective_pathway_rf(model, cones, history)

    # Then
    summed = sum(pathways.values(), torch.zeros_like(next(iter(pathways.values()))))
    assert torch.allclose(summed, total, atol=2e-6, rtol=0)


def test_named_pathway_output_sensitivity_interface_is_finite() -> None:
    # Given
    model = _model()
    cones = torch.randn(1, 20, 4)
    history = torch.zeros(1, 20, 2)

    # When
    sensitivity = pathway_output_sensitivity(model, cones, history, time_index=19)

    # Then
    assert set(sensitivity) == {
        "BC-sustained",
        "BC-transient",
        "AC-local",
        "AC-transient",
    }
    assert all(value.shape == (1, 2) for value in sensitivity.values())
    assert all(torch.isfinite(value).all() for value in sensitivity.values())
