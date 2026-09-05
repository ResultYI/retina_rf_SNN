from __future__ import annotations

import pytest
import torch

from evaluation.model_comparison.parameters import (
    parameter_inventory,
    parameter_snapshot,
)
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MechanisticRetinaConfig,
    PathwayClamp,
)
from models.mechanistic_retina.model import MechanisticModelError, build_mechanistic_retina
from training.mechanistic_retina.optimizer import build_phase1_optimizer, phase1_parameters
from training.mechanistic_retina.trainer import Phase1TrainingRequest, train_phase1


def _model(*, mechanism: bool = True):
    mode = (
        ArchitectureMode.MECHANISM_IDENTIFIABLE
        if mechanism
        else ArchitectureMode.LEGACY
    )
    return build_mechanistic_retina(
        MechanisticRetinaConfig(architecture_mode=mode),
        torch.tensor(
            [[0.00, 0.00], [0.05, 0.00], [0.10, 0.00], [0.15, 0.00]],
            dtype=torch.float32,
        ),
        torch.tensor([[0.04, 0.00], [0.08, 0.00]], dtype=torch.float32),
        ("midget", "midget"),
        ("ON", "ON"),
    )


def test_six_dimensional_kernel_mixing_matches_forward_autograd() -> None:
    model = _model()
    basis = model.feature_bank.basis_kernels()
    mixed = model.shared_subunits.mix_kernels(basis)
    connection = model.shared_subunits.connection_matrix()
    expected = torch.stack(
        tuple(
            sum(
                (
                    connection[target, source] * basis[source]
                    for source in range(basis.shape[0])
                ),
                torch.zeros_like(basis[0]),
            )
            for target in range(basis.shape[0])
        )
    )
    assert mixed.shape == basis.shape
    assert torch.allclose(mixed, expected, atol=1e-7, rtol=0)

    cones = torch.randn(1, model.config.lag_steps, basis.shape[-1]).requires_grad_(True)
    features = model.shared_subunits(model.feature_bank(cones))
    gradient = torch.autograd.grad(features[0, -1, 0, 0, 0, 0], cones)[0][0]
    assert torch.allclose(gradient, mixed[0, 0, 0, 0], atol=1e-7, rtol=0)

    pathway_cones = torch.zeros(
        1, model.config.lag_steps, basis.shape[-1], requires_grad=True
    )
    output = model.forward_sequence(
        pathway_cones,
        observed_counts=torch.zeros(1, model.config.lag_steps, 2),
    )
    currents = (
        output.bc_sustained_current,
        output.bc_transient_current,
        output.amacrine_local_current,
        output.amacrine_transient_current,
    )
    expected_pathways = []
    for current in currents:
        cells = []
        for cell in range(current.shape[-1]):
            cells.append(
                torch.autograd.grad(
                    current[0, -1, cell],
                    pathway_cones,
                    retain_graph=True,
                )[0][0]
            )
        expected_pathways.append(torch.stack(cells))
    for actual, expected_pathway in zip(
        model.pathway_base_rfs(), expected_pathways, strict=True
    ):
        assert torch.allclose(actual, expected_pathway, atol=1e-7, rtol=0)


@pytest.mark.parametrize("polarity", ("unknown", "INVALID", "off"))
def test_invalid_polarity_is_rejected(polarity: str) -> None:
    with pytest.raises(MechanisticModelError, match="unsupported polarities"):
        build_mechanistic_retina(
            MechanisticRetinaConfig(),
            torch.tensor([[0.0, 0.0], [0.1, 0.0]]),
            torch.tensor([[0.0, 0.0]]),
            ("midget",),
            (polarity,),
        )


@pytest.mark.parametrize("cell_type", ("unknown", "INVALID", "MIDGET"))
def test_invalid_cell_type_is_rejected(cell_type: str) -> None:
    with pytest.raises(MechanisticModelError, match="unsupported cell types"):
        build_mechanistic_retina(
            MechanisticRetinaConfig(),
            torch.tensor([[0.0, 0.0], [0.1, 0.0]]),
            torch.tensor([[0.0, 0.0]]),
            (cell_type,),
            ("ON",),
        )


def test_normal_training_gradients_clamps_and_parameter_categories() -> None:
    torch.manual_seed(7)
    model = _model()
    optimizer = build_phase1_optimizer(model, learning_rate=0.01)
    initial = parameter_snapshot(model)
    cones = torch.randn(3, 20, 4)
    history = torch.rand(3, 20, 2)

    optimizer.zero_grad(set_to_none=True)
    output = model.forward_sequence(cones, observed_counts=history)
    output.logits.square().mean().backward()

    intended = (
        model.bipolar.raw_weights,
        model.amacrine.raw_tau,
        model.amacrine.raw_delay,
        model.gates.raw_h1_amplitude,
        model.gates.ac_local,
        model.gates.ac_transient,
    )
    assert all(parameter.grad is not None for parameter in intended)
    assert all(torch.isfinite(parameter.grad).all() for parameter in intended if parameter.grad is not None)
    assert all(torch.count_nonzero(parameter.grad) > 0 for parameter in intended if parameter.grad is not None)
    assert all(any(parameter is listed for listed in phase1_parameters(model)) for parameter in intended)

    before_step = parameter_inventory(model, phase1_parameters(model))
    optimizer.step()
    after_step = parameter_inventory(
        model,
        phase1_parameters(model),
        initial_parameters=initial,
    )
    manual_optimizer_count = sum(parameter.numel() for parameter in phase1_parameters(model))
    manual_nonzero_gradient = sum(
        int(torch.count_nonzero(parameter.grad))
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    manual_updated = sum(
        int(torch.count_nonzero(parameter.detach() != initial[name]))
        for name, parameter in model.named_parameters()
    )
    assert before_step.total == sum(parameter.numel() for parameter in model.parameters())
    assert before_step.requires_grad == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert before_step.optimizer_listed == manual_optimizer_count
    assert before_step.nonzero_gradient == manual_nonzero_gradient
    assert before_step.actually_updated is None
    assert after_step.actually_updated == manual_updated
    assert 0 < after_step.actually_updated <= after_step.optimizer_listed

    clamps = frozenset(
        {
            PathwayClamp.H1,
            PathwayClamp.AMACRINE_LOCAL,
            PathwayClamp.AMACRINE_TRANSIENT,
        }
    )
    clamped = model.forward_sequence(cones, observed_counts=history, clamps=clamps)
    assert torch.count_nonzero(clamped.h1_surround_contribution) == 0
    assert torch.count_nonzero(clamped.amacrine_local_current) == 0
    assert torch.count_nonzero(clamped.amacrine_transient_current) == 0


def test_training_step_projects_learnable_gates_to_physiological_range() -> None:
    torch.manual_seed(11)
    model = _model()
    cones = torch.randn(2, 20, 4)
    history = torch.zeros(2, 20, 2)
    target = torch.rand(2, 20, 2)
    train_phase1(
        Phase1TrainingRequest(
            model,
            cones,
            target,
            torch.ones_like(target, dtype=torch.bool),
            cones,
            target,
            torch.ones_like(target, dtype=torch.bool),
            1,
            (0, 1),
            10.0,
            2,
            11,
        )
    )
    assert 0.0 <= float(model.gates.h1) <= 0.2
    assert 0.0 <= float(model.gates.history) <= 1.0
    effective = model.gates.values(frozenset())
    ac = torch.stack((effective.ac_local, effective.ac_transient))
    assert bool((ac >= 0).all())
    torch.testing.assert_close(ac.sum(), torch.ones(()))
