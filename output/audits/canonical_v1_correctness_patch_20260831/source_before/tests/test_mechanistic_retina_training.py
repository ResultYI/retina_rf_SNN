from __future__ import annotations

import inspect
from pathlib import Path

import torch

from evaluation.mechanistic_retina.rf_base import CandidateTeacherUsage, load_candidate0
from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig
from models.mechanistic_retina.model import build_mechanistic_retina
from training.mechanistic_retina.losses import expected_bernoulli_nll
from training.mechanistic_retina.optimizer import build_phase1_optimizer, phase1_parameters
from training.mechanistic_retina.stages import Phase1Schedule, build_seed_data
from training.mechanistic_retina.trainer import (
    Phase1TrainingRequest,
    fit_signed_control,
    load_checkpoint,
    matched_control_rf,
    save_checkpoint,
    signed_path_design,
    train_phase1,
)


def _model():
    return build_mechanistic_retina(
        MechanisticRetinaConfig(),
        torch.tensor(
            [[0.00, 0.00], [0.05, 0.00], [0.10, 0.00], [0.15, 0.00]],
            dtype=torch.float32,
        ),
        torch.tensor([[0.04, 0.00], [0.11, 0.00]], dtype=torch.float32),
        ("midget", "parasol"),
        ("ON", "OFF"),
    )


def test_expected_likelihood_contains_no_rf_target_argument() -> None:
    # Given
    expected = ("logits", "teacher_probability", "valid_mask")

    # When
    parameters = tuple(inspect.signature(expected_bernoulli_nll).parameters)

    # Then
    assert parameters == expected


def test_expected_likelihood_matches_softplus_minus_probability_logit() -> None:
    # Given
    logits = torch.tensor([[[0.2, -0.4]]])
    probability = torch.tensor([[[0.3, 0.7]]])
    mask = torch.ones_like(probability, dtype=torch.bool)

    # When
    loss = expected_bernoulli_nll(logits, probability, mask)

    # Then
    expected = (torch.nn.functional.softplus(logits) - probability * logits).mean()
    assert torch.equal(loss, expected)


def test_phase1_optimizer_excludes_disabled_neural_operator() -> None:
    # Given
    model = _model()

    # When
    parameters = phase1_parameters(model)

    # Then
    assert all(parameter is not model.operator.depthwise.weight for parameter in parameters)
    assert all(parameter is not model.operator.depthwise.bias for parameter in parameters)


def test_phase1_optimizer_covers_learnable_shared_connections() -> None:
    # Given
    model = build_mechanistic_retina(
        MechanisticRetinaConfig(
            architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE
        ),
        torch.tensor(
            [[0.00, 0.00], [0.05, 0.00], [0.10, 0.00], [0.15, 0.00]],
            dtype=torch.float32,
        ),
        torch.tensor([[0.04, 0.00], [0.06, 0.00]], dtype=torch.float32),
        ("midget", "midget"),
        ("ON", "ON"),
    )

    # When
    parameters = phase1_parameters(model)

    # Then
    assert any(
        parameter is model.shared_subunits.raw_connections
        for parameter in parameters
    )


def test_fixed_step_schedule_never_selects_a_validation_step() -> None:
    # Given
    expected_final_step = 400

    # When
    schedule = Phase1Schedule()

    # Then
    assert schedule.smoke_steps == (0, 10, 25, 50)
    assert schedule.final_steps == (0, 50, 100, 200, 400)
    assert schedule.final_step == expected_final_step


def test_checkpoint_roundtrip_preserves_numerical_output(tmp_path: Path) -> None:
    # Given
    model = _model()
    optimizer = build_phase1_optimizer(model, learning_rate=0.03)
    cones = torch.randn(1, 20, 4)
    history = torch.zeros(1, 20, 2)
    expected = model.forward_sequence(cones, observed_counts=history).logits.detach()
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, optimizer, seed=19, step=50)
    with torch.no_grad():
        model.rgc.response_bias.add_(1)

    # When
    state = load_checkpoint(path, model, optimizer)
    actual = model.forward_sequence(cones, observed_counts=history).logits.detach()

    # Then
    assert state.seed == 19
    assert state.step == 50
    assert torch.equal(actual, expected)


def test_signed_matched_control_recovers_noise_free_in_span_logits() -> None:
    # Given
    generator = torch.Generator().manual_seed(9)
    design = torch.randn(3, 20, 2, 24, generator=generator, dtype=torch.float64)
    coefficients = torch.randn(2, 24, generator=generator, dtype=torch.float64)
    bias = torch.tensor([-1.2, -2.1], dtype=torch.float64)
    target_logits = torch.einsum("btnf,nf->btn", design, coefficients) + bias
    probability = torch.sigmoid(target_logits)
    mask = torch.ones_like(probability, dtype=torch.bool)

    # When
    fit = fit_signed_control(design, probability, mask)

    # Then
    assert fit.converged
    assert torch.sqrt((fit.logits - target_logits).square().mean()) <= 1e-8


def test_seed_workspace_preserves_fixed_phase1_budget_and_identity() -> None:
    # Given
    candidate = load_candidate0(
        Path(
            ".omo/evidence/hierarchical-endpoint-and-v4-decision/teacher-preflight-results.json"
        ),
        usage=CandidateTeacherUsage.DEVELOPMENT_REFERENCE,
        reference_candidate_index=0,
    )

    # When
    data = build_seed_data(19, candidate)

    # Then
    assert data.train_cones.shape == (112, 320, 29)
    assert data.train_probability.shape == (112, 8, 320, 16)
    assert data.validation_cones.shape == (12, 320, 29)
    assert data.cell_ids == tuple(value.cell_id for value in candidate.metadata)


def test_matched_control_uses_exactly_the_models_24_signed_basis_columns() -> None:
    # Given
    model = _model()
    coefficients = torch.cat(
        (model.bipolar.positive_weights(), model.bipolar.positive_weights()), dim=1
    ).reshape(2, 24)

    # When
    design = signed_path_design(model, torch.randn(1, 20, 4))
    recovered = matched_control_rf(model, coefficients)

    # Then
    assert design.shape == (1, 20, 2, 24)
    pathways = model.pathway_base_rfs()
    assert torch.allclose(recovered, sum(pathways, torch.zeros_like(pathways[0])))


def test_phase1_training_uses_fixed_steps_and_reduces_expected_ce() -> None:
    # Given
    teacher = _model()
    cones = torch.randn(4, 20, 4)
    history = torch.zeros(4, 20, 2)
    probability = teacher.forward_sequence(
        cones, observed_counts=history
    ).spike_probability.detach()
    model = _model()
    with torch.no_grad():
        model.rgc.response_bias.add_(0.5)
    request = Phase1TrainingRequest(
        model,
        cones,
        probability,
        torch.ones_like(probability, dtype=torch.bool),
        cones,
        probability,
        torch.ones_like(probability, dtype=torch.bool),
        10,
        (0, 5, 10),
        0.03,
        4,
        19,
    )

    # When
    result = train_phase1(request)

    # Then
    assert tuple(point.step for point in result.checkpoints) == (0, 5, 10)
    assert result.checkpoints[-1].validation_ce < result.checkpoints[0].validation_ce
    assert result.gradients_finite
