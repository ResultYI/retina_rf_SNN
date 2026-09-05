from __future__ import annotations

import torch

from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.model import build_mechanistic_retina
from models.mechanistic_retina.neural_operators import PathwayLocalOperator


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


def test_neutral_operator_enabled_and_disabled_step_zero_are_identical() -> None:
    # Given
    model = _model()
    cones = torch.randn(2, 20, 4)
    history = torch.zeros(2, 20, 2)

    # When
    disabled = model.forward_sequence(
        cones, observed_counts=history, operators_enabled=False
    )
    enabled = model.forward_sequence(cones, observed_counts=history, operators_enabled=True)

    # Then
    assert torch.equal(enabled.logits, disabled.logits)


def test_operator_is_causal_bounded_and_depthwise_isolated() -> None:
    # Given
    operator = PathwayLocalOperator(0.1)
    with torch.no_grad():
        operator.depthwise.weight[0, 0] = 1
    first = torch.zeros(1, 20, 1, 4, 2, 3)
    second = first.clone()
    second[:, 10, :, 0, 0, 0] = 1

    # When
    baseline = operator(first, enabled=True)
    changed = operator(second, enabled=True)
    difference = changed - baseline

    # Then
    assert torch.equal(changed[:, :10], baseline[:, :10])
    assert torch.count_nonzero(difference[..., 1:, :, :]) == 0
    assert changed.min() >= torch.exp(torch.tensor(-0.1))
    assert changed.max() <= torch.exp(torch.tensor(0.1))


def test_operator_forward_backward_is_finite_and_never_emits_direct_output() -> None:
    # Given
    model = _model()
    cones = torch.randn(2, 20, 4, requires_grad=True)
    history = torch.zeros(2, 20, 2)

    # When
    output = model.forward_sequence(
        cones, observed_counts=history, operators_enabled=True
    )
    output.logits.sum().backward()

    # Then
    assert torch.isfinite(output.logits).all()
    assert cones.grad is not None
    assert torch.isfinite(cones.grad).all()
    assert not hasattr(model.operator, "direct_current")
    assert not hasattr(model.operator, "direct_logit")

