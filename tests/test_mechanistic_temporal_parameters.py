from __future__ import annotations

import pytest
import torch

from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MechanisticRetinaConfig,
)
from models.mechanistic_retina.model import build_mechanistic_retina
from models.mechanistic_retina.state import causal_fractional_delay
from training.mechanistic_retina.optimizer import (
    build_phase1_optimizer,
    phase1_parameters,
)


def _model():
    return build_mechanistic_retina(
        MechanisticRetinaConfig(
            architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE
        ),
        torch.tensor(
            [[0.00, 0.00], [0.05, 0.00], [0.10, 0.00], [0.15, 0.00]],
            dtype=torch.float32,
        ),
        torch.tensor([[0.04, 0.00], [0.08, 0.00]], dtype=torch.float32),
        ("midget", "midget"),
        ("ON", "ON"),
    )


def test_temporal_taus_are_bounded_learnable_and_optimizer_listed() -> None:
    model = _model()
    expected_path_taus = torch.tensor(
        (
            (35.0, 60.0, 100.0),
            (10.0, 20.0, 35.0),
        )
    )
    assert torch.allclose(model.h1.tau_ms, torch.tensor(50.0), atol=1e-5, rtol=0)
    assert torch.allclose(
        model.feature_bank.tau_ms, expected_path_taus, atol=1e-5, rtol=0
    )
    assert torch.allclose(
        model.amacrine.tau_ms, torch.tensor((35.0, 18.0)), atol=1e-5, rtol=0
    )

    elapsed = torch.arange(model.config.lag_steps - 1, -1, -1) * model.config.dt_ms
    scaled = elapsed.view(1, 1, -1) / expected_path_taus[..., None]
    undelayed_basis = scaled * torch.exp(1 - scaled)
    undelayed_basis = undelayed_basis / undelayed_basis.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    path_delays = torch.tensor((10.0, 5.0))
    semantic_lag_basis = (
        undelayed_basis.flip(-1).permute(2, 0, 1).reshape(1, model.config.lag_steps, -1)
    )
    channel_delays = path_delays[:, None].expand(-1, 3).reshape(-1)
    expected_basis = (
        causal_fractional_delay(
            semantic_lag_basis,
            channel_delays,
            dt_ms=model.config.dt_ms,
        )
        .reshape(model.config.lag_steps, 2, 3)
        .permute(1, 2, 0)
        .flip(-1)
    )
    assert torch.allclose(
        model.feature_bank.temporal_basis, expected_basis, atol=1e-6, rtol=0
    )

    tau_parameters = (
        model.h1.raw_tau,
        model.feature_bank.raw_tau,
        model.amacrine.raw_tau,
    )
    listed = phase1_parameters(model)
    assert all(
        any(parameter is item for item in listed) for parameter in tau_parameters
    )

    optimizer = build_phase1_optimizer(model, learning_rate=0.01)
    optimizer.zero_grad(set_to_none=True)
    cones = torch.randn(3, 20, 4)
    history = torch.rand(3, 20, 2)
    model.forward_sequence(
        cones, observed_counts=history
    ).logits.square().mean().backward()
    assert all(parameter.grad is not None for parameter in tau_parameters)
    assert all(
        torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad) > 0
        for parameter in tau_parameters
        if parameter.grad is not None
    )


def test_fractional_delays_are_bounded_learnable_and_optimizer_listed() -> None:
    model = _model()
    assert torch.allclose(model.h1.delay_ms, torch.tensor(5.0), atol=1e-5, rtol=0)
    assert torch.allclose(
        model.feature_bank.delay_ms,
        torch.tensor((10.0, 5.0)),
        atol=1e-5,
        rtol=0,
    )
    assert torch.all(model.feature_bank.delay_ms[:1] > model.feature_bank.delay_ms[1:2])
    torch.testing.assert_close(model.amacrine.delay_ms, torch.tensor((15.0, 7.5)))

    delay_parameters = (
        model.h1.raw_delay,
        model.feature_bank.raw_delay,
        model.amacrine.raw_delay,
    )
    listed = phase1_parameters(model)
    assert all(
        any(parameter is item for item in listed) for parameter in delay_parameters
    )

    optimizer = build_phase1_optimizer(model, learning_rate=0.01)
    optimizer.zero_grad(set_to_none=True)
    cones = torch.randn(3, 20, 4)
    history = torch.rand(3, 20, 2)
    model.forward_sequence(
        cones, observed_counts=history
    ).logits.square().mean().backward()
    assert all(parameter.grad is not None for parameter in delay_parameters)
    assert all(
        torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad) > 0
        for parameter in delay_parameters
        if parameter.grad is not None
    )


def test_fractional_delay_linearly_interpolates_adjacent_timesteps() -> None:
    values = torch.tensor((0.0, 10.0, 20.0, 30.0)).view(1, 4, 1)
    delay_ms = torch.tensor(2.5, requires_grad=True)

    delayed = causal_fractional_delay(values, delay_ms, dt_ms=5.0)

    assert torch.allclose(
        delayed,
        torch.tensor((0.0, 5.0, 15.0, 25.0)).view(1, 4, 1),
        atol=1e-7,
        rtol=0,
    )
    delayed[:, -1].sum().backward()
    assert delay_ms.grad is not None
    assert torch.isfinite(delay_ms.grad)
    assert torch.count_nonzero(delay_ms.grad) == 1


def test_temporal_tau_pairs_remain_physiologically_ordered() -> None:
    model = _model()
    with torch.no_grad():
        model.feature_bank.raw_tau[0].fill_(-100.0)
        model.feature_bank.raw_tau[1].fill_(100.0)
        model.amacrine.raw_tau[0].fill_(-100.0)
        model.amacrine.raw_tau[1].fill_(100.0)

    assert torch.all(model.feature_bank.tau_ms[0] > model.feature_bank.tau_ms[1])
    assert model.amacrine.tau_ms[0] > model.amacrine.tau_ms[1]


def test_explicit_delay_pairs_remain_bounded_and_ordered() -> None:
    model = _model()
    with torch.no_grad():
        model.h1.raw_delay.fill_(100.0)
        model.feature_bank.raw_delay[0].fill_(-100.0)
        model.feature_bank.raw_delay[1].fill_(100.0)
        model.amacrine.raw_delay[0].fill_(-100.0)
        model.amacrine.raw_delay[1].fill_(100.0)

    assert model.h1.delay_bounds_ms[0] <= model.h1.delay_ms
    assert model.h1.delay_ms <= model.h1.delay_bounds_ms[1]
    assert model.feature_bank.delay_ms[0] > model.feature_bank.delay_ms[1]
    assert model.amacrine.delay_ms[0] > model.amacrine.delay_ms[1]
    assert torch.all(
        model.feature_bank.delay_ms >= model.feature_bank.delay_bounds_ms[:, 0]
    )
    assert torch.all(
        model.feature_bank.delay_ms <= model.feature_bank.delay_bounds_ms[:, 1]
    )
    assert torch.all(model.amacrine.delay_ms >= model.amacrine.delay_bounds_ms[:, 0])
    assert torch.all(model.amacrine.delay_ms <= model.amacrine.delay_bounds_ms[:, 1])


def test_model_dtype_is_anchored_to_float32_geometry() -> None:
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = _model()
    finally:
        torch.set_default_dtype(original)

    cones = torch.randn(2, 20, 4, dtype=torch.float32)
    history = torch.zeros(2, 20, 2, dtype=torch.float32)
    output = model.forward_sequence(cones, observed_counts=history)
    assert output.logits.dtype is torch.float32
    assert torch.isfinite(output.logits).all()
    assert all(parameter.dtype is torch.float32 for parameter in model.parameters())
    assert all(
        buffer.dtype is torch.float32
        for buffer in model.buffers()
        if torch.is_floating_point(buffer)
    )


def test_invalid_temporal_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="sustained/local tau"):
        MechanisticRetinaConfig(
            bc_basis_tau_ms=((30.0, 40.0, 50.0), (35.0, 45.0, 55.0))
        )


def test_invalid_delay_bounds_and_order_are_rejected() -> None:
    with pytest.raises(ValueError, match="H1 delay temporal value"):
        MechanisticRetinaConfig(h1_delay_ms=25.0)
    with pytest.raises(ValueError, match="BC delay sustained/local delay"):
        MechanisticRetinaConfig(bc_delay_ms=(5.0, 10.0))
