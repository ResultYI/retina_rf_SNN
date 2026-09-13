import math

import torch

from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.retipath_spatial_ei import SpatialEIBackend, SpatialEIIntegrator


def integrator(backend: SpatialEIBackend) -> SpatialEIIntegrator:
    return SpatialEIIntegrator(MechanisticRetinaConfig(), backend,
                               torch.ones(1, 2), torch.ones(1, 2)).double()


def test_constant_conductance_exact_solution() -> None:
    model = integrator(SpatialEIBackend.CONDUCTANCE)
    e = torch.tensor([0.2, 3.0], dtype=torch.float64).expand(1, 37, 1, 2)
    i = torch.tensor([2.1, 0.7], dtype=torch.float64).expand_as(e)
    time = torch.arange(1, 38, dtype=torch.float64).view(1, -1, 1, 1) * model.dt_ms
    total = 1 + e + i
    target = (e - i / 3) / total
    expected = target + (model.v_0 - target) * torch.exp(-total * time / model.c_m)
    torch.testing.assert_close(model.voltage(e, i), expected, rtol=1e-13, atol=1e-14)


def test_constant_total_conductance_matches_current_control() -> None:
    torch.manual_seed(91)
    e = torch.rand(3, 43, 1, 2, dtype=torch.float64) * 2
    i = 2 - e
    b, c = integrator(SpatialEIBackend.CURRENT), integrator(SpatialEIBackend.CONDUCTANCE)
    torch.testing.assert_close(b.voltage(e, i), c.voltage(e, i), rtol=1e-13, atol=1e-14)


def test_zero_drive_rest_and_shared_positive_mapping() -> None:
    b, c = integrator(SpatialEIBackend.CURRENT), integrator(SpatialEIBackend.CONDUCTANCE)
    u = torch.linspace(-100, 100, 80, dtype=torch.float64).view(1, 40, 1, 2)
    for model in (b, c):
        e, i = model.conductances(u, -u)
        assert bool((e >= 0).all() & (i >= 0).all())
        assert bool(torch.isfinite(model(u, -u)).all())
        zero = torch.zeros_like(u)
        e0, i0 = model.conductances(zero, zero)
        torch.testing.assert_close(e0, torch.ones_like(e0), rtol=0, atol=1e-15)
        torch.testing.assert_close(i0, torch.ones_like(i0), rtol=0, atol=1e-15)
        torch.testing.assert_close(model(zero, zero), zero[..., 0], rtol=0, atol=1e-15)
        assert math.isclose(model.c_m / model.g_0, model.tau_ref_ms)
    for x, y in zip(b.conductances(u, -u), c.conductances(u, -u), strict=True):
        assert torch.equal(x, y)


def test_integrator_retains_past_gradient_without_future_leakage() -> None:
    model = integrator(SpatialEIBackend.CONDUCTANCE)
    e = torch.ones(1, 40, 1, 2, dtype=torch.float64, requires_grad=True)
    i = torch.ones_like(e)
    grad, = torch.autograd.grad(model.voltage(e, i)[0, 30].sum(), e)
    assert bool((grad[:, :31] > 0).all())
    assert torch.count_nonzero(grad[:, 31:]) == 0


def test_voltage_gradient_matches_finite_difference() -> None:
    model = integrator(SpatialEIBackend.CONDUCTANCE)
    e = torch.full((1, 8, 1, 2), 0.7, dtype=torch.float64, requires_grad=True)
    i = torch.full_like(e, 0.4)
    assert torch.autograd.gradcheck(model.voltage, (e, i.requires_grad_()), atol=1e-7, rtol=1e-5)
