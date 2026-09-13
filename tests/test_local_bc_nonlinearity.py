import torch

from models.mechanistic_retina.local_bc_nonlinearity import LocalBCPlacement, pool_local_bc
from models.mechanistic_retina.local_bc_nonlinearity import LocalBCNonlinearRetina
from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina


def test_same_sum_local_composition_and_linear_limit_gradient() -> None:
    states = torch.tensor([[2.0, -1.0], [1.0, 0.0]])
    alpha = torch.tensor(0.5)
    assert torch.equal(states.sum(-1), torch.ones(2))
    post = pool_local_bc(states, alpha, LocalBCPlacement.POST_POOL)
    pre = pool_local_bc(states, alpha, LocalBCPlacement.PRE_POOL)
    assert post[0] == post[1]
    assert pre[0] != pre[1]
    theta = torch.tensor(0.0, requires_grad=True)
    linear = pool_local_bc(states, theta.exp(), LocalBCPlacement.PRE_POOL)
    assert torch.equal(linear, states.sum(-1))
    linear.sum().backward()
    assert theta.grad is not None
    assert theta.grad.item() == -1.0


def test_nested_linear_limit_preserves_original_float32_reduction() -> None:
    torch.manual_seed(7)
    positions = torch.cartesian_prod(torch.linspace(-0.3, 0.3, 7), torch.linspace(-0.3, 0.3, 7))
    config = MechanisticRetinaConfig(architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE)
    args = (config, positions, torch.zeros(1, 2), ("parasol",), ("ON",))
    baseline = MechanisticGraphTemporalRetina(*args)
    cones = torch.randn(1, 24, 49)
    history = torch.zeros(1, 24, 1)
    expected = baseline(cones, observed_counts=history)
    for placement in LocalBCPlacement:
        model = LocalBCNonlinearRetina(*args, placement=placement)
        model.load_state_dict({**baseline.state_dict(), "theta": torch.zeros(())}, strict=True)
        actual = model(cones, observed_counts=history)
        assert torch.equal(actual.logits, expected.logits)
        actual.logits.sum().backward()
        assert model.theta.grad is not None
        assert model.theta.grad.item() != 0.0
