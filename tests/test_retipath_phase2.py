from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'work'))
from retipath_phase2_common import SelectionStop, jacobian_modes, mode_shares, paired_bootstrap
from models.mechanistic_retina.contracts import MechanisticRetinaConfig, ArchitectureMode
from models.mechanistic_retina.retipath_spatial_ei import SpatialEIBackend, SpatialEIRetiPath


def test_patience_is_400_updates_not_400_evaluations() -> None:
    status = SelectionStop(1., 0, 1., 0)
    for step in range(25, 400, 25):
        status = status.observe(1., step)
        assert not status.stopped(step)
    assert status.observe(1., 400).stopped(400)
    improved = status.observe(.9, 400)
    assert improved.best_step == 400 and not improved.stopped(400)
    tiny = improved.observe(.9 - 1e-8, 425)
    assert tiny.best_step == 425 and tiny.last_improvement_step == 400


def test_full_jacobian_mode_decomposition_for_all_backends() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(13)
    positions = torch.cartesian_prod(torch.linspace(-.3,.3,7), torch.linspace(-.3,.3,7))
    config = MechanisticRetinaConfig(architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE,
                                    dt_ms=1000/150, cell_specific_gains=True)
    x = torch.randn(1, 55, 49, dtype=torch.float64)
    history = torch.zeros(1, 55, 1, dtype=torch.float64)
    for backend in SpatialEIBackend:
        model = SpatialEIRetiPath(config, positions, torch.zeros(1,2), ('parasol',), ('ON',),
                                 backend=backend, rms_e=torch.ones(1,2), rms_i=torch.ones(1,2)).double()
        with torch.no_grad():
            model.theta.fill_(-.4)
        j, modes, quality = jacobian_modes(model, x, history)
        np.testing.assert_allclose(modes.sum(0), j, rtol=1e-8, atol=1e-10)
        shares = mode_shares(j, modes)
        np.testing.assert_allclose(shares['q'].sum(), 1, atol=1e-13)
        np.testing.assert_allclose(shares['signed_projection'].sum(), 1, atol=1e-8)
        assert quality['sum_relative_l2_error'] < 1e-8
        assert np.square(j[:-16]).sum() > 0


def test_mode_energy_shares_do_not_discard_interference() -> None:
    modes = np.array([[[2.,0.]], [[-1.,0.]]])
    shares = mode_shares(modes.sum(0), modes)
    np.testing.assert_allclose(shares['q'], [.8,.2])
    np.testing.assert_allclose(shares['signed_projection'], [2.,-1.])
    assert shares['cross_over_total_energy'] == -4


def test_equal_cell_bootstrap_uses_shared_indices() -> None:
    indices = np.array([[0,1,2],[0,0,0],[2,2,2],[1,1,1]])
    result = paired_bootstrap(np.array([-1.,-2.,3.]), indices)
    assert result['mean'] == 0 and result['wins'] == 2 and result['losses'] == 1
    assert result['ci_low'] < 0 < result['ci_high']
