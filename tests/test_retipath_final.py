from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'work'))

from retipath_final_common import PHASE2, load_retipath, application_imports
from retipath_phase2_analyze import model_from_checkpoint

application_imports()
from application_io import CLAMPS
from run_application import check_clamp
from application_io import CONDITIONS


def test_official_entry_replays_frozen_model_and_pathway_drive_clamps():
    torch.set_num_threads(1)
    path = PHASE2 / 'checkpoints/67_14/2026091301/C/refit.pt'
    current, cp = load_retipath(path)
    old, _ = model_from_checkpoint(path)
    old.eval().requires_grad_(False)
    generator = torch.Generator().manual_seed(73)
    x = torch.randn(2, 150, 289, generator=generator) * 0.2
    history = (torch.rand(2, 150, 1, generator=generator) < 0.1).float()
    with torch.inference_mode():
        normal = current.forward_sequence(x, observed_counts=history)
        for name, clamp in zip(CONDITIONS, CLAMPS, strict=True):
            expected = old.forward_sequence(x, observed_counts=history, clamps=clamp)
            actual = current.forward_sequence(x, observed_counts=history, clamps=clamp)
            assert all(torch.equal(a, b) for a, b in zip(actual.tensors(), expected.tensors(), strict=True))
            check_clamp(normal, actual, name)
        direct_off = current.spatial_components(x, CLAMPS[2])
        ac_off = current.spatial_components(x, CLAMPS[3])
        assert torch.count_nonzero(direct_off.u_e) == 0
        assert torch.count_nonzero(ac_off.u_i) == 0
        g_e, _ = current.spatial_ei.conductances(direct_off.u_e, direct_off.u_i)
        _, g_i = current.spatial_ei.conductances(ac_off.u_e, ac_off.u_i)
        torch.testing.assert_close(g_e, torch.ones_like(g_e), rtol=0, atol=1e-7)
        torch.testing.assert_close(g_i, torch.ones_like(g_i), rtol=0, atol=1e-7)
    assert all(torch.equal(v, cp['model'][k]) for k, v in current.state_dict().items())
    assert all(p.grad is None for p in current.parameters())


def test_official_loader_rejects_historical_backend():
    with pytest.raises(ValueError, match='conductance refit'):
        load_retipath(PHASE2 / 'checkpoints/67_14/2026091301/A/refit.pt')
