from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'work')]
from model import BasisForm, H1BCCapacityRetiPath, PointwiseInputBasis, capacity_parameters
from retipath_final_common import load_retipath
from retipath_phase2_common import SEEDS
from retipath_spatial_ei_pilot import model_args


def run() -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(20260914)
    registry = json.loads((ROOT / 'output/evaluations/retipath_final_model_evidence_20260913/model_registry.json').read_text())
    phase = json.loads((ROOT / 'output/experiments/retipath_spatial_ei_phase2_population/checkpoints/protocol.json').read_text())
    assert tuple(registry['seeds']) == SEEDS and phase['initialization'] == 'fresh'
    assert (phase['maximum_updates'], phase['patience_updates'], phase['evaluation_interval'],
            phase['lr'], phase['batch_size']) == (3000, 400, 25, .003, 4)
    representatives, verified = {}, 0
    for cell, row in registry['cells'].items():
        for seed, ref in row['RetiPath'].items():
            path = ROOT / ref['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == ref['sha256']
            cp = torch.load(path, weights_only=True, mmap=True)
            selected = torch.load(path.parent / 'inner.pt', weights_only=True, mmap=True)
            assert cp['backend'] == 'spatial_conductance' and cp['phase'] == 'refit'
            assert cp['trainable_parameters'] == 37 and cp['seed'] == int(seed) and cp['cell_id'] == cell
            assert selected['best_step'] == cp['step'] == ref['selected_updates'] and selected['step'] <= 3000
            assert selected['protocol_sha256'] == cp['protocol_sha256'] == phase['protocol_sha256']
            verified += 1
        representatives.setdefault(row['group'], path)
    identity_errors = {}
    for group, path in representatives.items():
        reference, cp = load_retipath(path)
        x = torch.randn(2, 150, 289) * .3
        history = torch.zeros(2, 150, 1)
        basis = PointwiseInputBasis(BasisForm.LINEAR, torch.ones(4), None)
        candidate = H1BCCapacityRetiPath(*model_args(cp), rms_e=torch.tensor(cp['rms']['e']),
                                       rms_i=torch.tensor(cp['rms']['i']), input_basis=basis, h1_rms=1.)
        missing = candidate.load_state_dict(cp['model'], strict=False)
        assert set(missing.missing_keys) == {'bc_basis_logits', 'input_basis.slopes', 'h1_rms'}
        assert not missing.unexpected_keys
        with torch.no_grad():
            a = reference(x, observed_counts=history).logits
            b = candidate(x, observed_counts=history).logits
            torch.testing.assert_close(a, b, rtol=2e-5, atol=3e-6)
            identity_errors[group] = float((a-b).abs().max())
    results = {}
    for form in BasisForm:
        basis = PointwiseInputBasis(form, torch.tensor([.5, 1., 2., 4.]),
                                    None if form is BasisForm.LINEAR else torch.tensor([-2., -.5, .5, 2.]))
        x = torch.linspace(-2, 2, 121).reshape(1, 121, 1).requires_grad_()
        values = basis(x)
        assert values.shape == (1, 121, 1, 4) and bool(torch.isfinite(values).all())
        assert bool((values[:, 1:]-values[:, :-1] >= 0).all())
        assert torch.equal(basis(torch.zeros_like(x)), torch.zeros_like(values))
        grad, = torch.autograd.grad(values.sum(), x)
        assert bool((grad > 0).all())
        candidate = H1BCCapacityRetiPath(*model_args(cp), rms_e=torch.tensor(cp['rms']['e']),
                                       rms_i=torch.tensor(cp['rms']['i']), input_basis=basis, h1_rms=1.)
        parameters = capacity_parameters(candidate)
        assert {id(p) for p in parameters} == {id(p) for p in candidate.parameters() if p.requires_grad}
        optimizer = torch.optim.Adam(parameters, lr=.003)
        assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in parameters}
        x = (torch.randn(2, 150, 289) * .3).requires_grad_()
        history = (torch.rand(2, 150, 1) < .3).float()
        logits = candidate(x, observed_counts=history).logits
        assert bool(torch.isfinite(logits).all())
        future_x, future_history = x.detach().clone(), history.clone()
        future_x[:, 81:] += 2.
        future_history[:, 80:] = 1-future_history[:, 80:]
        with torch.no_grad():
            changed = candidate(future_x, observed_counts=future_history).logits
        assert torch.equal(logits[:, :81], changed[:, :81])
        input_grad, = torch.autograd.grad(logits[:, 80].sum(), x, retain_graph=True)
        assert bool((input_grad[:, 81:] == 0).all()) and bool((input_grad[:, :80] != 0).any())
        torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 30:], history[:, 30:]).backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters)
        assert bool((candidate.basis_weights > 0).all())
        torch.testing.assert_close(candidate.basis_weights.sum(-1), torch.ones(2))
        buffer = io.BytesIO()
        torch.save({'form': form.value, 'model': candidate.state_dict()}, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=True)
        assert restored['form'] == form.value
        candidate.load_state_dict(restored['model'], strict=True)
        with torch.no_grad():
            assert torch.equal(candidate(x.detach(), observed_counts=history).logits, logits.detach())
        results[form.value] = {
            'trainable_parameters': sum(p.numel() for p in parameters),
            'zero_input_exact': True, 'future_input_and_current_history_no_leakage': True,
            'finite_gradients_all_optimizer_parameters': True, 'roundtrip_exact': True,
            'nonzero_gradient_parameter_names': [n for n, p in candidate.named_parameters()
                                                  if p.requires_grad and bool((p.grad != 0).any())],
        }
    return {'status': 'interface_checks_passed', 'frozen_current_checkpoints_verified': verified,
            'identity_linear_interface_max_error': identity_errors, 'synthetic_checks': results,
            'real_target_reads': 0, 'optimizer_updates': 0,
            'experimental_protocol_frozen': False,
            'fixed_basis_no_optimizer_parameters': True}


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
