#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0"]
# ///
# How to run: use the existing project environment from the repository root.
# PYTHONDONTWRITEBYTECODE=1 D:/anaconda/python.exe -B output/architecture_conformance_20260831/rf/primary_probe.py
# Archived reproduction script from the exact already-executed stdin probe.
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import torch

from evaluation.mechanistic_retina.karamanlis_v1_ac_runtime import EvaluationRequest, collect_responses
from evaluation.mechanistic_retina.pathway_decomposition import effective_pathway_rf
from evaluation.mechanistic_retina.rf_effective import effective_rf
from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig, PathwayClamp
from models.mechanistic_retina.model import build_mechanistic_retina

# Given: a fresh seeded Canonical V1 model and a complete 16-bin input window.
torch.manual_seed(73021)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
root = Path.cwd()
source_paths = sorted((root / 'models' / 'mechanistic_retina').glob('*.py'))
source_paths += [root / 'evaluation' / 'mechanistic_retina' / name for name in (
    'rf_effective.py', 'rf_base.py', 'pathway_decomposition.py', 'karamanlis_v1_ac_runtime.py')]
source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
axis = torch.tensor([-0.16, -0.08, 0.0, 0.08, 0.16], dtype=torch.float64)
positions = torch.cartesian_prod(axis, axis)
cells = torch.tensor([[0.0, 0.0], [0.02, 0.0], [0.0, 0.02], [0.02, 0.02]], dtype=torch.float64)
config = MechanisticRetinaConfig(architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE)
model = build_mechanistic_retina(config, positions, cells,
    ('midget', 'midget', 'parasol', 'parasol'), ('ON', 'OFF', 'ON', 'OFF')).eval()
time = torch.arange(16, dtype=torch.float64).view(1, 16, 1)
cone = torch.arange(25, dtype=torch.float64).view(1, 1, 25)
x = 0.27 * torch.sin(0.37 * time + 0.19 * cone) + 0.12 * torch.cos(0.23 * time - 0.41 * cone) + 0.02 * time / 15
history = (torch.arange(64).reshape(1, 16, 4) % 11 == 0).to(torch.float64)
conditions = {
    'normal': frozenset(),
    'H1-off': frozenset({PathwayClamp.H1}),
    'direct-BC-off': frozenset({PathwayClamp.DIRECT_BC_SUSTAINED, PathwayClamp.DIRECT_BC_TRANSIENT}),
    'AC-off': frozenset({PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}),
}
state_before = {name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest() for name, value in model.state_dict().items()}
result = {
    'scope': 'H RF/helper consistency only; no training, checkpoint, production edits, or filesystem writes',
    'python': sys.version, 'torch': torch.__version__, 'device': 'cpu', 'dtype': 'float64',
    'initialization': 'torch.manual_seed(73021); constructor default float32 parameters and geometry buffers then build_mechanistic_retina .to(float64)',
    'config': dataclasses.asdict(config), 'cone_positions': positions.tolist(), 'cell_positions': cells.tolist(),
    'fixture_shape': list(x.shape), 'history_shape': list(history.shape),
    'stimulus_formula': '0.27*sin(0.37*t+0.19*c)+0.12*cos(0.23*t-0.41*c)+0.02*t/15; t=0..15,c=0..24',
    'history_formula': '(arange(64).reshape(1,16,4)%11==0).float64',
    'input_sha256': hashlib.sha256(x.numpy().tobytes()).hexdigest(),
    'history_sha256': hashlib.sha256(history.numpy().tobytes()).hexdigest(),
    'tolerances': {'helper_atol': 1e-12, 'helper_rtol': 1e-10, 'central_fd_atol': 2e-8, 'central_fd_rtol': 2e-5},
    'source_hashes': source_hashes,
}
comparisons = []
finite_differences = []
normal_jacobian = None
for name, clamps in conditions.items():
    # When: independent torch.functional Jacobian and production RF helper differentiate the same final logit.
    jacobian = torch.autograd.functional.jacobian(
        lambda values: model.forward_sequence(values, observed_counts=history, clamps=clamps).logits[0, -1], x
    ).squeeze(1).unsqueeze(0)
    helper = effective_rf(model, x, history, clamps=clamps)
    output = model.forward_sequence(x, observed_counts=history, clamps=clamps)
    comparisons.append({
        'condition': name, 'shape': list(jacobian.shape), 'logits': output.logits[0, -1].detach().tolist(),
        'max_abs_difference': float((jacobian-helper).abs().max()),
        'bitwise_equal': torch.equal(jacobian, helper),
        'allclose': torch.allclose(jacobian, helper, atol=1e-12, rtol=1e-10),
        'jacobian_max_abs': float(jacobian.abs().max()),
        'production_jacobian_sha256': hashlib.sha256(jacobian.detach().numpy().tobytes()).hexdigest(),
        'helper_sha256': hashlib.sha256(helper.numpy().tobytes()).hexdigest(),
    })
    # Then: central differences independently verify two strongest and two fixed stimulus coordinates.
    coordinates = {(0, 12), (15, 24)}
    for cell_index in (0, 3):
        flat = int(jacobian[0, cell_index].abs().argmax())
        coordinates.add((flat // 25, flat % 25))
    for t, c in sorted(coordinates):
        for epsilon in (1e-4, 1e-5, 1e-6):
            delta = torch.zeros_like(x)
            delta[0, t, c] = epsilon
            with torch.no_grad():
                plus = model.forward_sequence(x+delta, observed_counts=history, clamps=clamps).logits[0, -1]
                minus = model.forward_sequence(x-delta, observed_counts=history, clamps=clamps).logits[0, -1]
            finite_difference = (plus-minus)/(2*epsilon)
            autograd = jacobian[0, :, t, c]
            finite_differences.append({
                'condition': name, 'coordinate': [0, t, c], 'epsilon': epsilon,
                'autograd': autograd.tolist(), 'central_fd': finite_difference.tolist(),
                'max_abs_difference': float((autograd-finite_difference).abs().max()),
                'allclose': torch.allclose(autograd, finite_difference, atol=2e-8, rtol=2e-5),
            })
    if not clamps:
        normal_jacobian = jacobian
pathways = effective_pathway_rf(model, x, history)
pathway_sum = torch.stack(tuple(pathways.values())).sum(0)
responses = collect_responses(EvaluationRequest(model, x, history, 1))
normal = model.forward_sequence(x, observed_counts=history)
ac_off = model.forward_sequence(x, observed_counts=history, clamps=conditions['AC-off'])
result['final_logit_jacobians'] = comparisons
result['central_finite_differences'] = finite_differences
result['effective_pathway_rf_sum'] = {
    'pathways': list(pathways), 'max_abs_difference': float((pathway_sum-normal_jacobian).abs().max()),
    'allclose': torch.allclose(pathway_sum, normal_jacobian, atol=1e-12, rtol=1e-10),
}
result['counterfactual_collect_responses'] = {
    'normal_logits_bitwise_equal': torch.equal(responses.normal.logits, normal.logits),
    'AC_off_logits_bitwise_equal': torch.equal(responses.clamped.logits, ac_off.logits),
    'normal_max_abs_difference': float((responses.normal.logits-normal.logits).abs().max()),
    'AC_off_max_abs_difference': float((responses.clamped.logits-ac_off.logits).abs().max()),
    'upstream_outputs_unchanged': responses.upstream_outputs_unchanged,
    'AC_local_exact_zero': bool(torch.count_nonzero(responses.clamped.ac_local)==0),
    'AC_transient_exact_zero': bool(torch.count_nonzero(responses.clamped.ac_transient)==0),
}
state_after = {name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest() for name, value in model.state_dict().items()}
result['model_state_unchanged'] = state_before == state_after
result['source_files_unchanged'] = all(hashlib.sha256(path.read_bytes()).hexdigest() == source_hashes[str(path.relative_to(root))] for path in source_paths)
result['initial_state_sha256_by_key'] = state_before
result['all_fd_pass'] = all(row['allclose'] for row in finite_differences)
result['max_fd_abs_difference'] = max(row['max_abs_difference'] for row in finite_differences)
print(json.dumps(result, ensure_ascii=False, allow_nan=False))
