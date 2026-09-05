#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0"]
# ///
# How to run: use the existing project environment from the repository root.
# PYTHONDONTWRITEBYTECODE=1 D:/anaconda/python.exe -B output/architecture_conformance_20260831/rf/basis_probe.py
# Archived reproduction script from the exact already-executed stdin probe.
from __future__ import annotations

import hashlib
import json

import torch

from evaluation.mechanistic_retina.rf_base import base_rf
from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig, PathwayClamp
from models.mechanistic_retina.model import build_mechanistic_retina

torch.manual_seed(73021)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
axis = torch.tensor([-0.16, -0.08, 0.0, 0.08, 0.16], dtype=torch.float64)
positions = torch.cartesian_prod(axis, axis)
cells = torch.tensor([[0.0, 0.0], [0.02, 0.0], [0.0, 0.02], [0.02, 0.02]], dtype=torch.float64)
config = MechanisticRetinaConfig(architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE)
model = build_mechanistic_retina(config, positions, cells, ('midget', 'midget', 'parasol', 'parasol'), ('ON', 'OFF', 'ON', 'OFF')).eval()
t = torch.arange(16, dtype=torch.float64).view(1, 16, 1)
c = torch.arange(25, dtype=torch.float64).view(1, 1, 25)
x = 0.27*torch.sin(0.37*t+0.19*c)+0.12*torch.cos(0.23*t-0.41*c)+0.02*t/15
history = (torch.arange(64).reshape(1, 16, 4)%11==0).to(torch.float64)
conditions = {
    'normal': frozenset(),
    'H1-off': frozenset({PathwayClamp.H1}),
    'direct-BC-off': frozenset({PathwayClamp.DIRECT_BC_SUSTAINED, PathwayClamp.DIRECT_BC_TRANSIENT}),
    'AC-off': frozenset({PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}),
}
rows = []
for name, clamps in conditions.items():
    output = model.forward_sequence(x, observed_counts=history, clamps=clamps)
    currents = torch.stack((output.bc_sustained_current, output.bc_transient_current, output.amacrine_local_current, output.amacrine_transient_current), dim=-1)
    basis = model.pathway_basis_features(x, clamps=clamps)
    shared_weights = model.bipolar.positive_weights().repeat(1, 2, 1, 1)
    contracted = (basis*shared_weights[None, None]).sum(dim=(-1,-2))
    zero = torch.zeros_like(x)
    zero_history = torch.zeros_like(history)
    direct_current_jacobian = torch.autograd.functional.jacobian(
        lambda values: model.forward_sequence(values, observed_counts=zero_history, clamps=clamps).total_current[0, -1], zero
    ).squeeze(1)
    helper_current_jacobian = base_rf(model, clamps=clamps)
    rows.append({
        'condition': name,
        'basis_shape': list(basis.shape),
        'current_shape': list(currents.shape),
        'basis_contracted_with_shared_BC_weights_current_max_abs_difference': float((contracted-currents).abs().max()),
        'basis_contracted_current_allclose': torch.allclose(contracted, currents, atol=1e-12, rtol=1e-10),
        'zero_input_base_rf_vs_production_total_current_jacobian_max_abs_difference': float((direct_current_jacobian-helper_current_jacobian).abs().max()),
        'zero_input_base_rf_allclose': torch.allclose(direct_current_jacobian, helper_current_jacobian, atol=1e-12, rtol=1e-10),
        'base_rf_sha256': hashlib.sha256(helper_current_jacobian.detach().numpy().tobytes()).hexdigest(),
    })
print(json.dumps({'fixture':'same as primary H probe seed73021 CPU float64 B1T16C25N4', 'scope':'basis/current estimand; NOT final-logit RF', 'rows':rows}))
