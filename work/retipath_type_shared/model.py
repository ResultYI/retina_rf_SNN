from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'work'), str(ROOT / 'work/retipath_author_baseline')]
from models.mechanistic_retina.retipath import RetiPath
from models.mechanistic_retina.state import causal_lowpass
from retipath_spatial_ei_pilot import model_args

SHARED = ('theta', 'h1.raw_tau', 'h1.raw_delay', 'feature_bank.raw_tau',
          'feature_bank.raw_delay', 'bipolar.raw_weights', 'amacrine.raw_tau',
          'amacrine.raw_delay', 'spatial_ei.log_a_e', 'spatial_ei.log_a_i')
SPECIFIC = ('gates.raw_h1_amplitude', 'gates.ac_local', 'gates.ac_transient',
            'gates.history', 'cell_gains.log_bc', 'cell_gains.log_ac',
            'rgc.response_bias', 'spatial_ei.log_output_scale')


class TypeSharedRetiPath(nn.Module):
    def __init__(self, metadata: dict, seed: int, rms: dict | None = None) -> None:
        super().__init__()
        self.cells = tuple(sorted(metadata))
        self.groups: dict[str, list[str]] = defaultdict(list)
        self.cell_models = nn.ModuleDict()
        for cell in self.cells:
            cp = metadata[cell]
            torch.manual_seed(seed)
            scales = {'e': [[1., 1.]], 'i': [[1., 1.]]} if rms is None else rms[cell]
            model = RetiPath(*model_args(cp), rms_e=torch.tensor(scales['e']),
                             rms_i=torch.tensor(scales['i']))
            trainable = {n for n, p in model.named_parameters() if p.requires_grad}
            assert trainable == set(SHARED + SPECIFIC)
            group = cp['polarities'][0] + ' ' + cp['cell_types'][0]
            if self.groups[group]:
                first = self.cell_models[self.groups[group][0]]
                for name in SHARED:
                    parent, _, leaf = name.rpartition('.')
                    target = model.get_submodule(parent) if parent else model
                    target.register_parameter(leaf, first.get_parameter(name))
            self.cell_models[cell] = model
            self.groups[group].append(cell)

    def trainable(self) -> tuple[nn.Parameter, ...]:
        return tuple(p for p in self.parameters() if p.requires_grad)

    def project(self) -> None:
        for model in self.cell_models.values():
            model.project_mechanism_parameters()

    def stimulus_drives(self, cones: torch.Tensor) -> dict[str, torch.Tensor]:
        result = {}
        for cells in self.groups.values():
            models = [self.cell_models[c] for c in cells]
            first = models[0]
            h1 = first.h1(cones, amplitude=cones.new_ones(()))
            components = []
            gates = [m.gates.values(frozenset()) for m in models]
            for model, gate in zip(models, gates, strict=True):
                features = model.feature_bank(cones - gate.h1 * h1.surround,
                                              mixer=model.shared_subunits)
                weighted = features * model.bipolar.positive_weights().repeat(1, 2, 1, 1)[None, None]
                pooled = weighted.sum(dim=(-1, -2))
                slope = torch.where(pooled >= 0, torch.ones_like(pooled), model.alpha)
                components.append((weighted.sum(-1) * slope[..., None]).transpose(-1, -2))
            parts = torch.cat(components, dim=2)
            b, t, n, k, _ = parts.shape
            states = first.amacrine.presynaptic_states(parts[..., 2:].reshape(b, t, n*k, 2))
            states = states.reshape(b, t, n, k, 2)
            ac_gates = torch.stack([torch.cat((g.ac_local, g.ac_transient)) for g in gates])
            bc_gain = torch.cat([m.cell_gains.bc for m in models])
            ac_gain = torch.cat([m.cell_gains.ac for m in models])
            u_e = parts[..., :2].sum(-1) * bc_gain[None, None, :, None]
            u_i = (states * ac_gates[None, None, :, None]).sum(-1) * ac_gain[None, None, :, None]
            integrator = first.spatial_ei
            rms_e = torch.cat([m.spatial_ei.rms_e for m in models])
            rms_i = torch.cat([m.spatial_ei.rms_i for m in models])
            g_e = F.softplus(integrator.b_0 + integrator.log_a_e.exp() * u_e / rms_e)
            g_i = F.softplus(integrator.b_0 + integrator.log_a_i.exp() * u_i / rms_i)
            voltage = integrator.voltage(g_e, g_i)
            scale = torch.stack([m.spatial_ei.log_output_scale.exp() for m in models])
            membrane = (voltage.mean(-1) - integrator.v_0) * scale
            adaptation = causal_lowpass(membrane, first.rgc.adaptation_decay)
            drive = first.rgc.logit_slope * membrane - first.rgc.adaptation_gain * adaptation
            for i, cell in enumerate(cells):
                result[cell] = drive[..., i:i+1] - models[i].rgc.logit_slope * models[i].rgc.threshold
        return result

    def logits_from_drive(self, drive: torch.Tensor, history: torch.Tensor, cell: str) -> torch.Tensor:
        model = self.cell_models[cell]
        return drive - model.gates.history * model.rgc.history_gain * history + model.rgc.response_bias

    def inventory(self) -> list[dict]:
        rows = []
        for cell, model in self.cell_models.items():
            group = next(g for g, cells in self.groups.items() if cell in cells)
            for name, param in model.named_parameters():
                scope = 'type-shared' if name in SHARED else 'cell-specific' if param.requires_grad else 'fixed unused operator'
                rows.append({'cell_id': cell, 'type': group, 'parameter': name, 'shape': list(param.shape),
                             'elements': param.numel(), 'scope': scope, 'requires_grad': param.requires_grad})
        return rows
