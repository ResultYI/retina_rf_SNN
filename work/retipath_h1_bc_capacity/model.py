from __future__ import annotations

from enum import StrEnum

import torch
from torch import nn
from torch.nn import functional as F

from models.mechanistic_retina.contracts import PathwayClamp
from models.mechanistic_retina.retipath import RetiPath
from models.mechanistic_retina.retipath_spatial_ei import SpatialComponents, spatial_ei_parameters


class BasisForm(StrEnum):
    LINEAR = 'linear'
    MONOTONE_SOFTPLUS = 'monotone_softplus'


class PointwiseInputBasis(nn.Module):
    def __init__(self, form: BasisForm, slopes: torch.Tensor, offsets: torch.Tensor | None) -> None:
        super().__init__()
        self.form = BasisForm(form)
        if slopes.shape != (4,) or not bool(torch.isfinite(slopes).all() & (slopes > 0).all()):
            raise ValueError('Four finite positive slopes are required')
        self.register_buffer('slopes', slopes.detach().clone())
        if self.form is BasisForm.MONOTONE_SOFTPLUS:
            if offsets is None or offsets.shape != (4,) or not bool(torch.isfinite(offsets).all()):
                raise ValueError('Four finite softplus offsets are required')
            self.register_buffer('offsets', offsets.detach().clone())
        elif offsets is not None:
            raise ValueError('The linear control has no unused offset parameters')
        else:
            self.register_buffer('offsets', None)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        scaled = values[..., None] * self.slopes
        if self.form is BasisForm.LINEAR:
            return scaled
        offsets = self.offsets.expand_as(scaled).contiguous()
        return F.softplus(scaled + offsets) - F.softplus(offsets)


class H1BCCapacityRetiPath(RetiPath):
    def __init__(self, *args, input_basis: PointwiseInputBasis, h1_rms: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not h1_rms > 0 or not torch.isfinite(torch.tensor(h1_rms)):
            raise ValueError('H1 train RMS must be finite and positive')
        self.input_basis = input_basis
        self.register_buffer('h1_rms', torch.tensor(h1_rms))
        self.bc_basis_logits = nn.Parameter(torch.zeros(2, 4))

    @property
    def basis_weights(self) -> torch.Tensor:
        return torch.softmax(self.bc_basis_logits, dim=-1)

    def expanded_bc_features(self, modulated_cones: torch.Tensor) -> torch.Tensor:
        expanded = self.input_basis(modulated_cones / self.h1_rms)
        batch, time, cones, modes = expanded.shape
        bank_input = expanded.permute(0, 3, 1, 2).reshape(batch*modes, time, cones)
        features = self.feature_bank(bank_input, mixer=self.shared_subunits)
        features = features.reshape(batch, modes, *features.shape[1:]).permute(0, 2, 3, 4, 5, 6, 1)
        weights = self.basis_weights.repeat(2, 1)[None, None, None, :, None, None]
        return (features * weights).sum(-1)

    def spatial_components(self, cones: torch.Tensor,
                           clamps: frozenset[PathwayClamp] = frozenset()) -> SpatialComponents:
        gates = self.gates.values(clamps)
        h1 = self.h1(cones, amplitude=gates.h1)
        features = self.expanded_bc_features(h1.modulated_cones)
        weighted = features * self.bipolar.positive_weights().repeat(1, 2, 1, 1)[None, None]
        pooled = weighted.sum(dim=(-1, -2))
        slope = torch.where(pooled >= 0, torch.ones_like(pooled), self.alpha)
        components = (weighted.sum(-1) * slope[..., None]).transpose(-1, -2)
        direct, broad = components[..., :2], components[..., 2:]
        b, t, n, k, p = broad.shape
        ac = self.amacrine(broad.reshape(b, t, n*k, p),
                           local_gate=gates.ac_local, transient_gate=gates.ac_transient)
        ac_states = torch.stack((ac.local_state, ac.transient_state), -1).reshape(b, t, n, k, p)
        ac_currents = torch.stack((ac.local_current, ac.transient_current), -1).reshape(b, t, n, k, p)
        mask = direct.new_tensor((PathwayClamp.DIRECT_BC_SUSTAINED not in clamps,
                                  PathwayClamp.DIRECT_BC_TRANSIENT not in clamps))
        currents = torch.cat((direct * mask, ac_currents), -1)
        if self.cell_gains is not None:
            currents = currents * self.cell_gains.pathway_values[None, None, :, None, :]
        return SpatialComponents(h1, direct, broad, ac_states, currents)


def capacity_parameters(model: H1BCCapacityRetiPath) -> tuple[nn.Parameter, ...]:
    intended = (*spatial_ei_parameters(model), model.bc_basis_logits)
    return tuple(p for p in intended if p.requires_grad)
