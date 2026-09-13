"""Opt-in two-scale E/I pilot; the default delegates to unchanged RetiPath."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.mechanistic_retina.contracts import (
    MechanisticRetinaConfig, MechanisticRetinaOutput, PathwayClamp,
)
from models.mechanistic_retina.h1_pathway import H1Output
from models.mechanistic_retina.local_bc_nonlinearity import LocalBCNonlinearRetina, LocalBCPlacement
from models.mechanistic_retina.state import causal_lowpass, fixed_one_bin_history_state


class SpatialEIBackend(StrEnum):
    RETIPATH = "retipath"
    CURRENT = "spatial_current"
    CONDUCTANCE = "spatial_conductance"


@torch.jit.script
def exponential_sequence(target: torch.Tensor, fraction: torch.Tensor, initial: float) -> torch.Tensor:
    """Exact per-bin affine update, retaining every preceding input in autograd."""
    state = torch.full_like(target[:, 0], initial)
    values = torch.jit.annotate(list[torch.Tensor], [])
    for t in range(target.shape[1]):
        state = state + fraction[:, t] * (target[:, t] - state)
        values.append(state)
    return torch.stack(values, dim=1)


class SpatialEIIntegrator(nn.Module):
    def __init__(self, config: MechanisticRetinaConfig, backend: SpatialEIBackend,
                 rms_e: torch.Tensor, rms_i: torch.Tensor) -> None:
        super().__init__()
        if backend is SpatialEIBackend.RETIPATH:
            raise ValueError("spatial integrator requires an explicit new backend")
        for rms in (rms_e, rms_i):
            if rms.shape != (1, 2) or not bool(torch.isfinite(rms).all()) or not bool((rms > 0).all()):
                raise ValueError("initial train RMS must be finite positive [N=1,K=2]")
        self.backend = backend
        self.register_buffer("rms_e", rms_e.detach().clone())
        self.register_buffer("rms_i", rms_i.detach().clone())
        self.log_a_e = nn.Parameter(torch.zeros(()))
        self.log_a_i = nn.Parameter(torch.zeros(()))
        self.log_output_scale = nn.Parameter(torch.zeros(()))
        self.dt_ms = config.dt_ms
        self.tau_ref_ms = config.membrane_tau_ms
        self.g_l, self.e_l, self.e_e, self.e_i = 1.0, 0.0, 1.0, -1.0 / 3.0
        self.g_0 = 3.0
        self.v_0 = 2.0 / 9.0
        self.c_m = self.g_0 * self.tau_ref_ms
        self.b_0 = math.log(math.expm1(1.0))

    def conductances(self, u_e: torch.Tensor, u_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (F.softplus(self.b_0 + self.log_a_e.exp() * u_e / self.rms_e),
                F.softplus(self.b_0 + self.log_a_i.exp() * u_i / self.rms_i))

    def voltage(self, g_e: torch.Tensor, g_i: torch.Tensor) -> torch.Tensor:
        match self.backend:
            case SpatialEIBackend.CONDUCTANCE:
                total = self.g_l + g_e + g_i
                target = (self.g_l * self.e_l + g_e * self.e_e + g_i * self.e_i) / total
            case SpatialEIBackend.CURRENT:
                total = torch.full_like(g_e, self.g_0)
                target = self.v_0 + ((g_e - 1) * (self.e_e - self.v_0)
                                    + (g_i - 1) * (self.e_i - self.v_0)) / self.g_0
            case _:
                raise ValueError("unsupported spatial backend")
        fraction = -torch.expm1(-self.dt_ms * total / self.c_m)
        return exponential_sequence(target, fraction, self.v_0)

    def forward(self, u_e: torch.Tensor, u_i: torch.Tensor) -> torch.Tensor:
        g_e, g_i = self.conductances(u_e, u_i)
        voltage = self.voltage(g_e, g_i)
        return self.log_output_scale.exp() * (voltage.mean(-1) - self.v_0)


@dataclass(frozen=True)
class SpatialComponents:
    h1: H1Output
    direct: torch.Tensor
    broad: torch.Tensor
    ac_states: torch.Tensor
    currents: torch.Tensor

    @property
    def u_e(self) -> torch.Tensor:
        return self.currents[..., :2].sum(-1)

    @property
    def u_i(self) -> torch.Tensor:
        return -self.currents[..., 2:].sum(-1)


class SpatialEIRetiPath(LocalBCNonlinearRetina):
    """K denotes existing effective spatial scales, not anatomical dendrites."""

    def __init__(self, config: MechanisticRetinaConfig, cone_positions: torch.Tensor,
                 cell_positions: torch.Tensor, cell_types: tuple[str, ...],
                 polarities: tuple[str, ...], *,
                 backend: SpatialEIBackend = SpatialEIBackend.RETIPATH,
                 rms_e: torch.Tensor | None = None, rms_i: torch.Tensor | None = None) -> None:
        super().__init__(config, cone_positions, cell_positions, cell_types, polarities,
                         placement=LocalBCPlacement.POST_POOL)
        self.backend = SpatialEIBackend(backend)
        self.spatial_ei: SpatialEIIntegrator | None = None
        if self.backend is not SpatialEIBackend.RETIPATH:
            if rms_e is None or rms_i is None:
                raise ValueError("new backend requires frozen initial train RMS")
            self.spatial_ei = SpatialEIIntegrator(config, self.backend, rms_e, rms_i)

    def spatial_components(self, cones: torch.Tensor,
                           clamps: frozenset[PathwayClamp] = frozenset()) -> SpatialComponents:
        gates = self.gates.values(clamps)
        h1 = self.h1(cones, amplitude=gates.h1)
        features = self.feature_bank(h1.modulated_cones, mixer=self.shared_subunits)
        weighted = features * self.bipolar.positive_weights().repeat(1, 2, 1, 1)[None, None]
        pooled = weighted.sum(dim=(-1, -2))
        slope = torch.where(pooled >= 0, torch.ones_like(pooled), self.alpha)
        # The whole sustained/transient branch chooses one slope for both K modes.
        components = (weighted.sum(-1) * slope[..., None]).transpose(-1, -2)
        direct, broad = components[..., :2], components[..., 2:]
        b, t, n, k, p = broad.shape
        ac = self.amacrine(broad.reshape(b, t, n * k, p),
                           local_gate=gates.ac_local, transient_gate=gates.ac_transient)
        ac_states = torch.stack((ac.local_state, ac.transient_state), -1).reshape(b, t, n, k, p)
        ac_currents = torch.stack((ac.local_current, ac.transient_current), -1).reshape(b, t, n, k, p)
        direct_mask = direct.new_tensor((PathwayClamp.DIRECT_BC_SUSTAINED not in clamps,
                                         PathwayClamp.DIRECT_BC_TRANSIENT not in clamps))
        currents = torch.cat((direct * direct_mask, ac_currents), -1)
        if self.cell_gains is not None:
            currents = currents * self.cell_gains.pathway_values[None, None, :, None, :]
        return SpatialComponents(h1, direct, broad, ac_states, currents)

    def forward_sequence(self, cones: torch.Tensor, *, observed_counts: torch.Tensor,
                         clamps: frozenset[PathwayClamp] = frozenset(),
                         operators_enabled: bool = False) -> MechanisticRetinaOutput:
        if self.spatial_ei is None:
            return super().forward_sequence(cones, observed_counts=observed_counts,
                                            clamps=clamps, operators_enabled=operators_enabled)
        if operators_enabled or cones.ndim != 3 or observed_counts.shape != (*cones.shape[:2], 1):
            raise ValueError("spatial pilot requires [B,T,C], [B,T,1], disabled operators")
        clamps = frozenset(PathwayClamp(c) for c in clamps)
        parts = self.spatial_components(cones, clamps)
        membrane = self.spatial_ei(parts.u_e, parts.u_i)
        adaptation = causal_lowpass(membrane, self.rgc.adaptation_decay)
        history = fixed_one_bin_history_state(observed_counts, float(self.rgc.history_decay))
        gates = self.gates.values(clamps)
        adaptation_term = 0 if PathwayClamp.RGC_ADAPTATION in clamps else self.rgc.adaptation_gain * adaptation
        logits = (self.rgc.logit_slope * (membrane - self.rgc.threshold) - adaptation_term
                  - gates.history * self.rgc.history_gain * history + self.rgc.response_bias)
        direct, broad = parts.direct.sum(-2), parts.broad.sum(-2)
        bc_s, bc_t, ac_s, ac_t = parts.currents.sum(-2).unbind(-1)
        ac_states = parts.ac_states.sum(-2)
        on, off = self.bipolar.on_mask.view(1, 1, -1), self.bipolar.off_mask.view(1, 1, -1)
        return MechanisticRetinaOutput(
            parts.h1.graph_drive, parts.h1.state, parts.h1.surround,
            direct[..., 0] * on, bc_s * on, direct[..., 1] * on, bc_t * on,
            direct[..., 0] * off, bc_s * off, direct[..., 1] * off, bc_t * off,
            bc_s, bc_t, ac_states[..., 0], ac_s, ac_states[..., 1], ac_t,
            bc_s + bc_t + ac_s + ac_t, torch.zeros_like(membrane), membrane,
            adaptation, history, logits, torch.sigmoid(logits), direct, broad,
        )


def spatial_ei_parameters(model: SpatialEIRetiPath) -> tuple[nn.Parameter, ...]:
    from training.mechanistic_retina.optimizer import phase1_parameters
    extra = () if model.spatial_ei is None else tuple(model.spatial_ei.parameters())
    return (*phase1_parameters(model), model.theta, *extra)
