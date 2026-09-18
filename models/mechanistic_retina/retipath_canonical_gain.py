from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from models.mechanistic_retina.bipolar_subunits import BipolarSubunits
from models.mechanistic_retina.contracts import MechanisticRetinaConfig, PathwayClamp
from models.mechanistic_retina.pathway_gates import GateValues, PathwayGates
from models.mechanistic_retina.retipath import RetiPath
from models.mechanistic_retina.retipath_spatial_ei import SpatialComponents, SpatialEIBackend, SpatialEIIntegrator


CANONICAL_GAIN_SCHEMA = "retipath_canonical_gain_v1"
REMOVED_GAIN_KEYS = frozenset({
    "bipolar.raw_weights", "gates.ac_local", "gates.ac_transient", "cell_gains.log_bc", "cell_gains.log_ac",
    "spatial_ei.log_a_e", "spatial_ei.log_a_i",
})


class ConditionalBipolarSubunits(BipolarSubunits):
    def __init__(self, previous: BipolarSubunits) -> None:
        nn.Module.__init__(self)
        self.conditional_log_ratios = nn.Parameter(previous.raw_weights.new_zeros((1, 2, 5)))
        for name in ("group_index", "on_mask", "off_mask"):
            self.register_buffer(name, getattr(previous, name).clone())

    def positive_weights(self) -> torch.Tensor:
        ratios = self.conditional_log_ratios
        logits = torch.cat((ratios, torch.zeros_like(ratios[..., :1])), -1)
        return F.softmax(logits, -1).reshape(1, 2, 2, 3)[self.group_index]


class CanonicalPathwayGates(PathwayGates):
    def __init__(self, previous: PathwayGates) -> None:
        nn.Module.__init__(self)
        self.raw_h1_amplitude = previous.raw_h1_amplitude
        self.history = previous.history
        for name in ("group_index", "h1_amplitude_bounds"):
            self.register_buffer(name, getattr(previous, name).clone())

    def values(self, clamps: frozenset[PathwayClamp]) -> GateValues:
        unit_gate = torch.ones_like(self.group_index, dtype=self.history.dtype)
        return GateValues(
            self._value(self.h1, PathwayClamp.H1, clamps),
            self._value(unit_gate, PathwayClamp.AMACRINE_LOCAL, clamps),
            self._value(unit_gate, PathwayClamp.AMACRINE_TRANSIENT, clamps),
            self._value(self.history, PathwayClamp.RGC_HISTORY, clamps),
        )


class CanonicalGainCoordinates(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_G_E = nn.Parameter(torch.zeros(1))
        self.log_G_I = nn.Parameter(torch.zeros(1))
        self.delta_E = nn.Parameter(torch.zeros(1))
        self.delta_I = nn.Parameter(torch.zeros(1))
        self.register_buffer("legacy_a_E", torch.ones(1))
        self.register_buffer("legacy_a_I", torch.ones(1))

    @staticmethod
    def weights(delta: torch.Tensor) -> torch.Tensor:
        return F.softmax(torch.stack((delta, torch.zeros_like(delta)), -1), -1)

    @property
    def w_E(self) -> torch.Tensor:
        return self.weights(self.delta_E)

    @property
    def w_I(self) -> torch.Tensor:
        return self.weights(self.delta_I)

    @property
    def G_E(self) -> torch.Tensor:
        return self.log_G_E.exp()

    @property
    def G_I(self) -> torch.Tensor:
        return self.log_G_I.exp()


class CanonicalSpatialEIIntegrator(SpatialEIIntegrator):
    def __init__(self, config: MechanisticRetinaConfig, rms_e: torch.Tensor, rms_i: torch.Tensor) -> None:
        super().__init__(config, SpatialEIBackend.CONDUCTANCE, rms_e, rms_i)
        del self.log_a_e
        del self.log_a_i

    def conductances(self, effective_e: torch.Tensor, effective_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return F.softplus(self.b_0 + effective_e / self.rms_e), F.softplus(self.b_0 + effective_i / self.rms_i)


@dataclass(frozen=True)
class CanonicalSpatialComponents(SpatialComponents):
    relative_e: torch.Tensor
    relative_i: torch.Tensor
    effective_e: torch.Tensor
    effective_i: torch.Tensor

    @property
    def u_e(self) -> torch.Tensor:
        return self.effective_e

    @property
    def u_i(self) -> torch.Tensor:
        return self.effective_i


class CanonicalGainRetiPath(RetiPath):
    """Same function family in gain-gauge-free coordinates; old checkpoints require explicit migration."""

    def __init__(self, config: MechanisticRetinaConfig, cone_positions: torch.Tensor,
                 cell_positions: torch.Tensor, cell_types: tuple[str, ...], polarities: tuple[str, ...],
                 *, rms_e: torch.Tensor, rms_i: torch.Tensor) -> None:
        if not config.cell_specific_gains or config.cell_specific_pathway_mixture:
            raise ValueError("canonical migration requires the formal aggregate-gain configuration")
        super().__init__(config, cone_positions, cell_positions, cell_types, polarities, rms_e=rms_e, rms_i=rms_i)
        self.bipolar = ConditionalBipolarSubunits(self.bipolar)
        self.gates = CanonicalPathwayGates(self.gates)
        self.cell_gains = None
        self.canonical_gains = CanonicalGainCoordinates()
        self.spatial_ei = CanonicalSpatialEIIntegrator(config, rms_e, rms_i)

    def spatial_components(self, cones: torch.Tensor,
                           clamps: frozenset[PathwayClamp] = frozenset()) -> CanonicalSpatialComponents:
        gates = self.gates.values(clamps)
        h1 = self.h1(cones, amplitude=gates.h1)
        features = self.feature_bank(h1.modulated_cones, mixer=self.shared_subunits)
        weighted = features * self.bipolar.positive_weights().repeat(1, 2, 1, 1)[None, None]
        pooled = weighted.sum(dim=(-1, -2))
        slope = torch.where(pooled >= 0, torch.ones_like(pooled), self.alpha)
        components = (weighted.sum(-1) * slope[..., None]).transpose(-1, -2)
        direct, broad = components[..., :2], components[..., 2:]
        b, t, n, k, p = broad.shape
        ac = self.amacrine(broad.reshape(b, t, n * k, p),
                           local_gate=gates.ac_local, transient_gate=gates.ac_transient)
        ac_states = torch.stack((ac.local_state, ac.transient_state), -1).reshape(b, t, n, k, p)
        ac_currents = torch.stack((ac.local_current, ac.transient_current), -1).reshape(b, t, n, k, p)
        direct_mask = direct.new_tensor((PathwayClamp.DIRECT_BC_SUSTAINED not in clamps,
                                         PathwayClamp.DIRECT_BC_TRANSIENT not in clamps))
        gain = self.canonical_gains
        e_composition = direct * direct_mask * gain.w_E[None, None, :, None, :]
        i_composition = ac_currents * gain.w_I[None, None, :, None, :]
        relative_e, relative_i = e_composition.sum(-1), -i_composition.sum(-1)
        effective_e = gain.G_E[None, None, :, None] * relative_e
        effective_i = gain.G_I[None, None, :, None] * relative_i
        e = gain.G_E[None, None, :, None, None] * e_composition
        i = gain.G_I[None, None, :, None, None] * i_composition
        return CanonicalSpatialComponents(h1, direct, broad, ac_states, torch.cat((e, i), -1),
                                          relative_e, relative_i, effective_e, effective_i)


def canonicalize_gain_state(old: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Factor the actual 37-parameter checkpoint, including BC branch mass and AC gate composition."""
    if not REMOVED_GAIN_KEYS <= old.keys() or old["bipolar.raw_weights"].shape != (1, 2, 2, 3):
        raise ValueError("expected an unmigrated formal N=1,K=2 aggregate-gain state")
    if "canonical_gains.log_G_E" in old:
        raise ValueError("checkpoint is already canonical")
    relevant = [old[key] for key in REMOVED_GAIN_KEYS]
    if any(not bool(torch.isfinite(value).all()) for value in relevant):
        raise ValueError("nonfinite gain coordinates cannot be migrated")
    raw = old["bipolar.raw_weights"].double().flatten(-2)
    log_mass = raw.logsumexp(-1) - raw.flatten(1).logsumexp(-1)[:, None]
    gate_logits = torch.stack((old["gates.ac_local"], old["gates.ac_transient"]), -1).double()
    log_q = F.log_softmax(gate_logits, -1)
    log_i_mass = log_mass + log_q
    log_z = log_i_mass.logsumexp(-1)
    result = {key: value.clone() for key, value in old.items() if key not in REMOVED_GAIN_KEYS}
    dtype = old["bipolar.raw_weights"].dtype
    added = {
        "bipolar.conditional_log_ratios": raw[..., :5] - raw[..., 5:],
        "canonical_gains.log_G_E": old["cell_gains.log_bc"].double() + old["spatial_ei.log_a_e"].double(),
        "canonical_gains.log_G_I": old["cell_gains.log_ac"].double() + old["spatial_ei.log_a_i"].double() + log_z,
        "canonical_gains.delta_E": log_mass[:, 0] - log_mass[:, 1],
        "canonical_gains.delta_I": log_i_mass[:, 0] - log_i_mass[:, 1],
        "canonical_gains.legacy_a_E": old["spatial_ei.log_a_e"].double().exp().reshape(1),
        "canonical_gains.legacy_a_I": old["spatial_ei.log_a_i"].double().exp().reshape(1),
    }
    for key, value in added.items():
        result[key] = value.to(dtype=dtype)
        if not bool(torch.isfinite(result[key]).all()):
            raise ValueError(f"coordinate out of target dtype range: {key}")
    if any(not bool((result[key] > 0).all()) for key in ("canonical_gains.legacy_a_E", "canonical_gains.legacy_a_I")):
        raise ValueError("legacy alias scales must be strictly positive")
    return result
