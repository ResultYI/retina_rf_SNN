from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from experiments.retipath_multiscale_v0.circuit import area_integral_weights, fixed_geometry
from experiments.retipath_multiscale_v0.contracts import DT_MS, validate_pixel_bounds
from models.mechanistic_retina.retipath_spatial_ei import exponential_sequence
from models.mechanistic_retina.state import (
    causal_fractional_delay, causal_lowpass, fixed_one_bin_history_state,
)


class BCOutput(StrEnum):
    LEGACY_PRELU = "LegacyPReLU"
    BASELINE_SOFTPLUS = "BaselineSoftplus"


class Intervention(StrEnum):
    NORMAL = "NORMAL"
    BLOCK_H1_FEEDBACK = "BLOCK_H1_FEEDBACK"
    BLOCK_DIRECT_BC_DRIVE = "BLOCK_DIRECT_BC_DRIVE"
    BLOCK_AC_POSTSYNAPTIC_DRIVE = "BLOCK_AC_POSTSYNAPTIC_DRIVE"


@dataclass(frozen=True)
class Stimulus:
    values: Tensor  # [B,T,P], piecewise constant contrast at 150 Hz
    pixel_bounds: Tensor  # [P,xy,lower_upper], degrees
    pixel_area: Tensor
    time_ms: Tensor  # consecutive bin midpoints
    input_valid: Tensor

    def validate(self) -> None:
        x = self.values
        if x.ndim != 3 or min(x.shape) == 0 or not x.is_floating_point():
            raise ValueError("values must be nonempty floating [B,T,P]")
        validate_pixel_bounds(self.pixel_bounds)
        if self.pixel_bounds.shape[0] != x.shape[-1] or self.pixel_area.shape != (x.shape[-1],):
            raise ValueError("pixel count/area mismatch")
        for value in (self.pixel_bounds, self.pixel_area, self.time_ms):
            if value.dtype != x.dtype or value.device != x.device:
                raise ValueError("coordinates and values must share dtype/device")
        area = (self.pixel_bounds[..., 1] - self.pixel_bounds[..., 0]).prod(-1)
        if not torch.allclose(area, self.pixel_area, rtol=1e-6, atol=0):
            raise ValueError("pixel_area differs from physical bounds")
        if (self.input_valid.shape != x.shape or self.input_valid.dtype != torch.bool
                or self.input_valid.device != x.device or not bool(self.input_valid.all())):
            raise ValueError("full-graph forward requires complete known input")
        time = (torch.arange(x.shape[1], dtype=x.dtype, device=x.device) + 0.5) * DT_MS
        if self.time_ms.shape != time.shape or not torch.allclose(self.time_ms, time):
            raise ValueError("time_ms must use the fixed 150 Hz midpoint clock")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("nonfinite stimulus")


@dataclass(frozen=True)
class Trace:
    inputs: dict[str, Tensor]
    states: dict[str, Tensor]
    outputs: dict[str, Tensor]
    initial_state: dict[str, Tensor]
    intervention: Intervention
    bc_output: BCOutput

    @property
    def bc_quantity_kind(self) -> str:
        return ("signed_effective_output" if self.bc_output is BCOutput.LEGACY_PRELU
                else "nonnegative_release_like_proxy")

    def observation(self, port: str) -> Tensor:
        """Identity port only; no claim of calibrated voltage/current/sensor units."""
        if port in self.states:
            return self.states[port]
        return self.outputs[port]


class PooledField(nn.Module):
    """Family centers plus zero-sum contrasts; no redundant free unit parameters."""

    def __init__(self, family: Tensor, lo: float, hi: float, initial: float | tuple[float, ...],
                 *, group: str, sd: float | None, center_sd: float | None = None,
                 dtype: torch.dtype) -> None:
        super().__init__()
        self.group, self.lo, self.hi = group, lo, hi
        self.sd, self.center_sd = sd, center_sd
        self.register_buffer("family", family)
        n_family = int(family.max()) + 1
        physical = torch.as_tensor(initial, dtype=dtype).expand(n_family)
        fraction = (physical - lo) / (hi - lo)
        raw = torch.logit(fraction)
        self.center = nn.Parameter(raw.clone())
        self.register_buffer("initial_center", raw.clone())
        columns = []
        if sd is not None:
            for f in range(n_family):
                ids = torch.where(family == f)[0]
                for k in range(1, ids.numel()):
                    column = torch.zeros(family.numel(), dtype=dtype)
                    column[ids[:k]] = 1 / math.sqrt(k * (k + 1))
                    column[ids[k]] = -k / math.sqrt(k * (k + 1))
                    columns.append(column)
        basis = torch.stack(columns, 1) if columns else torch.empty(family.numel(), 0, dtype=dtype)
        self.register_buffer("basis", basis)
        self.register_parameter("contrast", nn.Parameter(torch.zeros(basis.shape[1], dtype=dtype))
                                if basis.shape[1] else None)

    def deviations(self) -> Tensor:
        return (self.basis @ self.contrast if self.contrast is not None
                else torch.zeros_like(self.family, dtype=self.center.dtype))

    def forward(self) -> Tensor:
        raw = self.center[self.family] + self.deviations()
        return self.lo + (self.hi - self.lo) * raw.sigmoid()

    def penalty(self) -> Tensor:
        value = self.center.new_zeros(())
        if self.contrast is not None:
            value = value + 0.5 * (self.deviations() / self.sd).square().sum()
        if self.center_sd is not None:
            value = value + 0.5 * ((self.center - self.initial_center) / self.center_sd).square().sum()
        return value


def _spatial(receivers: Tensor, sources: Tensor, sigma: float, radius: float) -> Tensor:
    distance = torch.cdist(receivers, sources)
    weights = torch.exp(-0.5 * (distance / sigma).square()) * (distance <= radius)
    if bool((weights.sum(-1) == 0).any()):
        raise ValueError("empty fixed route support")
    return weights / weights.sum(-1, keepdim=True)


def _lowpass(value: Tensor, tau: Tensor | float) -> Tensor:
    tau = torch.as_tensor(tau, dtype=value.dtype, device=value.device)
    return causal_lowpass(value, torch.exp(-DT_MS / tau))


class PopulationRetipath(nn.Module):
    def __init__(self, *, bc_output: BCOutput | str = BCOutput.LEGACY_PRELU,
                 dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self._bc_output = BCOutput(bc_output)
        g = fixed_geometry(dtype=dtype)
        for name, value in {
            "input_xy": g.input_xy, "h_xy": g.h_xy,
            "bc_xy": g.bc_xy.repeat(2, 1), "ac_xy": g.ac_xy.repeat(4, 1),
            "rgc_xy": g.rgc_xy.repeat(2, 1), "G": g.G, "F_H": g.G.T.clone(),
            "bc_family": torch.arange(2).repeat_interleave(25),
            "ac_family": torch.arange(4).repeat_interleave(9),
            "bc_polarity": torch.tensor([1., -1.], dtype=dtype).repeat_interleave(25),
            "ac_polarity": torch.tensor([1., -1., 1., -1.], dtype=dtype).repeat_interleave(9),
            "rgc_polarity": torch.tensor([1., -1.], dtype=dtype),
        }.items():
            self.register_buffer(name, value)
        direct = torch.zeros(2, 2, 50, dtype=dtype)
        direct[0, :, :25], direct[1, :, 25:] = g.D, g.D
        ba = torch.zeros(36, 50, dtype=dtype)
        for f in range(4):
            sigma, radius = (0.15, 0.23) if f < 2 else (0.30, 0.65)
            ba[f * 9:(f + 1) * 9, (f % 2) * 25:(f % 2 + 1) * 25] = _spatial(
                g.ac_xy, g.bc_xy, sigma, radius)
        self.register_buffer("pi_BR", direct)
        self.register_buffer("pi_BA", ba)
        self.register_buffer("pi_AR", g.I.repeat(2, 1, 4))
        self.register_buffer("same_polarity_AR", self.rgc_polarity[:, None] == self.ac_polarity)
        self.register_buffer("gamma_BA", torch.ones(36, dtype=dtype))

        h_family = torch.zeros(25, dtype=torch.long)
        definitions = {
            "tau_H": (h_family, 10., 100., 50., "state", .3, None),
            "delay_H": (h_family, 0., 20., 5., "state", .3, None),
            "a_H": (h_family, 0., .8, .3, "coupling", .1, None),
            "tau_f_B": (self.bc_family, 8., 40., 22., "state", .6, None),
            "tau_gap_B": (self.bc_family, 20., 140., 78., "state", .6, None),
            "delay_B": (self.bc_family, 0., 20., 2., "state", .6, None),
            "kappa_B": (self.bc_family, 0., 1., .5, "state", .6, None),
            "tau_A": (self.ac_family, 20., 200., (50., 50., 140., 140.), "state", None, None),
            "delay_A": (self.ac_family, 0., 20., 5., "state", None, None),
            "b_A": (self.ac_family, -4., 4., 0., "output", None, None),
            "gamma_BR": (torch.arange(4), .2, 8., 1.5, "coupling", None, .3),
            "gamma_AR": (torch.arange(16), 0., 8., .375, "coupling", None, .3),
            "bias": (torch.arange(2), -4., -.5, -2.4, "output", None, .3),
        }
        if self.bc_output is BCOutput.LEGACY_PRELU:
            definitions["alpha_B"] = (self.bc_family, .05, 1., .5, "output", .6, None)
        else:
            definitions["b_B"] = (self.bc_family, -4., 4., 0., "output", .6, None)
        self.fields = nn.ModuleDict({
            name: PooledField(family.clone(), lo, hi, initial, group=group,
                              sd=sd, center_sd=center_sd, dtype=dtype)
            for name, (family, lo, hi, initial, group, sd, center_sd) in definitions.items()
        })

    @property
    def bc_output(self) -> BCOutput:
        return self._bc_output

    def physical_parameters(self) -> dict[str, Tensor]:
        p = {name: field() for name, field in self.fields.items()}
        p["tau_s_B"] = p["tau_f_B"] + p["tau_gap_B"]
        p["gamma_BR"] = p["gamma_BR"].reshape(2, 2)
        p["gamma_AR"] = p["gamma_AR"].reshape(2, 2, 4)
        p["gamma_BA"] = self.gamma_BA
        return p

    def parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        return {group: tuple(p for field in self.fields.values() if field.group == group
                             for p in field.parameters()) for group in ("state", "output", "coupling")}

    def parameter_registry(self) -> dict[str, dict[str, object]]:
        h_ports = ("h_H", "H1_feedback", "s_B", "delta_r_B", "a_A", "o_A", "d_E", "d_I", "gE", "gI", "logit")
        bc_ports = h_ports[2:]
        output_name = "alpha_B" if self.bc_output is BCOutput.LEGACY_PRELU else "b_B"
        routes = {"tau_H": h_ports, "delay_H": h_ports, "a_H": h_ports[1:],
                  **{name: bc_ports for name in ("tau_f_B", "tau_gap_B", "delay_B", "kappa_B")},
                  output_name: bc_ports[1:],
                  "tau_A": ("a_A", "o_A", "d_I", "gI", "logit"),
                  "delay_A": ("a_A", "o_A", "d_I", "gI", "logit"),
                  "b_A": ("o_A", "d_I", "gI", "logit"),
                  "gamma_BR": ("d_E", "gE", "logit"),
                  "gamma_AR": ("d_I", "gI", "logit"), "bias": ("logit",)}
        registry = {name: {
            "class": field.group, "owner": name, "family_ids": field.family.tolist(),
            "sharing": "partial_pooling" if field.contrast is not None else "family_shared",
            "bounds": (field.lo, field.hi), "transform": "bounded_sigmoid_raw",
            "deviation_sd": field.sd, "center_prior_sd": field.center_sd,
            "fixed": False, "permitted_data_ports": routes[name] + ("probability",),
            "trainable_count": sum(p.numel() for p in field.parameters()),
        } for name, field in self.fields.items()}
        registry["gamma_BA"] = {"class": "coupling", "sharing": "fixed_one", "fixed": True,
                                "permitted_data_ports": (), "trainable_count": 0}
        return registry

    def hierarchy_penalty(self) -> Tensor:
        return torch.stack([field.penalty() for field in self.fields.values()]).sum()

    def regularized_objective(self, data_loss: Tensor) -> Tensor:
        if data_loss.ndim != 0 or not bool(torch.isfinite(data_loss)):
            raise ValueError("data_loss must be a finite scalar")
        return data_loss + self.hierarchy_penalty()

    def pathway_support(self) -> dict[str, Tensor]:
        """Structural ancestor apertures, independent of learned amplitudes and interventions."""
        def compose(left: Tensor, right: Tensor) -> Tensor:
            return (left.to(self.G.dtype) @ right.to(self.G.dtype)) > 0
        h = self.G != 0
        feedback = compose(self.F_H != 0, h)
        bc = (feedback | torch.eye(25, dtype=torch.bool, device=h.device)).repeat(2, 1)
        ac = compose(self.pi_BA != 0, bc)
        direct = compose(self.pi_BR != 0, bc)
        inhibitory = compose(self.pi_AR != 0, ac)
        return {"h_H": h, "H1_feedback": feedback, "s_B": bc, "delta_r_B": bc,
                "a_A": ac, "o_A": ac, "d_E": direct, "d_I": inhibitory,
                "RGC": (direct | inhibitory).any(1)}

    def forward(self, stimulus: Stimulus, *, observed_events: Tensor,
                intervention: Intervention | str = Intervention.NORMAL,
                reset: str = "baseline") -> Trace:
        stimulus.validate()
        intervention = Intervention(intervention)
        if reset != "baseline":
            raise ValueError("Stage A supports independent baseline-reset sequences only")
        x = stimulus.values
        if x.dtype != self.G.dtype or x.device != self.G.device:
            raise ValueError("model and stimulus must share dtype/device")
        if (observed_events.shape != (*x.shape[:2], 2) or observed_events.dtype != x.dtype
                or observed_events.device != x.device
                or not bool(((observed_events == 0) | (observed_events == 1)).all())):
            raise ValueError("observed_events must be matching binary [B,T,2]")
        p = self.physical_parameters()
        q = x @ area_integral_weights(stimulus.pixel_bounds, self.input_xy).T
        u_h = q @ self.G.T
        h = _lowpass(causal_fractional_delay(u_h, p["delay_H"], dt_ms=DT_MS), p["tau_H"])
        feedback = (h * p["a_H"]) @ self.F_H.T
        if intervention is Intervention.BLOCK_H1_FEEDBACK:
            feedback = torch.zeros_like(feedback)
        q_mod = q - feedback
        u_b = q_mod.repeat(1, 1, 2) * self.bc_polarity
        v_b = _lowpass(causal_fractional_delay(u_b, p["delay_B"], dt_ms=DT_MS), p["tau_f_B"])
        w_b = _lowpass(v_b, p["tau_s_B"])
        s_b = v_b - p["kappa_B"] * w_b
        if self.bc_output is BCOutput.LEGACY_PRELU:
            r_b = torch.where(s_b >= 0, s_b, p["alpha_B"] * s_b)
            r_b0 = torch.zeros_like(p["alpha_B"])
        else:
            r_b = F.softplus(p["b_B"] + s_b)
            r_b0 = F.softplus(p["b_B"])
        delta_b = r_b - r_b0
        c_e = delta_b[:, :, None, None, :] * (p["gamma_BR"][..., None] * self.pi_BR)
        if intervention is Intervention.BLOCK_DIRECT_BC_DRIVE:
            c_e = torch.zeros_like(c_e)
        d_e = c_e.sum(-1)
        u_a = (delta_b @ self.pi_BA.T) * p["gamma_BA"]
        a = _lowpass(causal_fractional_delay(u_a, p["delay_A"], dt_ms=DT_MS), p["tau_A"])
        o_a, o_a0 = F.softplus(p["b_A"] + a), F.softplus(p["b_A"])
        delta_a = o_a - o_a0
        c_i = delta_a[:, :, None, None, :] * (p["gamma_AR"][..., self.ac_family] * self.pi_AR)
        if intervention is Intervention.BLOCK_AC_POSTSYNAPTIC_DRIVE:
            c_i = torch.zeros_like(c_i)
        d_i = c_i.sum(-1)
        b0 = math.log(math.expm1(1.0))
        g_e, g_i = F.softplus(b0 + d_e), F.softplus(b0 + d_i)
        total = 1 + g_e + g_i
        voltage = exponential_sequence((g_e - g_i / 3) / total,
                                       -torch.expm1(-DT_MS * total / 60), 2 / 9)
        membrane = voltage.mean(-1) - 2 / 9
        adaptation = _lowpass(membrane, 120.)
        history = fixed_one_bin_history_state(observed_events, math.exp(-DT_MS / 40))
        logit = 8 * membrane - .5 * adaptation - history + p["bias"]
        states = {"h_H": h, "v_B": v_b, "w_B": w_b, "s_B": s_b,
                  "a_A": a, "V": voltage, "adaptation": adaptation, "history": history}
        initial = {name: torch.full_like(value[:, 0], 2 / 9 if name == "V" else 0.)
                   for name, value in states.items()}
        return Trace(
            inputs={"q": q, "u_H": u_h, "q_modulated": q_mod, "u_B": u_b, "u_A": u_a},
            states=states,
            outputs={"H1_feedback": feedback, "r_B": r_b, "r_B0": r_b0, "delta_r_B": delta_b,
                     "o_A": o_a, "o_A0": o_a0, "delta_o_A": delta_a,
                     "c_E": c_e, "c_I": c_i, "d_E": d_e, "d_I": d_i,
                     "gE": g_e, "gI": g_i, "logit": logit, "probability": logit.sigmoid()},
            initial_state=initial, intervention=intervention, bc_output=self.bc_output,
        )
