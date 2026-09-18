from __future__ import annotations

from enum import StrEnum
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.mechanistic_retina.state import causal_lowpass, fixed_one_bin_history_state
from models.mechanistic_retina.retipath_spatial_ei import exponential_sequence
from .contracts import CircuitGeometry, DT_MS, Intervention, InterventionSpec, StateBundle, StimulusBatch


# name: (lower, upper, student initial center, semantic parameter group)
PARAMETERS = {
    "tau_h": (10.0, 100.0, 50.0, "H"), "a_h": (0.0, 0.8, 0.30, "F"),
    "tau_f": (8.0, 40.0, 22.0, "B_s"), "tau_s": (40.0, 160.0, 100.0, "B_s"),
    "alpha": (0.05, 1.0, 0.50, "B_o"),
    "tau_a1": (20.0, 80.0, 50.0, "A"), "tau_a2": (80.0, 200.0, 140.0, "A"),
    "g_e": (0.2, 8.0, 1.5, "E"), "delta_e": (-4.0, 4.0, 0.0, "E"),
    "g_i": (0.2, 8.0, 1.5, "I"), "delta_i": (-4.0, 4.0, 0.0, "I"),
    "bias": (-4.0, -0.5, -2.4, "R"),
}


class TrainingCondition(StrEnum):
    RGC_ONLY = "rgc_only"
    JOINT = "joint"
    PROGRESSIVE = "progressive"


def active_parameter_names(condition: TrainingCondition | str, macro_step: int) -> tuple[str, ...]:
    condition = TrainingCondition(condition)
    if isinstance(macro_step, bool) or not isinstance(macro_step, int) or not 1 <= macro_step <= 180:
        raise ValueError("the designed macro-step range is 1..180")
    if condition is TrainingCondition.PROGRESSIVE:
        if macro_step <= 30:
            return ("tau_h",)
        if macro_step <= 60:
            return ("a_h", "tau_f", "tau_s", "alpha")
    return tuple(PARAMETERS)


def _positions(axis: Tensor) -> Tensor:
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), -1).reshape(-1, 2)


def _weights(receivers: Tensor, sources: Tensor, sigma: Tensor, radius: float) -> Tensor:
    distance = torch.cdist(receivers, sources)
    value = torch.exp(-distance.square() / (2 * sigma.reshape(-1, 1).square()))
    value = value * (distance <= radius)
    return value / value.sum(-1, keepdim=True)


def fixed_geometry(*, dtype: torch.dtype = torch.float32) -> CircuitGeometry:
    bc = _positions(torch.tensor((-0.30, -0.15, 0.0, 0.15, 0.30), dtype=dtype))
    ac = _positions(torch.tensor((-0.30, 0.0, 0.30), dtype=dtype))
    rgc = torch.zeros((1, 2), dtype=dtype)
    return CircuitGeometry(
        bc, bc.clone(), bc.clone(), ac, rgc,
        _weights(bc, bc, torch.tensor(0.18, dtype=dtype), 0.36),
        _weights(ac, bc, torch.tensor(0.20, dtype=dtype), 0.36),
        _weights(rgc.expand(2, 2), bc, torch.tensor((0.15, 0.30), dtype=dtype), 0.45),
        _weights(rgc.expand(2, 2), ac, torch.tensor((0.15, 0.30), dtype=dtype), 0.45),
    )


def area_integral_weights(pixel_bounds: Tensor, node_xy: Tensor) -> Tensor:
    lower = torch.maximum(pixel_bounds[None, :, :, 0], node_xy[:, None, :] - 0.05)
    upper = torch.minimum(pixel_bounds[None, :, :, 1], node_xy[:, None, :] + 0.05)
    weights = (upper - lower).clamp_min(0).prod(-1) / 0.01
    if not torch.allclose(weights.sum(-1), torch.ones_like(node_xy[:, 0]), rtol=0, atol=2e-6):
        raise ValueError("pixel field does not fully cover each physical node aperture")
    return weights


def _delay1(value: Tensor) -> Tensor:
    return torch.cat((torch.zeros_like(value[:, :1]), value[:, :-1]), dim=1)


def _lowpass(value: Tensor, tau: Tensor | float) -> Tensor:
    flat = value.flatten(2)
    taus = torch.as_tensor(tau, dtype=value.dtype, device=value.device).reshape(-1)
    decay = torch.exp(-DT_MS / taus).repeat(flat.shape[-1] // taus.numel())
    return causal_lowpass(flat, decay).reshape(value.shape)


class LocalRetipathV0(nn.Module):
    def __init__(self, *, dtype: torch.dtype = torch.float32, seed: int | None = None) -> None:
        super().__init__()
        geometry = fixed_geometry(dtype=dtype)
        for name in ("input_xy", "h_xy", "bc_xy", "ac_xy", "rgc_xy", "G", "A", "D", "I"):
            self.register_buffer(name, getattr(geometry, name))
        generator = None if seed is None else torch.Generator().manual_seed(seed)
        raw = {}
        for name, (lo, hi, center, _) in PARAMETERS.items():
            fraction = (center - lo) / (hi - lo)
            initial = math.atanh(center / 4) if name.startswith("delta_") else math.log(fraction / (1 - fraction))
            value = torch.tensor(initial, dtype=dtype)
            if generator is not None:
                value = value + 0.15 * torch.randn((), generator=generator, dtype=dtype)
            raw[name] = nn.Parameter(value)
        self.raw = nn.ParameterDict(raw)

    @property
    def geometry(self) -> CircuitGeometry:
        return CircuitGeometry(self.input_xy, self.h_xy, self.bc_xy, self.ac_xy, self.rgc_xy,
                               self.G, self.A, self.D, self.I)

    def physical_parameters(self) -> dict[str, Tensor]:
        return {name: (4 * raw.tanh() if name.startswith("delta_") else
                       PARAMETERS[name][0] + (PARAMETERS[name][1] - PARAMETERS[name][0]) * raw.sigmoid())
                for name, raw in self.raw.items()}

    def parameter_groups(self, condition: TrainingCondition | str, macro_step: int) -> dict[str, tuple[nn.Parameter, ...]]:
        active = active_parameter_names(condition, macro_step)
        return {group: tuple(self.raw[name] for name in active if PARAMETERS[name][3] == group)
                for group in ("H", "F", "B_s", "B_o", "A", "E", "I", "R")}

    def configure_trainable(self, condition: TrainingCondition | str, macro_step: int) -> tuple[str, ...]:
        active = active_parameter_names(condition, macro_step)
        for name, parameter in self.raw.items():
            parameter.requires_grad_(name in active)
            parameter.grad = None
        return active

    def forward(self, stimulus: StimulusBatch, *, observed_events: Tensor,
                intervention: InterventionSpec = InterventionSpec()) -> StateBundle:
        stimulus.validate()
        if not isinstance(intervention, InterventionSpec):
            raise TypeError("intervention must use an explicit InterventionSpec")
        values = stimulus.values
        if values.dtype != self.G.dtype or values.device != self.G.device:
            raise ValueError("model and stimulus must share dtype/device")
        if observed_events.shape != (*values.shape[:2], 1) or observed_events.dtype != values.dtype or observed_events.device != values.device:
            raise ValueError("observed_events must match [B,T,1], dtype and device")
        if not bool(((observed_events == 0) | (observed_events == 1)).all()):
            raise ValueError("observed_events must contain binary occupancies")
        theta = self.physical_parameters()
        Q = area_integral_weights(stimulus.pixel_bounds, self.input_xy)
        x = values[..., 0] @ Q.T
        u_h = x @ self.G.T
        h = _lowpass(_delay1(u_h), theta["tau_h"])
        feedback = theta["a_h"] * (h @ self.G)
        u_b = x - feedback
        q_f, q_s = _lowpass(u_b, theta["tau_f"]), _lowpass(u_b, theta["tau_s"])
        s_b = torch.stack((q_f, q_f - q_s), -1)
        o_b = torch.where(s_b >= 0, s_b, theta["alpha"] * s_b)
        w_e = torch.stack((theta["delta_e"], theta["delta_e"].new_zeros(()))).softmax(0)
        w_i = torch.stack((theta["delta_i"], theta["delta_i"].new_zeros(()))).softmax(0)
        d_e = theta["g_e"] * (o_b * w_e).sum(-1)[..., None] * self.D.T
        if intervention.kind is Intervention.BLOCK_DIRECT_BC_DRIVE:
            d_e = torch.zeros_like(d_e)
        u_e = d_e.sum(-2, keepdim=True)
        u_a = torch.einsum("ji,btid->btjd", self.A, o_b)
        a_ac = _lowpass(_delay1(u_a), torch.stack((theta["tau_a1"], theta["tau_a2"])))
        j_i = -theta["g_i"] * (a_ac * w_i).sum(-1)[..., None] * self.I.T
        u_i = -j_i.sum(-2, keepdim=True)
        b0 = math.log(math.expm1(1.0))
        g_e, g_i = F.softplus(b0 + u_e), F.softplus(b0 + u_i)
        total = 1 + g_e + g_i
        v_inf = (g_e - g_i / 3) / total
        voltage = exponential_sequence(v_inf, -torch.expm1(-DT_MS * total / 60), 2 / 9)
        membrane = voltage.mean(-1) - 2 / 9
        adaptation = _lowpass(membrane, 120.0)
        history = fixed_one_bin_history_state(observed_events, math.exp(-DT_MS / 40))
        logits = 8 * membrane - 0.5 * adaptation - history + theta["bias"]
        return StateBundle(
            inputs={"x": x[..., None], "u_H": u_h[..., None], "u_B": u_b[..., None], "u_A": u_a},
            states={"h": h[..., None], "q_f": q_f[..., None], "q_s": q_s[..., None], "s_B": s_b,
                    "a_AC": a_ac, "V": voltage, "z": adaptation, "q": history},
            outputs={"f": feedback[..., None], "o_B": o_b, "d_E": d_e, "j_I": j_i,
                     "uE": u_e, "uI": u_i, "gE": g_e, "gI": g_i, "m": membrane,
                     "ell": logits, "p": logits.sigmoid()},
            sequence_id=stimulus.sequence_id, split_group_id=stimulus.split_group_id,
            intervention=intervention,
        )
