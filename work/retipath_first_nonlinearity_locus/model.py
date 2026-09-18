from __future__ import annotations

import torch

from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.retipath_canonical_gain import CanonicalGainRetiPath, CanonicalSpatialComponents
from models.mechanistic_retina.state import causal_lowpass, fixed_one_bin_history_state
from evaluation.mechanistic_retina.mechanism_observation import observe_mechanism


class ConeRelatedEffectiveInputNonlinearityCandidate(CanonicalGainRetiPath):
    def effective_input(self, cones):
        return torch.where(cones >= 0, cones, self.alpha * cones)

    def spatial_components(self, cones, clamps=frozenset()):
        if clamps:
            raise ValueError("This locus comparison does not support pathway interventions")
        gates = self.gates.values(frozenset())
        h1 = self.h1(self.effective_input(cones), amplitude=gates.h1)
        features = self.feature_bank(h1.modulated_cones, mixer=self.shared_subunits)
        weighted = features * self.bipolar.positive_weights().repeat(1, 2, 1, 1)[None, None]
        components = weighted.sum(-1).transpose(-1, -2)
        direct, broad = components[..., :2], components[..., 2:]
        b, t, n, k, p = broad.shape
        ac = self.amacrine(broad.reshape(b, t, n * k, p),
                           local_gate=gates.ac_local, transient_gate=gates.ac_transient)
        ac_states = torch.stack((ac.local_state, ac.transient_state), -1).reshape(b, t, n, k, p)
        ac_currents = torch.stack((ac.local_current, ac.transient_current), -1).reshape(b, t, n, k, p)
        direct_mask = direct.new_ones(2)
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


def build(cp, condition, *, dtype=torch.float32, device="cpu"):
    if condition not in ("M0", "M1"):
        raise ValueError(condition)
    cls = CanonicalGainRetiPath if condition == "M0" else ConeRelatedEffectiveInputNonlinearityCandidate
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cp["seed"])
        model = cls(MechanisticRetinaConfig(**cp["model_config"]), cp["cone_positions_degs"],
                    cp["cell_positions_degs"], tuple(cp["cell_types"]), tuple(cp["polarities"]),
                    rms_e=torch.tensor(cp["rms"]["e"]), rms_i=torch.tensor(cp["rms"]["i"]))
    model.load_state_dict(cp["model"], strict=True)
    model.to(device=device, dtype=dtype)
    params = [p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in params) == 33
    assert len({id(p) for p in params}) == len(params)
    return model, params


def observe(model, cones, observed_counts):
    if type(model) is CanonicalGainRetiPath:
        return dict(observe_mechanism(model, cones, observed_counts=observed_counts).tensors)
    if type(model) is not ConeRelatedEffectiveInputNonlinearityCandidate:
        raise TypeError(type(model))
    parts = model.spatial_components(cones)
    integrator = model.spatial_ei
    g_e, g_i = integrator.conductances(parts.u_e, parts.u_i)
    voltage = integrator.voltage(g_e, g_i)
    membrane = integrator.log_output_scale.exp() * (voltage.mean(-1) - integrator.v_0)
    adaptation = causal_lowpass(membrane, model.rgc.adaptation_decay)
    history = fixed_one_bin_history_state(observed_counts, float(model.rgc.history_decay))
    gates = model.gates.values(frozenset())
    adaptation_term = model.rgc.adaptation_gain * adaptation
    logits = (model.rgc.logit_slope * (membrane - model.rgc.threshold) - adaptation_term
              - gates.history * model.rgc.history_gain * history + model.rgc.response_bias)
    return {
        "h1_graph_drive": parts.h1.graph_drive, "h1_state": parts.h1.state,
        "h1_feedback": parts.h1.surround, "h1_modulated_input": parts.h1.modulated_cones,
        "bc_direct_effective_drive": parts.direct, "bc_broad_effective_drive": parts.broad,
        "ac_states": parts.ac_states, "ac_inhibitory_drive": -parts.currents[..., 2:],
        "effective_drive_E": parts.u_e, "effective_drive_I": parts.u_i,
        "gE": g_e, "gI": g_i, "V1": voltage[..., 0], "V2": voltage[..., 1],
        "membrane_readout": membrane, "adaptation_state": adaptation, "adaptation_term": adaptation_term,
        "logit": logits, "probability": torch.sigmoid(logits), "history_state": history,
    }
