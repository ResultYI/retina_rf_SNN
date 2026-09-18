from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

import torch

from models.mechanistic_retina.contracts import PathwayClamp
from models.mechanistic_retina.retipath import RetiPath
from models.mechanistic_retina.retipath_canonical_gain import CanonicalGainRetiPath, CanonicalSpatialComponents
from models.mechanistic_retina.retipath_spatial_ei import SpatialEIBackend
from models.mechanistic_retina.state import causal_lowpass, fixed_one_bin_history_state


class PhysiologyStatus(StrEnum):
    EFFECTIVE = "EFFECTIVE"
    PHYSIOLOGY_MOTIVATED = "PHYSIOLOGY_MOTIVATED"
    PHYSIOLOGICALLY_VALIDATED = "PHYSIOLOGICALLY_VALIDATED"


@dataclass(frozen=True)
class TraceVariableSpec:
    name: str
    shape: str
    mathematical_meaning: str
    normalization_unit: str
    physiology_interpretation: str
    status: PhysiologyStatus = PhysiologyStatus.EFFECTIVE


_PM = PhysiologyStatus.PHYSIOLOGY_MOTIVATED
_SPECS = (
    TraceVariableSpec("input_weber", "[B,T,X]", "x: supplied L+M Weber stimulus proxy", "dimensionless Weber contrast", "Not measured cone current or cone release"),
    TraceVariableSpec("h1_graph_edge_index", "[2,E]", "row/source indices of fixed G", "integer input-grid indices", "Effective local connectivity, not anatomical synapses", _PM),
    TraceVariableSpec("h1_graph_edge_weight", "[E]", "row-normalized Gaussian edge weights of G", "dimensionless; sum per row = 1", "Spatial pooling prior", _PM),
    TraceVariableSpec("h1_graph_drive", "[B,T,X]", "G x", "Weber-derived effective drive", "Pooled outer-retinal input proxy", _PM),
    TraceVariableSpec("h1_state", "[B,T,X]", "LP_H(delay_H(G x)); initial state 0", "Weber-derived effective state; not mV", "Delayed low-pass H1-like state, not validated H1 membrane voltage", _PM),
    TraceVariableSpec("h1_feedback", "[B,T,X]", "A_H G^T h1_state; set to zero by feedback block", "Weber-derived effective drive", "Subtractive surround contribution; not a measured feedback current", _PM),
    TraceVariableSpec("h1_modulated_input", "[B,T,X]", "x - h1_feedback", "Weber-derived effective drive", "Input to BC feature bank, not cone release"),
    TraceVariableSpec("bc_direct_effective_drive", "[B,T,N,K,2]", "D_kp = s_p sum_r W_pkr F_direct,pkr; common branch slope s_p", "signed effective drive before cell gain and block", "Direct sustained/transient-labeled BC feature components; not release"),
    TraceVariableSpec("bc_broad_effective_drive", "[B,T,N,K,2]", "Q_kp = s_p sum_r W_pkr F_broad,pkr; branch-specific common slope", "signed effective drive before AC delay/filter", "Broad input to AC branches; same polarity and weights as direct bank"),
    TraceVariableSpec("bc_direct_transmitted_drive", "[B,T,N,K,2]", "beta_BC D_kp after direct pathway mask", "signed effective drive", "Drive reaching excitatory input mapping; not nA or release"),
    TraceVariableSpec("ac_states", "[B,T,N,K,2]", "A_kp = LP_AC,p(delay_AC,p(Q_kp))", "signed effective state before gate/gain", "Local/transient-labeled filtering states; no AC-pre or BC-target state"),
    TraceVariableSpec("ac_postsynaptic_signed_drive", "[B,T,N,K,2]", "I_kp = -beta_AC q_p A_kp after AC gate/block", "signed effective current proxy; not nA", "Postsynaptic inhibitory pathway contribution; sign removed before positive mapping"),
    TraceVariableSpec("ac_inhibitory_drive", "[B,T,N,K,2]", "-I_kp", "signed deviation from baseline", "Negative values mean reduced inhibitory drive, not negative conductance"),
    TraceVariableSpec("uE", "[B,T,N,K]", "sum_p bc_direct_transmitted_drive", "non-canonical scale-dependent quantity", "Legacy alias: excitatory input in the old gain convention"),
    TraceVariableSpec("uI", "[B,T,N,K]", "-sum_p ac_postsynaptic_signed_drive", "non-canonical scale-dependent quantity", "Legacy alias: inhibitory input in the old gain convention, never abs(current)"),
    TraceVariableSpec("gE", "[B,T,N,K]", "softplus(b0 + aE uE/sE)", "normalized conductance; gE(0)=1; not nS", "Positive effective E conductance", _PM),
    TraceVariableSpec("gI", "[B,T,N,K]", "softplus(b0 + aI uI/sI); zero only for explicit total-conductance removal", "normalized conductance; normal gI(0)=1; not nS", "Positive effective I conductance; drive block retains background", _PM),
    TraceVariableSpec("V1", "[B,T,N]", "first mode of exact per-bin conductance voltage update from V0=2/9", "normalized voltage; EL=0, EE=1, EI=-1/3; not mV", "First effective spatial mode, not a dendritic compartment", _PM),
    TraceVariableSpec("V2", "[B,T,N]", "second mode of exact per-bin conductance voltage update from V0=2/9", "normalized voltage; not mV", "Second effective spatial mode, not a dendritic compartment", _PM),
    TraceVariableSpec("membrane_readout", "[B,T,N]", "exp(log_output_scale) * (mean_k V_k - V0)", "scaled normalized voltage", "Effective pooled membrane signal; no second membrane low-pass"),
    TraceVariableSpec("adaptation_state", "[B,T,N]", "LP_adaptation(membrane_readout); initial state 0", "signed readout scale", "Effective output adaptation, not a specific ion-channel state"),
    TraceVariableSpec("adaptation_term", "[B,T,N]", "eta * adaptation_state; zero for term removal", "logit units; subtracted from logit", "Output correction; state retained when term removed"),
    TraceVariableSpec("history_input", "[B,T,N]", "supplied observed occupancy, or all zeros for FIX_HISTORY_ZERO", "events/bin (Bernoulli occupancy for formal scoring)", "Conditioning sequence, not generated spikes"),
    TraceVariableSpec("history_state", "[B,T,N]", "LP_history(history_input shifted by one bin); initial state 0", "filtered occupancy", "Strictly-past conditional spike-history feature"),
    TraceVariableSpec("history_term", "[B,T,N]", "q_history * history_gain * history_state", "logit units; subtracted from logit", "Effective suppressive history contribution, not an identified refractory channel"),
    TraceVariableSpec("logit", "[B,T,N]", "slope*(membrane_readout-threshold)-adaptation_term-history_term+bias", "log odds per bin", "Conditional Bernoulli response drive"),
    TraceVariableSpec("probability", "[B,T,N]", "sigmoid(logit)", "probability of occupancy in one dt bin; not Hz", "Conditional prediction; zero-history response is not free-running mean"),
)
TRACE_VARIABLE_SPECS: Mapping[str, TraceVariableSpec] = MappingProxyType({s.name: s for s in _SPECS})

_CANONICAL_OVERRIDES = (
    TraceVariableSpec("bc_direct_effective_drive", "[B,T,N,K,2]", "x_E,kp using per-branch normalized BC basis weights", "canonical component scale, before pathway composition and strength", "Effective BC components, not release"),
    TraceVariableSpec("bc_broad_effective_drive", "[B,T,N,K,2]", "broad components using the same per-branch normalized BC weights", "canonical component scale", "Effective BC input to AC, not release"),
    TraceVariableSpec("bc_direct_transmitted_drive", "[B,T,N,K,2]", "G_E w_E,p x_E,kp after direct mask", "canonical effective-drive units before frozen RMS division", "Effective direct-path contribution"),
    TraceVariableSpec("ac_states", "[B,T,N,K,2]", "x_I,kp = original delay/lowpass applied to canonical broad components", "canonical component scale, before pathway composition and strength", "Effective AC states, not membrane voltage"),
    TraceVariableSpec("ac_postsynaptic_signed_drive", "[B,T,N,K,2]", "-G_I w_I,p x_I,kp after AC output mask", "canonical effective-drive units before frozen RMS division", "Signed effective postsynaptic contribution"),
    TraceVariableSpec("ac_inhibitory_drive", "[B,T,N,K,2]", "G_I w_I,p x_I,kp after AC output mask", "signed canonical drive deviation; not nA", "Negative values mean below-background input"),
    TraceVariableSpec("uE", "[B,T,N,K]", "effective_drive_E / frozen legacy_a_E", "non-canonical scale-dependent quantity", "Legacy alias only; excluded from canonical prediction path"),
    TraceVariableSpec("uI", "[B,T,N,K]", "effective_drive_I / frozen legacy_a_I", "non-canonical scale-dependent quantity", "Legacy alias only; excluded from canonical prediction path"),
    TraceVariableSpec("gE", "[B,T,N,K]", "softplus(b0 + effective_drive_E / frozen sE)", "normalized conductance; not nS", "Physiology-motivated effective quantity", _PM),
    TraceVariableSpec("gI", "[B,T,N,K]", "softplus(b0 + effective_drive_I / frozen sI); explicit total block sets zero", "normalized conductance; not nS", "Physiology-motivated effective quantity", _PM),
    TraceVariableSpec("E_components", "[B,T,N,K,2]", "x_E,kp = bc_direct_effective_drive, before mask", "per-branch normalized basis scale", "Effective synaptic input components"),
    TraceVariableSpec("I_components", "[B,T,N,K,2]", "x_I,kp = ac_states, before output mask", "per-branch normalized basis scale", "Effective AC input components"),
    TraceVariableSpec("G_E", "[N]", "aE*(beta_BC_s+beta_BC_t); stored as exp(log_G_E)", "positive effective overall E strength", "Not a calibrated synaptic conductance"),
    TraceVariableSpec("G_I", "[N]", "aI*(beta_AC_local+beta_AC_transient); stored as exp(log_G_I)", "positive effective overall I strength", "Not a calibrated synaptic conductance"),
    TraceVariableSpec("w_E", "[N,2]", "softmax([delta_E,0]); delta_E=log(w_E_s/w_E_t)", "nonnegative relative weights summing to one", "Coefficient composition, not measured synaptic fractions"),
    TraceVariableSpec("w_I", "[N,2]", "softmax([delta_I,0]); delta_I=log(w_I_local/w_I_transient)", "nonnegative relative weights summing to one", "Coefficient composition, not measured synaptic fractions"),
    TraceVariableSpec("relative_drive_E", "[B,T,N,K]", "sum_p w_E,p x_E,kp after direct mask", "composition-normalized component scale", "Effective mixture before overall strength"),
    TraceVariableSpec("relative_drive_I", "[B,T,N,K]", "sum_p w_I,p x_I,kp after AC output mask", "composition-normalized component scale", "Effective mixture before overall strength"),
    TraceVariableSpec("normalized_drive_E", "[B,T,N,K]", "relative_drive_E / frozen sE", "fixed training-RMS normalized input", "Effective normalized E deviation"),
    TraceVariableSpec("normalized_drive_I", "[B,T,N,K]", "relative_drive_I / frozen sI", "fixed training-RMS normalized input", "Effective normalized I deviation"),
    TraceVariableSpec("effective_drive_E", "[B,T,N,K]", "G_E * relative_drive_E", "canonical effective drive; divide by sE in conductance mapping", "Effective E input strength times composition"),
    TraceVariableSpec("effective_drive_I", "[B,T,N,K]", "G_I * relative_drive_I", "canonical effective drive; divide by sI in conductance mapping", "Effective I input strength times composition"),
)
CANONICAL_TRACE_VARIABLE_SPECS: Mapping[str, TraceVariableSpec] = MappingProxyType({
    **TRACE_VARIABLE_SPECS, **{s.name: s for s in _CANONICAL_OVERRIDES},
})


class Intervention(StrEnum):
    NORMAL = "NORMAL"
    BLOCK_H1_FEEDBACK = "BLOCK_H1_FEEDBACK"
    BLOCK_DIRECT_BC_DRIVE = "BLOCK_DIRECT_BC_DRIVE"
    BLOCK_AC_POSTSYNAPTIC_DRIVE = "BLOCK_AC_POSTSYNAPTIC_DRIVE"
    REMOVE_TOTAL_INHIBITORY_CONDUCTANCE = "REMOVE_TOTAL_INHIBITORY_CONDUCTANCE"
    REMOVE_ADAPTATION_TERM = "REMOVE_ADAPTATION_TERM"
    FIX_HISTORY_ZERO = "FIX_HISTORY_ZERO"


@dataclass(frozen=True)
class InterventionSemantics:
    target: str
    operation: str
    baseline_preserved: bool
    state_preserved: tuple[str, ...]
    downstream_recomputed: tuple[str, ...]
    history_condition: str
    biological_analogue: str
    known_mismatch: str


_TAIL = ("V1", "V2", "membrane_readout", "adaptation_state", "adaptation_term", "logit", "probability")
_OBSERVED = "supplied observed history; strictly past; fixed across stimulus-pathway comparisons"
_SEMANTICS = {
    Intervention.NORMAL: InterventionSemantics(
        "none", "unmodified formal forward calculation", True, ("all normal state rules",), (), _OBSERVED,
        "none; conditional observation", "Latent-state correspondence is not physiological validation."),
    Intervention.BLOCK_H1_FEEDBACK: InterventionSemantics(
        "h1_feedback", "set H1 output amplitude to zero; recompute x-feedback", True,
        ("h1_graph_drive", "h1_state", "history_state"),
        ("h1_modulated_input", "bc_direct_effective_drive", "bc_broad_effective_drive", "bc_direct_transmitted_drive", "ac_states", "ac_postsynaptic_signed_drive", "ac_inhibitory_drive", "uE", "uI", "gE", "gI", *_TAIL), _OBSERVED,
        "ideal selective interruption of modeled outer-retinal feedback output",
        "Not H1 ablation, voltage clamp, or a specific drug; both downstream E and I may change."),
    Intervention.BLOCK_DIRECT_BC_DRIVE: InterventionSemantics(
        "bc_direct_transmitted_drive", "mask both direct BC branches before cell gain; uE=0, gE=softplus(b0)", True,
        ("h1_state", "bc_direct_effective_drive", "bc_broad_effective_drive", "ac_states", "history_state"),
        ("uE", "gE", *_TAIL), _OBSERVED,
        "ideal suppression of stimulus-dependent direct excitatory drive",
        "Retains tonic gE background and broad BC to AC; not BC-cell silencing or total glutamate-receptor blockade."),
    Intervention.BLOCK_AC_POSTSYNAPTIC_DRIVE: InterventionSemantics(
        "ac_postsynaptic_signed_drive", "zero both AC output gates; uI=0, gI=softplus(b0)", True,
        ("h1_state", "bc_direct_effective_drive", "bc_broad_effective_drive", "ac_states", "history_state"),
        ("ac_inhibitory_drive", "uI", "gI", *_TAIL), _OBSERVED,
        "ideal suppression of stimulus-dependent modeled postsynaptic inhibitory drive",
        "Removes positive and negative drive deviations, retains tonic gI background; no BC-presynaptic inhibition or receptor specificity."),
    Intervention.REMOVE_TOTAL_INHIBITORY_CONDUCTANCE: InterventionSemantics(
        "gI", "replace mapped gI by zero for every bin; retain original V0 initial state and Cm", False,
        ("h1_state", "bc_direct_effective_drive", "bc_broad_effective_drive", "ac_states", "uE", "uI", "gE", "history_state"),
        _TAIL, _OBSERVED,
        "ideal removal of all modeled inhibitory conductance at sequence onset",
        "Includes tonic background; not the AC drive block. No receptor specificity, compensatory changes, or new resting-state equilibration."),
    Intervention.REMOVE_ADAPTATION_TERM: InterventionSemantics(
        "adaptation_term", "set subtracted adaptation term to zero; retain adaptation state", True,
        ("h1_state", "ac_states", "V1", "V2", "membrane_readout", "adaptation_state", "history_state"),
        ("logit", "probability"), _OBSERVED,
        "removal of the modeled output adaptation contribution",
        "Not a specific ion-channel block; adaptation state and all upstream dynamics remain."),
    Intervention.FIX_HISTORY_ZERO: InterventionSemantics(
        "history_input", "condition on all-zero observed occupancy before the one-bin shift/filter", True,
        ("h1_state", "ac_states", "V1", "V2", "membrane_readout", "adaptation_state"),
        ("history_state", "history_term", "logit", "probability"),
        "all-zero conditioning sequence; no generated-spike feedback",
        "conditional computational probe with no prior spikes",
        "Not physical elimination of refractoriness; differs from retaining observed history state while gating its output."),
}
_CLAMPS = {
    Intervention.BLOCK_H1_FEEDBACK: frozenset({PathwayClamp.H1}),
    Intervention.BLOCK_DIRECT_BC_DRIVE: frozenset({PathwayClamp.DIRECT_BC_SUSTAINED, PathwayClamp.DIRECT_BC_TRANSIENT}),
    Intervention.BLOCK_AC_POSTSYNAPTIC_DRIVE: frozenset({PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}),
    Intervention.REMOVE_ADAPTATION_TERM: frozenset({PathwayClamp.RGC_ADAPTATION}),
}


@dataclass(frozen=True)
class InterventionSpec:
    kind: Intervention = Intervention.NORMAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", Intervention(self.kind))

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.kind.value,
            **asdict(_SEMANTICS[self.kind]),
            "baseline_definition": "zero Weber input and zero history; gE0=gI0=softplus(b0), original V0 and bias",
            "time_scope": "entire independent sequence; original V0 initialization; no pre-equilibration",
            "legacy_clamps": sorted(c.value for c in _CLAMPS.get(self.kind, frozenset())),
        }


@dataclass(frozen=True)
class PhysiologyTrace:
    tensors: Mapping[str, torch.Tensor]
    intervention: InterventionSpec
    schema_version: int = 1

    @property
    def logits(self) -> torch.Tensor:
        return self.tensors["logit"]

    @property
    def probability(self) -> torch.Tensor:
        return self.tensors["probability"]

    def metadata(self) -> dict[str, object]:
        specs = CANONICAL_TRACE_VARIABLE_SPECS if self.schema_version == 2 else TRACE_VARIABLE_SPECS
        return {
            "schema_version": self.schema_version,
            "intervention": self.intervention.to_record(),
            "variables": [
                {**asdict(specs[name]), "actual_shape": list(value.shape),
                 "dtype": str(value.dtype), "device": str(value.device), "requires_grad": value.requires_grad}
                for name, value in self.tensors.items()
            ],
        }


def observe_mechanism(
    model: RetiPath,
    cones: torch.Tensor,
    *,
    observed_counts: torch.Tensor,
    intervention: InterventionSpec = InterventionSpec(),
) -> PhysiologyTrace:
    """Observe the formal model without mutation; returned tensors retain autograd."""
    if type(model) not in (RetiPath, CanonicalGainRetiPath) or model.backend is not SpatialEIBackend.CONDUCTANCE:
        raise ValueError("mechanism observation requires the formal RetiPath conductance entry point")
    if not isinstance(intervention, InterventionSpec):
        raise TypeError("intervention must be an explicit InterventionSpec")
    if cones.ndim != 3 or cones.shape[1] == 0 or observed_counts.shape != (*cones.shape[:2], 1):
        raise ValueError("expected nonempty [B,T,X] input and [B,T,1] observed history")
    integrator = model.spatial_ei
    if integrator is None:
        raise ValueError("formal RetiPath requires its spatial integrator")
    kind = intervention.kind
    clamps = _CLAMPS.get(kind, frozenset())
    parts = model.spatial_components(cones, clamps)
    u_e, u_i = parts.u_e, parts.u_i
    g_e, g_i = integrator.conductances(u_e, u_i)
    if kind is Intervention.REMOVE_TOTAL_INHIBITORY_CONDUCTANCE:
        g_i = torch.zeros_like(g_i)
    voltage = integrator.voltage(g_e, g_i)
    membrane = integrator.log_output_scale.exp() * (voltage.mean(-1) - integrator.v_0)
    adaptation = causal_lowpass(membrane, model.rgc.adaptation_decay)
    history_input = torch.zeros_like(observed_counts) if kind is Intervention.FIX_HISTORY_ZERO else observed_counts
    history = fixed_one_bin_history_state(history_input, float(model.rgc.history_decay))
    gates = model.gates.values(clamps)
    adaptation_term = 0 if kind is Intervention.REMOVE_ADAPTATION_TERM else model.rgc.adaptation_gain * adaptation
    # Match the frozen forward's multiplication and subtraction order exactly.
    history_term = gates.history * model.rgc.history_gain * history
    logits = (model.rgc.logit_slope * (membrane - model.rgc.threshold) - adaptation_term
              - history_term + model.rgc.response_bias)
    values = {
        "input_weber": cones,
        "h1_graph_edge_index": model.h1.graph.edge_index.clone(),
        "h1_graph_edge_weight": model.h1.graph.edge_weight.clone(),
        "h1_graph_drive": parts.h1.graph_drive,
        "h1_state": parts.h1.state,
        "h1_feedback": parts.h1.surround,
        "h1_modulated_input": parts.h1.modulated_cones,
        "bc_direct_effective_drive": parts.direct,
        "bc_broad_effective_drive": parts.broad,
        "bc_direct_transmitted_drive": parts.currents[..., :2],
        "ac_states": parts.ac_states,
        "ac_postsynaptic_signed_drive": parts.currents[..., 2:],
        "ac_inhibitory_drive": -parts.currents[..., 2:],
        "uE": u_e, "uI": u_i, "gE": g_e, "gI": g_i,
        "V1": voltage[..., 0], "V2": voltage[..., 1],
        "membrane_readout": membrane,
        "adaptation_state": adaptation,
        "adaptation_term": torch.zeros_like(adaptation) if isinstance(adaptation_term, int) else adaptation_term,
        "history_input": history_input, "history_state": history, "history_term": history_term,
        "logit": logits, "probability": torch.sigmoid(logits),
    }
    if isinstance(model, CanonicalGainRetiPath):
        if not isinstance(parts, CanonicalSpatialComponents):
            raise TypeError("canonical model must expose canonical spatial components")
        gain = model.canonical_gains
        values.update({
            "E_components": parts.direct, "I_components": parts.ac_states,
            "G_E": gain.G_E, "G_I": gain.G_I, "w_E": gain.w_E, "w_I": gain.w_I,
            "relative_drive_E": parts.relative_e, "relative_drive_I": parts.relative_i,
            "normalized_drive_E": parts.relative_e / integrator.rms_e,
            "normalized_drive_I": parts.relative_i / integrator.rms_i,
            "effective_drive_E": u_e, "effective_drive_I": u_i,
            "uE": u_e / gain.legacy_a_E[None, None, :, None],
            "uI": u_i / gain.legacy_a_I[None, None, :, None],
        })
        return PhysiologyTrace(MappingProxyType(values), intervention, schema_version=2)
    return PhysiologyTrace(MappingProxyType(values), intervention)
