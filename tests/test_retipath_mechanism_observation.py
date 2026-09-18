import json
from dataclasses import asdict

import pytest
import torch

from evaluation.mechanistic_retina.mechanism_observation import (
    Intervention, InterventionSpec, PhysiologyStatus, TRACE_VARIABLE_SPECS, observe_mechanism,
)
from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig, PathwayClamp
from models.mechanistic_retina.retipath import RetiPath


def bitwise_equal(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype
            and a.detach().cpu().numpy().tobytes() == b.detach().cpu().numpy().tobytes())


def assert_bits(a, b):
    assert bitwise_equal(a, b), float((a - b).abs().max())


def check_normal(model, x, y):
    with torch.no_grad():
        expected = model.forward_sequence(x, observed_counts=y)
        actual = observe_mechanism(model, x, observed_counts=y)
    assert_bits(actual.logits, expected.logits)
    assert_bits(actual.probability, expected.spike_probability)
    aliases = {
        "h1_graph_drive": expected.cone_graph_drive, "h1_state": expected.h1_state,
        "h1_feedback": expected.h1_surround_contribution, "membrane_readout": expected.rgc_membrane,
        "adaptation_state": expected.rgc_adaptation, "history_state": expected.rgc_history_state,
    }
    for name, value in aliases.items():
        assert_bits(actual.tensors[name], value)
    assert_bits(actual.tensors["bc_direct_effective_drive"].sum(-2), expected.bc_direct_presynaptic)
    assert_bits(actual.tensors["bc_broad_effective_drive"].sum(-2), expected.bc_broad_presynaptic)
    assert_bits(actual.tensors["ac_states"].sum(-2)[..., 0], expected.amacrine_local_state)
    assert_bits(actual.tensors["ac_states"].sum(-2)[..., 1], expected.amacrine_transient_state)
    assert set(actual.tensors) == set(TRACE_VARIABLE_SPECS)
    json.dumps(actual.metadata())
    assert all(s.status != PhysiologyStatus.PHYSIOLOGICALLY_VALIDATED for s in TRACE_VARIABLE_SPECS.values())
    return {"logits_bitwise_equal": True, "probability_bitwise_equal": True, "max_abs_error": 0.0,
            "legacy_trace_aliases_bitwise_equal": True, "variable_count": len(actual.tensors)}


def check_interventions(model, x, y):
    before = {k: v.clone() for k, v in model.state_dict().items()}
    flags = [(m.training) for m in model.modules()]
    gradients = [(p.requires_grad, None if p.grad is None else p.grad.clone()) for p in model.parameters()]
    rng = torch.random.get_rng_state().clone()
    input_before, history_before = x.clone(), y.clone()
    tail = {"V1", "V2", "membrane_readout", "adaptation_state", "adaptation_term", "logit", "probability"}
    allowed = {
        Intervention.BLOCK_H1_FEEDBACK: tail | {"h1_feedback", "h1_modulated_input", "bc_direct_effective_drive", "bc_broad_effective_drive", "bc_direct_transmitted_drive", "ac_states", "ac_postsynaptic_signed_drive", "ac_inhibitory_drive", "uE", "uI", "gE", "gI"},
        Intervention.BLOCK_DIRECT_BC_DRIVE: tail | {"bc_direct_transmitted_drive", "uE", "gE"},
        Intervention.BLOCK_AC_POSTSYNAPTIC_DRIVE: tail | {"ac_postsynaptic_signed_drive", "ac_inhibitory_drive", "uI", "gI"},
        Intervention.REMOVE_TOTAL_INHIBITORY_CONDUCTANCE: tail | {"gI"},
        Intervention.REMOVE_ADAPTATION_TERM: {"adaptation_term", "logit", "probability"},
        Intervention.FIX_HISTORY_ZERO: {"history_input", "history_state", "history_term", "logit", "probability"},
    }
    legacy = {
        Intervention.BLOCK_H1_FEEDBACK: {PathwayClamp.H1},
        Intervention.BLOCK_DIRECT_BC_DRIVE: {PathwayClamp.DIRECT_BC_SUSTAINED, PathwayClamp.DIRECT_BC_TRANSIENT},
        Intervention.BLOCK_AC_POSTSYNAPTIC_DRIVE: {PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT},
        Intervention.REMOVE_ADAPTATION_TERM: {PathwayClamp.RGC_ADAPTATION},
    }
    report = {}
    with torch.no_grad():
        normal = observe_mechanism(model, x, observed_counts=y)
        for kind, permitted in allowed.items():
            trace = observe_mechanism(model, x, observed_counts=y, intervention=InterventionSpec(kind))
            t = trace.tensors
            changed = {name for name in t if not bitwise_equal(t[name], normal.tensors[name])}
            assert changed <= permitted, (kind, changed - permitted)
            assert all(torch.isfinite(v).all() for v in t.values())
            assert (t["gE"] >= 0).all() and (t["gI"] >= 0).all()
            reference = None
            if kind in legacy:
                reference = model.forward_sequence(x, observed_counts=y, clamps=frozenset(legacy[kind]))
            elif kind is Intervention.FIX_HISTORY_ZERO:
                reference = model.forward_sequence(x, observed_counts=torch.zeros_like(y))
            if reference is not None:
                assert_bits(trace.logits, reference.logits)
                assert_bits(trace.probability, reference.spike_probability)
            if kind is Intervention.BLOCK_H1_FEEDBACK:
                assert torch.count_nonzero(t["h1_feedback"]) == 0
                assert_bits(t["h1_modulated_input"], x)
            if kind is Intervention.BLOCK_DIRECT_BC_DRIVE:
                assert torch.count_nonzero(t["uE"]) == 0
                background, _ = model.spatial_ei.conductances(torch.zeros_like(t["uE"]), t["uI"])
                assert_bits(t["gE"], background)
            if kind is Intervention.BLOCK_AC_POSTSYNAPTIC_DRIVE:
                assert torch.count_nonzero(t["uI"]) == 0
                _, background = model.spatial_ei.conductances(t["uE"], torch.zeros_like(t["uI"]))
                assert_bits(t["gI"], background)
                assert (t["gI"] > 0).all()
            if kind is Intervention.REMOVE_TOTAL_INHIBITORY_CONDUCTANCE:
                assert torch.count_nonzero(t["gI"]) == 0
                integrator = model.spatial_ei
                ge = t["gE"][:, 0].double()
                target = ge / (1 + ge)
                v_first = target + (integrator.v_0 - target) * torch.exp(-integrator.dt_ms * (1 + ge) / integrator.c_m)
                actual = torch.stack((t["V1"][:, 0], t["V2"][:, 0]), -1)
                torch.testing.assert_close(actual.double(), v_first, atol=1e-7, rtol=1e-6)
            if kind is Intervention.REMOVE_ADAPTATION_TERM:
                assert torch.count_nonzero(t["adaptation_term"]) == 0
            if kind is Intervention.FIX_HISTORY_ZERO:
                assert all(torch.count_nonzero(t[n]) == 0 for n in ("history_input", "history_state", "history_term"))
            record = trace.intervention.to_record()
            assert {"target", "operation", "baseline_preserved", "state_preserved", "downstream_recomputed", "history_condition", "biological_analogue", "known_mismatch"} <= set(record)
            json.dumps(record)
            report[kind.value] = {"only_declared_targets_and_downstream_changed": True,
                                  "changed_variables": sorted(changed), "finite_nonnegative_conductances": True,
                                  "legacy_or_zero_history_prediction_bitwise_equal": True if reference is not None else None}
        zero_normal = observe_mechanism(model, torch.zeros_like(x), observed_counts=torch.zeros_like(y))
        for kind in allowed:
            zero = observe_mechanism(model, torch.zeros_like(x), observed_counts=torch.zeros_like(y), intervention=InterventionSpec(kind))
            if kind is Intervention.REMOVE_TOTAL_INHIBITORY_CONDUCTANCE:
                assert not bitwise_equal(zero.tensors["V1"], zero_normal.tensors["V1"])
            else:
                assert_bits(zero.logits, zero_normal.logits)
            report[kind.value]["declared_zero_input_baseline_behavior_verified"] = True
    for k, v in model.state_dict().items():
        assert_bits(v, before[k])
    assert flags == [m.training for m in model.modules()]
    for p, (requires_grad, grad) in zip(model.parameters(), gradients, strict=True):
        assert p.requires_grad == requires_grad
        assert (p.grad is None) if grad is None else bitwise_equal(p.grad, grad)
    assert_bits(torch.random.get_rng_state(), rng)
    assert_bits(x, input_before)
    assert_bits(y, history_before)
    return {"interventions": report, "parameters_buffers_modes_gradients_rng_and_inputs_unchanged": True}


def check_causality(model, x, y):
    t = x.shape[1] // 2
    changed_x = x.clone()
    changed_x[:, t + 1:] = -changed_x[:, t + 1:] + 0.75
    changed_y = y.clone()
    changed_y[:, t:] = 1 - changed_y[:, t:]
    results = {}
    for kind in Intervention:
        spec = InterventionSpec(kind)
        with torch.no_grad():
            normal = observe_mechanism(model, x, observed_counts=y, intervention=spec)
            future = observe_mechanism(model, changed_x, observed_counts=y, intervention=spec)
            spikes = observe_mechanism(model, x, observed_counts=changed_y, intervention=spec)
        for name in normal.tensors:
            if name not in {"h1_graph_edge_index", "h1_graph_edge_weight", "history_input"}:
                assert_bits(normal.tensors[name][:, :t + 1], future.tensors[name][:, :t + 1])
                assert_bits(normal.tensors[name][:, :t + 1], spikes.tensors[name][:, :t + 1])
        differentiable_x = x.clone().requires_grad_(True)
        trace = observe_mechanism(model, differentiable_x, observed_counts=y, intervention=spec)
        for voltage_name in ("V1", "V2"):
            value = trace.tensors[voltage_name]
            voltage_gradient = torch.autograd.grad(value[:, t].sum(), differentiable_x, retain_graph=True)[0]
            assert torch.isfinite(voltage_gradient).all()
            assert voltage_gradient[:, :t - 16].abs().sum() > 0
        gradient = torch.autograd.grad(trace.logits[:, t].sum(), differentiable_x)[0]
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient[:, t + 1:]) == 0
        assert gradient[:, :t + 1].abs().sum() > 0
        if kind is Intervention.NORMAL:
            official = model.forward_sequence(differentiable_x, observed_counts=y)
            official_gradient = torch.autograd.grad(official.logits[:, t].sum(), differentiable_x)[0]
            assert_bits(gradient, official_gradient)
        results[kind.value] = {"future_stimulus_and_current_future_spike_invariance_bitwise": True,
                               "finite_nonzero_input_jacobian": True, "future_input_jacobian_zero": True,
                               "both_voltage_modes_connected_to_prefix_older_than_16_bins": True}
    return results


@pytest.fixture
def fixture_model():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(915)
        positions = torch.cartesian_prod(torch.linspace(-0.3, 0.3, 7), torch.linspace(-0.3, 0.3, 7))
        config = MechanisticRetinaConfig(architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE,
                                        cell_specific_gains=True, dt_ms=1000 / 150)
        model = RetiPath(config, positions, torch.zeros(1, 2), ("parasol",), ("ON",),
                         rms_e=torch.tensor([[0.8, 1.2]]), rms_i=torch.tensor([[0.6, 1.1]]))
        x = torch.randn(2, 64, 49)
        y = (torch.rand(2, 64, 1) > 0.7).float()
    return model, x, y


def test_normal(fixture_model):
    check_normal(*fixture_model)


def test_intervention_scope_and_nonmutation(fixture_model):
    check_interventions(*fixture_model)


def test_causality_and_autograd(fixture_model):
    check_causality(*fixture_model)


def test_contract_rejects_ambiguous_names(fixture_model):
    model, x, y = fixture_model
    with pytest.raises(ValueError):
        InterventionSpec("AC-off")
    with pytest.raises(TypeError):
        observe_mechanism(model, x, observed_counts=y, intervention="no-H1")
    with pytest.raises(ValueError):
        observe_mechanism(model, x[:, :0], observed_counts=y[:, :0])
    restored = InterventionSpec(**asdict(InterventionSpec(Intervention.BLOCK_H1_FEEDBACK)))
    assert restored.to_record() == InterventionSpec(Intervention.BLOCK_H1_FEEDBACK).to_record()
