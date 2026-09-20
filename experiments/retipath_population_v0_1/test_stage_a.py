from dataclasses import replace
import inspect
import math

import pytest
import torch

from experiments.retipath_multiscale_v0.contracts import DT_MS, regular_pixel_bounds
from .circuit import BCOutput, Intervention, PopulationRetipath, Stimulus


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(size=32, steps=48, dtype=torch.float64):
    bounds = regular_pixel_bounds(size, dtype=dtype)
    xy = bounds.mean(-1)
    time = (torch.arange(steps, dtype=dtype) + .5) * DT_MS
    spatial = .5 + .3 * xy[:, 0].sign() - .1 * xy[:, 1].sign()
    wave = torch.sin(torch.arange(steps, dtype=dtype) * .23)
    values = (wave[:, None] * spatial)[None].repeat(2, 1, 1)
    values[1] *= -.7
    stimulus = Stimulus(values, bounds, (bounds[..., 1] - bounds[..., 0]).prod(-1),
                        time, torch.ones_like(values, dtype=torch.bool))
    events = torch.zeros(2, steps, 2, dtype=dtype)
    events[0, 3::11, 0] = 1
    events[1, 5::13, 1] = 1
    return stimulus, events


def forward(model, stimulus=None, events=None, **kwargs):
    if stimulus is None:
        stimulus, events = fixture(dtype=model.G.dtype)
    return model(stimulus, observed_events=events, **kwargs)


@pytest.mark.parametrize("kind", list(BCOutput))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shapes_finite_and_baselines(kind, dtype):
    model = PopulationRetipath(bc_output=kind, dtype=dtype)
    stimulus, events = fixture(dtype=dtype)
    result = forward(model, stimulus, events)
    expected = {"h_H": (2, 48, 25), "H1_feedback": (2, 48, 25),
                "s_B": (2, 48, 50), "delta_r_B": (2, 48, 50),
                "a_A": (2, 48, 36), "o_A": (2, 48, 36),
                "d_E": (2, 48, 2, 2), "d_I": (2, 48, 2, 2),
                "gE": (2, 48, 2, 2), "gI": (2, 48, 2, 2), "V": (2, 48, 2, 2),
                "logit": (2, 48, 2), "probability": (2, 48, 2)}
    for name, shape in expected.items():
        assert result.observation(name).shape == shape
    for mapping in (result.inputs, result.states, result.outputs, result.initial_state):
        assert all(torch.isfinite(t).all() for t in mapping.values())
    assert result.outputs["gE"].min() > 0 and result.outputs["gI"].min() > 0
    assert ((result.outputs["probability"] > 0) & (result.outputs["probability"] < 1)).all()
    zero = replace(stimulus, values=torch.zeros_like(stimulus.values))
    baseline = forward(model, zero, torch.zeros_like(events))
    for name in ("h_H", "s_B", "delta_r_B", "a_A", "delta_o_A", "d_E", "d_I"):
        torch.testing.assert_close(baseline.observation(name), torch.zeros_like(baseline.observation(name)))
    for name in ("gE", "gI"):
        torch.testing.assert_close(baseline.outputs[name], torch.ones_like(baseline.outputs[name]))
    torch.testing.assert_close(baseline.states["V"], torch.full_like(baseline.states["V"], 2 / 9))
    if kind is BCOutput.BASELINE_SOFTPLUS:
        assert result.bc_quantity_kind == "nonnegative_release_like_proxy"
        assert result.outputs["r_B"].min() > 0
        assert result.outputs["delta_r_B"].min() < 0
    else:
        assert result.bc_quantity_kind == "signed_effective_output"
        assert result.outputs["r_B"].min() < 0


@pytest.mark.parametrize("kind", list(BCOutput))
def test_causal_prefix_events_input_gradients_and_reset(kind):
    model = PopulationRetipath(bc_output=kind)
    stimulus, events = fixture()
    cut = 23
    original = forward(model, stimulus, events)
    future = stimulus.values.clone()
    future[:, cut:] = 1.8 * future[:, cut:] + .4
    changed_events = events.clone()
    changed_events[:, cut:] = 1 - changed_events[:, cut:]
    changed = forward(model, replace(stimulus, values=future), changed_events)
    for name in original.states:
        torch.testing.assert_close(original.states[name][:, :cut], changed.states[name][:, :cut], atol=1e-12, rtol=1e-12)
    for name, value in original.outputs.items():
        if value.ndim >= 3:
            torch.testing.assert_close(value[:, :cut], changed.outputs[name][:, :cut], atol=1e-12, rtol=1e-12)
    event_only = forward(model, stimulus, changed_events)
    torch.testing.assert_close(original.outputs["logit"][:, :cut + 1], event_only.outputs["logit"][:, :cut + 1])
    assert not torch.equal(original.outputs["logit"][:, cut + 1:], event_only.outputs["logit"][:, cut + 1:])
    x = stimulus.values.clone().requires_grad_()
    e = events.clone().requires_grad_()
    result = forward(model, replace(stimulus, values=x), e)
    dx, de = torch.autograd.grad(result.outputs["logit"][:, cut].sum(), (x, e))
    assert dx[:, :cut].abs().sum() > 0 and dx[:, cut + 1:].count_nonzero() == 0
    assert de[:, :cut].abs().sum() > 0 and de[:, cut:].count_nonzero() == 0
    repeat = forward(model, stimulus, events)
    torch.testing.assert_close(original.outputs["logit"], repeat.outputs["logit"], atol=0, rtol=0)
    short = replace(stimulus, values=stimulus.values[:, :cut], time_ms=stimulus.time_ms[:cut],
                    input_valid=stimulus.input_valid[:, :cut])
    prefix = forward(model, short, events[:, :cut])
    torch.testing.assert_close(original.outputs["logit"][:, :cut], prefix.outputs["logit"], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("kind", list(BCOutput))
def test_state_output_coupling_gradient_routing(kind):
    model = PopulationRetipath(bc_output=kind)
    trace = forward(model)
    h = {"tau_H", "delay_H"}
    bc = h | {"a_H", "tau_f_B", "tau_gap_B", "delay_B", "kappa_B"}
    output = "alpha_B" if kind is BCOutput.LEGACY_PRELU else "b_B"
    b_out = bc | {output}
    ac = b_out | {"tau_A", "delay_A"}
    expected = {"h_H": h, "H1_feedback": h | {"a_H"}, "s_B": bc,
                "delta_r_B": b_out, "a_A": ac, "o_A": ac | {"b_A"},
                "d_E": b_out | {"gamma_BR"}, "gE": b_out | {"gamma_BR"},
                "d_I": ac | {"b_A", "gamma_AR"}, "gI": ac | {"b_A", "gamma_AR"},
                "logit": ac | {"b_A", "gamma_AR", "gamma_BR", "bias"}}
    named = list(model.named_parameters())
    for port, allowed in expected.items():
        value = trace.observation(port)
        weight = torch.linspace(.7, 1.3, value.numel(), dtype=value.dtype).reshape(value.shape)
        grads = torch.autograd.grad((value * weight).square().mean(), [p for _, p in named],
                                    retain_graph=True, allow_unused=True)
        reached = set()
        for (name, _), grad in zip(named, grads):
            if grad is not None and grad.abs().max() > 1e-14:
                reached.add(name.split(".")[1])
            if name.split(".")[1] not in allowed:
                assert grad is None or grad.count_nonzero() == 0, (port, name)
        assert reached == allowed, (port, reached, allowed)
    groups = model.parameter_groups()
    grouped_ids = [id(p) for parameters in groups.values() for p in parameters]
    assert len(grouped_ids) == len(set(grouped_ids)) == len(named)
    assert set(grouped_ids) == {id(p) for _, p in named}
    assert model.fields["a_H"].group == model.fields["gamma_BR"].group == model.fields["gamma_AR"].group == "coupling"
    assert model.parameter_registry()["gamma_BA"]["trainable_count"] == 0
    assert not model.gamma_BA.requires_grad
    assert ("b_B" in model.fields) != ("alpha_B" in model.fields)


@pytest.mark.parametrize("kind", list(BCOutput))
def test_output_and_coupling_changes_do_not_rewrite_upstream_states(kind):
    model = PopulationRetipath(bc_output=kind)
    before = forward(model)
    output = "alpha_B" if kind is BCOutput.LEGACY_PRELU else "b_B"
    with torch.no_grad():
        model.fields[output].center.add_(.2)
    after = forward(model)
    torch.testing.assert_close(before.states["s_B"], after.states["s_B"], atol=0, rtol=0)
    assert not torch.equal(before.outputs["delta_r_B"], after.outputs["delta_r_B"])
    with torch.no_grad():
        model.fields["a_H"].center.add_(.2)
    h_changed = forward(model)
    torch.testing.assert_close(after.states["h_H"], h_changed.states["h_H"], atol=0, rtol=0)
    assert not torch.equal(after.outputs["H1_feedback"], h_changed.outputs["H1_feedback"])
    with torch.no_grad():
        model.fields["gamma_BR"].center.add_(.2)
    direct_changed = forward(model)
    for name in ("s_B", "a_A"):
        torch.testing.assert_close(h_changed.states[name], direct_changed.states[name], atol=0, rtol=0)
    torch.testing.assert_close(h_changed.outputs["delta_r_B"], direct_changed.outputs["delta_r_B"], atol=0, rtol=0)
    assert not torch.equal(h_changed.outputs["d_E"], direct_changed.outputs["d_E"])


@pytest.mark.parametrize("kind", list(BCOutput))
def test_physical_grid_consistency_and_incomplete_coverage(kind):
    model = PopulationRetipath(bc_output=kind)
    coarse, e1 = fixture(32)
    fine, e2 = fixture(64)
    t1, t2 = forward(model, coarse, e1), forward(model, fine, e2)
    for name in t1.states:
        torch.testing.assert_close(t1.states[name], t2.states[name], atol=2e-12, rtol=2e-12)
    for name in t1.outputs:
        torch.testing.assert_close(t1.outputs[name], t2.outputs[name], atol=2e-12, rtol=2e-12)
    smaller_bounds = coarse.pixel_bounds * .2
    cropped = replace(coarse, pixel_bounds=smaller_bounds, pixel_area=coarse.pixel_area * .04)
    for intervention in Intervention:
        with pytest.raises(ValueError, match="fully cover"):
            forward(model, cropped, e1, intervention=intervention)
    support = model.pathway_support()
    assert support["h_H"].shape == (25, 25) and support["RGC"].shape == (2, 25)
    assert support["RGC"].all()


@pytest.mark.parametrize("kind", list(BCOutput))
def test_on_off_local_broad_and_signed_drives(kind):
    model = PopulationRetipath(bc_output=kind)
    stimulus, events = fixture(40)
    xy = stimulus.pixel_bounds.mean(-1)
    patch = ((xy[:, 0] > -.35) & (xy[:, 0] < -.25)
             & (xy[:, 1] > -.35) & (xy[:, 1] < -.25)).to(xy.dtype)
    stimulus = replace(stimulus, values=.6 * patch.expand_as(stimulus.values))
    trace = forward(model, stimulus, events, intervention=Intervention.BLOCK_H1_FEEDBACK)
    torch.testing.assert_close(trace.inputs["u_B"][..., :25], -trace.inputs["u_B"][..., 25:])
    assert trace.inputs["u_A"][..., 4].abs().max() == 0
    assert trace.inputs["u_A"][..., 22].max() > 0
    assert trace.inputs["u_A"][..., 31].min() < 0
    assert model.pi_BR[0, :, 25:].count_nonzero() == 0
    assert model.pi_BR[1, :, :25].count_nonzero() == 0
    for f in range(4):
        wrong = slice(25, 50) if f % 2 == 0 else slice(0, 25)
        assert model.pi_BA[f * 9:(f + 1) * 9, wrong].count_nonzero() == 0
        torch.testing.assert_close(model.pi_AR[..., f * 9:(f + 1) * 9].sum(-1), torch.ones(2, 2, dtype=xy.dtype))
    torch.testing.assert_close(model.pi_BR.sum(-1), torch.ones(2, 2, dtype=xy.dtype))
    assert model.same_polarity_AR[0, :9].all() and not model.same_polarity_AR[0, 9:18].any()
    assert model.same_polarity_AR[1, 9:18].all()
    assert trace.outputs["d_E"][..., 1, :].min() < 0
    assert trace.outputs["gE"][..., 1, :].min() < 1
    assert trace.outputs["c_I"][..., 27:].min() < 0
    gamma = model.physical_parameters()["gamma_AR"][..., model.ac_family]
    expected_i = (trace.outputs["delta_o_A"][:, :, None, None, :] * gamma * model.pi_AR).sum(-1)
    torch.testing.assert_close(trace.outputs["d_I"], expected_i)


@pytest.mark.parametrize("kind", list(BCOutput))
@pytest.mark.parametrize("intervention", list(Intervention)[1:])
def test_interventions_preserve_upstream_and_recompute_membrane(kind, intervention):
    model = PopulationRetipath(bc_output=kind)
    stimulus, events = fixture()
    normal = forward(model, stimulus, events)
    blocked = forward(model, stimulus, events, intervention=intervention)
    torch.testing.assert_close(normal.states["history"], blocked.states["history"], atol=0, rtol=0)
    torch.testing.assert_close(normal.states["h_H"], blocked.states["h_H"], atol=0, rtol=0)
    if intervention is Intervention.BLOCK_H1_FEEDBACK:
        assert blocked.outputs["H1_feedback"].count_nonzero() == 0
        torch.testing.assert_close(blocked.inputs["q_modulated"], blocked.inputs["q"])
        assert not torch.equal(normal.states["s_B"], blocked.states["s_B"])
    else:
        for name in ("v_B", "w_B", "s_B", "a_A"):
            torch.testing.assert_close(normal.states[name], blocked.states[name], atol=0, rtol=0)
        for name in ("H1_feedback", "delta_r_B", "o_A", "delta_o_A"):
            torch.testing.assert_close(normal.outputs[name], blocked.outputs[name], atol=0, rtol=0)
        drive, contribution, tonic, preserved = (
            ("d_E", "c_E", "gE", "gI") if intervention is Intervention.BLOCK_DIRECT_BC_DRIVE
            else ("d_I", "c_I", "gI", "gE"))
        assert blocked.outputs[drive].count_nonzero() == 0
        assert blocked.outputs[contribution].count_nonzero() == 0
        torch.testing.assert_close(blocked.outputs[tonic], torch.ones_like(blocked.outputs[tonic]))
        torch.testing.assert_close(normal.outputs[preserved], blocked.outputs[preserved], atol=0, rtol=0)
    ge, gi = blocked.outputs["gE"], blocked.outputs["gI"]
    assert ge.min() > 0 and gi.min() > 0
    voltage = torch.full_like(ge[:, 0], 2 / 9)
    adaptation = torch.zeros_like(voltage.mean(-1))
    expected_v, expected_a = [], []
    decay_a = math.exp(-DT_MS / 120)
    for t in range(ge.shape[1]):
        total = 1 + ge[:, t] + gi[:, t]
        equilibrium = (ge[:, t] - gi[:, t] / 3) / total
        voltage = equilibrium + (voltage - equilibrium) * torch.exp(-DT_MS * total / 60)
        adaptation = decay_a * adaptation + (1 - decay_a) * (voltage.mean(-1) - 2 / 9)
        expected_v.append(voltage)
        expected_a.append(adaptation)
    v, a = torch.stack(expected_v, 1), torch.stack(expected_a, 1)
    torch.testing.assert_close(blocked.states["V"], v, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(blocked.states["adaptation"], a, atol=2e-12, rtol=2e-12)
    expected_logit = 8 * (v.mean(-1) - 2 / 9) - .5 * a - blocked.states["history"] + model.physical_parameters()["bias"]
    torch.testing.assert_close(blocked.outputs["logit"], expected_logit)
    torch.testing.assert_close(blocked.outputs["probability"], expected_logit.sigmoid())
    assert not torch.equal(normal.states["V"], blocked.states["V"])
    assert not torch.equal(normal.outputs["logit"], blocked.outputs["logit"])


@pytest.mark.parametrize("kind", list(BCOutput))
def test_partial_pooling_individuality_and_mandatory_objective_term(kind):
    model = PopulationRetipath(bc_output=kind)
    with torch.no_grad():
        for name in ("tau_H", "a_H", "tau_f_B"):
            model.fields[name].contrast[0] = .15
    p = model.physical_parameters()
    for name in ("tau_H", "a_H", "tau_f_B"):
        field = model.fields[name]
        assert p[name][0] != p[name][1]
        for family in field.family.unique():
            torch.testing.assert_close(field.deviations()[field.family == family].sum(), torch.tensor(0., dtype=model.G.dtype))
    assert (p["tau_s_B"] > p["tau_f_B"]).all()
    for name in ("tau_A", "delay_A", "b_A"):
        assert model.fields[name].contrast is None
        assert model.fields[name].center.numel() == 4
        for family in range(4):
            values = p[name][model.ac_family == family]
            assert torch.equal(values, values[0].expand_as(values))
    assert model.fields["a_H"].sd <= model.fields["tau_H"].sd / 3 + 1e-15
    assert model.fields["tau_f_B"].sd == 2 * model.fields["tau_H"].sd
    assert model.fields["a_H"].penalty() > model.fields["tau_H"].penalty()
    penalty = model.hierarchy_penalty()
    total = model.regularized_objective(penalty.new_tensor(2.))
    torch.testing.assert_close(total, 2 + penalty)
    grad, = torch.autograd.grad(total, model.fields["a_H"].contrast)
    assert grad[0] > 0
    assert not any("gamma_BA" in name for name, _ in model.named_parameters())
    before_support = model.pathway_support()
    with torch.no_grad():
        model.fields["gamma_AR"].center.fill_(-100.)
    after_support = model.pathway_support()
    assert all(torch.equal(value, after_support[name]) for name, value in before_support.items())


@pytest.mark.parametrize("kind", list(BCOutput))
@pytest.mark.parametrize("intervention,disconnected", [
    (Intervention.BLOCK_H1_FEEDBACK, {"tau_H", "delay_H", "a_H"}),
    (Intervention.BLOCK_DIRECT_BC_DRIVE, {"gamma_BR"}),
    (Intervention.BLOCK_AC_POSTSYNAPTIC_DRIVE, {"tau_A", "delay_A", "b_A", "gamma_AR"}),
])
def test_intervention_gradient_disconnection(kind, intervention, disconnected):
    model = PopulationRetipath(bc_output=kind)
    trace = forward(model, intervention=intervention)
    parameters = list(model.named_parameters())
    grads = torch.autograd.grad(trace.outputs["logit"].square().mean(),
                                [p for _, p in parameters], allow_unused=True)
    reached = set()
    for (name, _), grad in zip(parameters, grads):
        field = name.split(".")[1]
        if field in disconnected:
            assert grad is None or grad.count_nonzero() == 0
        elif grad is not None and grad.abs().max() > 1e-14:
            reached.add(field)
    assert reached == set(model.fields) - disconnected


def test_input_contract_rejections_and_no_dataset_selector():
    model = PopulationRetipath()
    stimulus, events = fixture()
    missing = stimulus.input_valid.clone()
    missing[0, 0, 0] = False
    invalid = [replace(stimulus, input_valid=missing),
               replace(stimulus, pixel_area=stimulus.pixel_area * 2),
               replace(stimulus, time_ms=stimulus.time_ms + DT_MS),
               replace(stimulus, values=stimulus.values * float("nan"))]
    for item in invalid:
        with pytest.raises(ValueError):
            forward(model, item, events)
    for kwargs in ({"reset": "carry"}, {"intervention": "UNKNOWN"}):
        with pytest.raises(ValueError):
            forward(model, stimulus, events, **kwargs)
    with pytest.raises(ValueError):
        forward(model, stimulus, events + .5)
    with pytest.raises(ValueError):
        PopulationRetipath(bc_output="AUTO_BY_DATASET")
    assert PopulationRetipath().bc_output is BCOutput.LEGACY_PRELU
    assert "dataset_id" not in inspect.signature(model.forward).parameters
    assert "session_id" not in inspect.signature(model.forward).parameters
    with pytest.raises(TypeError):
        forward(model, stimulus, events, dataset_id="choose_mechanism")
