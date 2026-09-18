import json

import pytest
import torch

from evaluation.mechanistic_retina.mechanism_observation import (
    CANONICAL_TRACE_VARIABLE_SPECS, Intervention, InterventionSpec, observe_mechanism,
)
from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig
from models.mechanistic_retina.retipath import RetiPath
from models.mechanistic_retina.retipath_canonical_gain import CanonicalGainRetiPath, REMOVED_GAIN_KEYS, canonicalize_gain_state


CHECKED_VARIABLES = ("logit", "probability", "gE", "gI", "V1", "V2", "membrane_readout")
ATOL, RTOL = 2e-6, 2e-5


def compare_traces(old, new):
    errors = {}
    for name in CHECKED_VARIABLES:
        a, b = old.tensors[name], new.tensors[name]
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(b, a, atol=ATOL, rtol=RTOL)
        errors[name] = float((b - a).abs().max())
    return errors


@pytest.fixture
def migrated_pair():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(915)
        positions = torch.cartesian_prod(torch.linspace(-0.3, 0.3, 7), torch.linspace(-0.3, 0.3, 7))
        config = MechanisticRetinaConfig(architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE,
                                        cell_specific_gains=True, dt_ms=1000 / 150)
        args = (config, positions, torch.zeros(1, 2), ("parasol",), ("OFF",))
        rms = {"rms_e": torch.tensor([[0.8, 1.2]]), "rms_i": torch.tensor([[0.6, 1.1]])}
        old, new = RetiPath(*args, **rms), CanonicalGainRetiPath(*args, **rms)
        with torch.no_grad():
            old.bipolar.raw_weights.copy_(torch.randn_like(old.bipolar.raw_weights))
            old.gates.ac_local.fill_(0.7)
            old.gates.ac_transient.fill_(-0.4)
            old.cell_gains.log_bc.fill_(0.8)
            old.cell_gains.log_ac.fill_(-0.3)
            old.spatial_ei.log_a_e.fill_(-0.2)
            old.spatial_ei.log_a_i.fill_(0.6)
            old.theta.fill_(-0.4)
        new.load_state_dict(canonicalize_gain_state(old.state_dict()), strict=True)
        x = torch.randn(2, 64, 49)
        y = (torch.rand(2, 64, 1) > 0.7).float()
    return old, new, x, y


def test_all_interventions_preserve_prediction(migrated_pair):
    old, new, x, y = migrated_pair
    with torch.no_grad():
        for kind in Intervention:
            spec = InterventionSpec(kind)
            a = observe_mechanism(old, x, observed_counts=y, intervention=spec)
            b = observe_mechanism(new, x, observed_counts=y, intervention=spec)
            compare_traces(a, b)
        official_new = new.forward_sequence(x, observed_counts=y)
        assert torch.equal(b.tensors["h1_state"], a.tensors["h1_state"])
        normal = observe_mechanism(new, x, observed_counts=y)
        assert torch.equal(normal.logits, official_new.logits)
        assert torch.equal(normal.probability, official_new.spike_probability)


def test_coordinates_and_trace_contract(migrated_pair):
    old, new, x, y = migrated_pair
    assert sum(p.numel() for p in old.parameters() if p.requires_grad) == 37
    assert sum(p.numel() for p in new.parameters() if p.requires_grad) == 33
    assert not REMOVED_GAIN_KEYS.intersection(dict(new.named_parameters()))
    trace = observe_mechanism(new, x, observed_counts=y)
    assert trace.schema_version == 2
    assert set(trace.tensors) == set(CANONICAL_TRACE_VARIABLE_SPECS)
    json.dumps(trace.metadata())
    t = trace.tensors
    for side in ("E", "I"):
        weights = t[f"w_{side}"]
        torch.testing.assert_close(weights.sum(-1), torch.ones(1), atol=torch.finfo(weights.dtype).eps, rtol=0)
        assert (weights > 0).all()
        assert CANONICAL_TRACE_VARIABLE_SPECS[f"w_{side}"].status == "EFFECTIVE"
        assert CANONICAL_TRACE_VARIABLE_SPECS[f"G_{side}"].status == "EFFECTIVE"
        relative = (weights[None, None, :, None, :] * t[f"{side}_components"]).sum(-1)
        torch.testing.assert_close(relative, t[f"relative_drive_{side}"], atol=1e-7, rtol=1e-6)
        assert torch.equal(t[f"effective_drive_{side}"], t[f"G_{side}"][None, None, :, None] * t[f"relative_drive_{side}"])
        assert "non-canonical scale-dependent" in CANONICAL_TRACE_VARIABLE_SPECS[f"u{side}"].normalization_unit
    expected = trace.logits.detach().clone()
    with torch.no_grad():
        new.canonical_gains.legacy_a_E.mul_(7)
        new.canonical_gains.legacy_a_I.mul_(3)
    changed_alias = observe_mechanism(new, x, observed_counts=y)
    assert torch.equal(expected, changed_alias.logits)
    assert not torch.equal(t["uE"], changed_alias.tensors["uE"])
    coordinates = tuple(new.canonical_gains.parameters())
    gradients = torch.autograd.grad(changed_alias.logits.sum(), coordinates)
    assert len(gradients) == 4
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)


def test_old_gauge_orbits_collapse(migrated_pair):
    old, new, x, y = migrated_pair
    initial = canonicalize_gain_state(old.state_dict())
    shifted = {key: value.clone() for key, value in old.state_dict().items()}
    shifted["cell_gains.log_bc"] += 0.7
    shifted["spatial_ei.log_a_e"] -= 0.7
    shifted["cell_gains.log_ac"] -= 0.4
    shifted["spatial_ei.log_a_i"] += 0.4
    shifted["bipolar.raw_weights"] += 0.8
    shifted["gates.ac_local"] += 0.6
    shifted["gates.ac_transient"] += 0.6
    collapsed = canonicalize_gain_state(shifted)
    for name in dict(new.named_parameters()):
        torch.testing.assert_close(collapsed[name], initial[name], atol=2e-6, rtol=2e-6)
    with pytest.raises(ValueError):
        canonicalize_gain_state(initial)
