from __future__ import annotations

import pytest
import torch

from evaluation.mechanistic_retina.rf_effective import effective_rf
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MechanisticRetinaConfig,
    PathwayClamp,
)
from models.mechanistic_retina.model import (
    MechanisticGraphTemporalRetina,
    build_mechanistic_retina,
)


def _model() -> MechanisticGraphTemporalRetina:
    return build_mechanistic_retina(
        MechanisticRetinaConfig(
            architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE,
            cell_specific_gains=True,
        ),
        torch.tensor(((0.0, 0.0), (0.04, 0.0), (0.08, 0.0), (0.12, 0.0))),
        torch.zeros(2, 2),
        ("midget", "parasol"),
        ("ON", "OFF"),
    )


def _cones() -> torch.Tensor:
    return torch.linspace(-0.3, 0.5, 24 * 4).reshape(1, 24, 4)


def test_ac_has_nonzero_graph_dependency_on_broad_bc_and_shared_weights() -> None:
    model = _model()
    output = model.forward_sequence(_cones(), observed_counts=torch.zeros(1, 24, 2))
    ac = output.amacrine_local_current + output.amacrine_transient_current

    gradients = torch.autograd.grad(
        ac[:, -1].square().sum(),
        (output.bc_broad_presynaptic, model.bipolar.raw_weights),
    )

    assert output.bc_broad_presynaptic.shape == (1, 24, 2, 2)
    assert all(torch.isfinite(value).all() for value in gradients)
    assert all(torch.count_nonzero(value) > 0 for value in gradients)


def test_ac_has_no_independent_encoder_parameters_or_stimulus_bypass() -> None:
    model = _model()
    assert set(dict(model.amacrine.named_parameters())) == {"raw_tau", "raw_delay"}
    assert not hasattr(model.amacrine, "raw_weights")
    assert not hasattr(model.amacrine, "positive_weights")
    assert not hasattr(model.amacrine, "base_kernels")

    def zero_presynaptic(
        module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        return (torch.zeros_like(inputs[0]),)

    hook = model.amacrine.register_forward_pre_hook(zero_presynaptic)
    try:
        output = model.forward_sequence(_cones(), observed_counts=torch.zeros(1, 24, 2))
    finally:
        hook.remove()

    assert torch.count_nonzero(output.bc_broad_presynaptic) > 0
    assert torch.count_nonzero(output.amacrine_local_current) == 0
    assert torch.count_nonzero(output.amacrine_transient_current) == 0


def test_two_bc_views_use_identical_encoder_parameters_and_only_distinct_support() -> None:
    model = _model()
    cones = _cones()
    h1 = model.h1(cones, amplitude=model.gates.h1)
    features = model.shared_subunits(model.feature_bank(h1.modulated_cones))
    weights = model.bipolar.positive_weights()[None, None]

    output = model.forward_sequence(cones, observed_counts=torch.zeros(1, 24, 2))

    torch.testing.assert_close(output.bc_direct_presynaptic, (features[..., :2, :, :] * weights).sum((-1, -2)))
    torch.testing.assert_close(output.bc_broad_presynaptic, (features[..., 2:, :, :] * weights).sum((-1, -2)))
    bank = model.feature_bank
    assert bank.raw_tau.shape == (2, 3)
    assert bank.raw_delay.shape == (2,)
    assert bank.temporal_basis.shape == (2, 3, model.config.lag_steps)
    spatial = torch.stack(tuple(
        bank.spatial_basis * support[:, None]
        for support in (bank.bc_support, bank.bc_support, bank.ac_support, bank.ac_support)
    ), dim=1)
    spatial = spatial / spatial.sum(-1, keepdim=True)
    torch.testing.assert_close(bank.path_spatial_basis, spatial)
    temporal = bank.temporal_basis.repeat(2, 1, 1)
    expected = torch.einsum("n,prl,npsc->npsrlc", bank.polarity_sign, temporal, spatial)
    torch.testing.assert_close(bank.basis_kernels(), expected)
    assert not torch.equal(output.bc_direct_presynaptic, output.bc_broad_presynaptic)


def test_h1_off_propagates_to_both_bc_views_and_ac() -> None:
    model = _model()
    cones, history = _cones(), torch.zeros(1, 24, 2)
    normal = model.forward_sequence(cones, observed_counts=history)

    off = model.forward_sequence(cones, observed_counts=history, clamps=frozenset({PathwayClamp.H1}))

    assert torch.count_nonzero(off.h1_surround_contribution) == 0
    assert not torch.equal(normal.bc_direct_presynaptic, off.bc_direct_presynaptic)
    assert not torch.equal(normal.bc_broad_presynaptic, off.bc_broad_presynaptic)
    assert not torch.equal(normal.amacrine_local_state, off.amacrine_local_state)
    assert not torch.equal(normal.amacrine_transient_state, off.amacrine_transient_state)


def test_ac_off_changes_only_final_ac_contributions_before_rgc() -> None:
    model = _model()
    cones, history = _cones(), torch.zeros(1, 24, 2)
    normal = model.forward_sequence(cones, observed_counts=history)
    clamps = frozenset({PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT})

    off = model.forward_sequence(cones, observed_counts=history, clamps=clamps)

    for name in ("h1_state", "h1_surround_contribution", "bc_direct_presynaptic", "bc_broad_presynaptic", "bc_sustained_current", "bc_transient_current", "amacrine_local_state", "amacrine_transient_state"):
        assert torch.equal(getattr(normal, name), getattr(off, name))
    assert torch.count_nonzero(off.amacrine_local_current) == 0
    assert torch.count_nonzero(off.amacrine_transient_current) == 0


def test_direct_bc_off_preserves_broad_bc_drive_and_downstream_ac() -> None:
    model = _model()
    cones, history = _cones(), torch.zeros(1, 24, 2)
    normal = model.forward_sequence(cones, observed_counts=history)
    clamps = frozenset({PathwayClamp.DIRECT_BC_SUSTAINED, PathwayClamp.DIRECT_BC_TRANSIENT})

    off = model.forward_sequence(cones, observed_counts=history, clamps=clamps)

    for name in ("h1_state", "h1_surround_contribution", "bc_direct_presynaptic", "bc_broad_presynaptic", "amacrine_local_state", "amacrine_transient_state", "amacrine_local_current", "amacrine_transient_current"):
        assert torch.equal(getattr(normal, name), getattr(off, name))
    assert torch.count_nonzero(off.bc_sustained_current) == 0
    assert torch.count_nonzero(off.bc_transient_current) == 0
    assert torch.count_nonzero(off.amacrine_local_current) > 0
    assert torch.count_nonzero(off.amacrine_transient_current) > 0


def test_shared_bc_cascade_retains_stimulus_and_history_causality() -> None:
    model = _model()
    cones, history = _cones(), torch.zeros(1, 24, 2)
    normal = model.forward_sequence(cones, observed_counts=history)
    changed_cones, changed_history = cones.clone(), history.clone()
    changed_cones[:, 13:] += 10
    changed_history[:, 12:] = 1

    changed = model.forward_sequence(changed_cones, observed_counts=changed_history)

    assert torch.equal(normal.logits[:, :13], changed.logits[:, :13])
    assert torch.equal(normal.bc_broad_presynaptic[:, :13], changed.bc_broad_presynaptic[:, :13])


def test_cascade_global_and_pathway_rf_match_true_forward_derivatives() -> None:
    model = _model().double()
    cones = _cones()[:, -16:].double().requires_grad_()
    history = torch.zeros(1, 16, 2, dtype=torch.float64)
    direction = torch.cos(torch.arange(cones.numel(), dtype=torch.float64)).reshape_as(cones)

    rf = effective_rf(model, cones, history)
    epsilon = 1e-5
    plus = model.forward_sequence(cones + epsilon * direction, observed_counts=history)
    minus = model.forward_sequence(cones - epsilon * direction, observed_counts=history)
    finite_difference = (plus.logits[:, -1] - minus.logits[:, -1]) / (2 * epsilon)
    torch.testing.assert_close((rf * direction[:, None]).sum((-1, -2)), finite_difference, rtol=1e-5, atol=1e-9)
    output = model.forward_sequence(cones, observed_counts=history)
    currents = (output.bc_sustained_current, output.bc_transient_current, output.amacrine_local_current, output.amacrine_transient_current)
    for current, kernel in zip(currents, model.pathway_base_rfs(), strict=True):
        actual = torch.stack(tuple(torch.autograd.grad(current[0, -1, cell], cones, retain_graph=True)[0][0] for cell in range(2)))
        torch.testing.assert_close(kernel, actual, rtol=1e-5, atol=1e-9)


@pytest.mark.parametrize("strict", (True, False))
def test_checkpoint_without_new_causal_identity_is_explicitly_rejected(strict: bool) -> None:
    model = _model()
    state = model.state_dict()
    del state["_causal_contract_id"]

    with pytest.raises(RuntimeError, match="causal contract"):
        model.load_state_dict(state, strict=strict)


@pytest.mark.parametrize("strict", (True, False))
def test_checkpoint_with_wrong_causal_identity_is_explicitly_rejected(strict: bool) -> None:
    model = _model()
    state = model.state_dict()
    state["_causal_contract_id"] = torch.zeros_like(state["_causal_contract_id"])

    with pytest.raises(RuntimeError, match="causal contract"):
        model.load_state_dict(state, strict=strict)


@pytest.mark.parametrize("strict", (True, False))
@pytest.mark.parametrize("nested", (True, False))
@pytest.mark.parametrize("missing_identity", (True, False))
def test_old_checkpoint_rejection_precedes_parameter_mutation(
    strict: bool, nested: bool, missing_identity: bool,
) -> None:
    model = _model()
    container = torch.nn.ModuleDict({"retina": model}) if nested else model
    before = {name: value.clone() for name, value in container.state_dict().items()}
    incoming = {name: value.clone() for name, value in before.items()}
    for value in incoming.values():
        if torch.is_floating_point(value):
            value.add_(1.0)
    identity_key = "retina._causal_contract_id" if nested else "_causal_contract_id"
    if missing_identity:
        del incoming[identity_key]
    else:
        incoming[identity_key].zero_()

    with pytest.raises(RuntimeError, match="causal contract"):
        container.load_state_dict(incoming, strict=strict)

    assert all(torch.equal(before[name], value) for name, value in container.state_dict().items())


def test_old_causal_config_and_ambiguous_bc_clamp_are_rejected() -> None:
    with pytest.raises(ValueError, match="causal contract"):
        MechanisticRetinaConfig(causal_contract="independent-bc-ac-stimulus-encoders")
    for old in ("no-BC-sustained", "no-BC-transient"):
        with pytest.raises(ValueError):
            PathwayClamp(old)


def test_frozen_forward_is_finite_and_does_not_carry_state_between_calls() -> None:
    model = _model()
    cones, history = _cones(), torch.zeros(1, 24, 2)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    first = model.forward_sequence(cones, observed_counts=history)
    repeated = model.forward_sequence(cones, observed_counts=history)

    assert all(torch.isfinite(value).all() for value in first.tensors())
    assert all(torch.equal(a, b) for a, b in zip(first.tensors(), repeated.tensors(), strict=True))
    assert all(torch.equal(before[name], tensor) for name, tensor in model.state_dict().items())
