from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from models.mechanistic_retina.amacrine_pathways import AmacrinePathways
from models.mechanistic_retina.bipolar_subunits import BipolarSubunits
from models.mechanistic_retina.cell_specific_gains import CellSpecificPathwayGains
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MechanisticRetinaConfig,
    PathwayClamp,
)
from models.mechanistic_retina.model import build_mechanistic_retina
from models.mechanistic_retina.pathway_gates import PathwayGates
from training.mechanistic_retina.optimizer import phase1_parameters


_CELL_TYPES = ("midget", "midget", "parasol", "parasol")
_POLARITIES = ("ON", "OFF", "ON", "OFF")


def _config() -> MechanisticRetinaConfig:
    return MechanisticRetinaConfig(
        architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE
    )


def _raw_for_scaled_softplus(raw: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.log(torch.expm1(F.softplus(raw) * scale))


def test_h1_has_one_bounded_effective_amplitude_parameter() -> None:
    # Given: a fresh Canonical V1 model using the evidence-contract defaults.
    model = build_mechanistic_retina(
        _config(),
        torch.tensor(
            ((0.0, 0.0), (0.04, 0.0), (0.08, 0.0), (0.13, 0.0), (0.18, 0.0))
        ),
        torch.tensor(((0.0, 0.0),)),
        ("midget",),
        ("ON",),
    )

    # When: the H1 strength and trainable parameter contract are inspected.
    amplitude = model.gates.h1
    parameter_names = dict(model.named_parameters())
    buffer_names = dict(model.named_buffers())
    optimizer_parameters = phase1_parameters(model)

    # Then: one bounded parameter owns the complete H1 effective amplitude.
    torch.testing.assert_close(amplitude, amplitude.new_tensor(0.01))
    torch.testing.assert_close(
        model.gates.h1_amplitude_bounds,
        amplitude.new_tensor((0.0, 0.2)),
    )
    assert 0.0 < float(amplitude) < 0.2
    assert (
        parameter_names["gates.raw_h1_amplitude"]
        is model.gates.raw_h1_amplitude
    )
    assert any(
        parameter is model.gates.raw_h1_amplitude
        for parameter in optimizer_parameters
    )
    assert "h1.gain" not in buffer_names


def test_h1_amplitude_is_bounded_before_exact_zero_structural_clamp() -> None:
    # Given: an H1 amplitude with the canonical evidence bounds.
    gates = PathwayGates(
        0.5,
        group_index=torch.tensor((0,)),
        trainable=True,
    )

    # When: its raw value spans both extremes and H1 is structurally clamped.
    with torch.no_grad():
        gates.raw_h1_amplitude.fill_(100.0)
    upper = gates.h1.detach().clone()
    with torch.no_grad():
        gates.raw_h1_amplitude.fill_(-100.0)
    lower = gates.h1.detach().clone()
    clamped = gates.values(frozenset({PathwayClamp.H1})).h1

    # Then: the learned value stays bounded and the post-transform mask is exact zero.
    lower_bound, upper_bound = gates.h1_amplitude_bounds
    assert float(lower_bound) <= float(lower) <= float(upper_bound)
    assert float(lower_bound) <= float(upper) <= float(upper_bound)
    assert torch.count_nonzero(clamped) == 0


def test_effective_weights_and_normal_ac_gates_are_unit_normalized() -> None:
    # Given: nonuniform shared BC weights and downstream AC group gates.
    bipolar = BipolarSubunits(_CELL_TYPES, _POLARITIES, shared=True)
    amacrine = AmacrinePathways(_config(), _CELL_TYPES, _POLARITIES)
    gates = PathwayGates(
        0.5, group_index=amacrine.group_index, trainable=True
    )
    with torch.no_grad():
        bipolar.raw_weights.add_(
            torch.linspace(-0.4, 0.4, bipolar.raw_weights.numel()).reshape_as(
                bipolar.raw_weights
            )
        )
        gates.ac_local.fill_(1.2)
        gates.ac_transient.fill_(-0.7)

    # When: effective strengths are evaluated in a normal forward contract.
    bc_weights = bipolar.positive_weights()
    gate_values = gates.values(frozenset())
    ac_gates = torch.stack(
        (gate_values.ac_local, gate_values.ac_transient), dim=1
    )

    # Then: the shared encoder has one simplex per group; downstream AC has
    # only a nonnegative group-specific two-way mixture, not encoder weights.
    torch.testing.assert_close(
        bc_weights.sum(dim=(1, 2, 3)), torch.ones(len(_CELL_TYPES))
    )
    torch.testing.assert_close(ac_gates.sum(dim=1), torch.ones(len(_CELL_TYPES)))
    assert bool((bc_weights >= 0).all())
    assert bool((ac_gates >= 0).all())
    assert set(dict(amacrine.named_parameters())) == {"raw_tau", "raw_delay"}


def test_weight_normalization_breaks_former_inverse_gain_scale_symmetry() -> None:
    # Given: the old exact rescaling of positive pathway weights and inverse gains.
    bipolar = BipolarSubunits(_CELL_TYPES, _POLARITIES, shared=True)
    gains = CellSpecificPathwayGains(len(_CELL_TYPES))
    bc_before = gains.bc[:, None, None, None] * bipolar.positive_weights()
    broad_before = gains.ac[:, None, None, None] * bipolar.positive_weights()
    scale = 2.0

    # When: every pre-normalization positive weight is scaled while the aggregate
    # amplitude gain is divided by the same factor.
    with torch.no_grad():
        bipolar.raw_weights.copy_(
            _raw_for_scaled_softplus(bipolar.raw_weights, scale)
        )
        gains.log_bc.sub_(math.log(scale))
        gains.log_ac.sub_(math.log(scale))
    bc_after = gains.bc[:, None, None, None] * bipolar.positive_weights()
    broad_after = gains.ac[:, None, None, None] * bipolar.positive_weights()

    # Then: normalized pathway weights cannot absorb inverse aggregate gains.
    assert not torch.allclose(bc_after, bc_before)
    assert not torch.allclose(broad_after, broad_before)


def test_ac_gate_simplex_breaks_former_inverse_gain_scale_symmetry() -> None:
    # Given: equal normal AC gates and an aggregate AC gain.
    group_index = torch.tensor((0, 0, 1))
    gates = PathwayGates(0.5, group_index=group_index, trainable=True)
    gains = CellSpecificPathwayGains(3)
    normal = gates.values(frozenset())
    gates_before = torch.stack((normal.ac_local, normal.ac_transient), dim=1)
    strength_before = gains.ac[:, None] * gates_before
    scale = 1.8

    # When: both learned gate parameters are scaled while g_AC is divided.
    with torch.no_grad():
        gates.ac_local.mul_(scale)
        gates.ac_transient.mul_(scale)
        gains.log_ac.sub_(math.log(scale))
    scaled = gates.values(frozenset())
    gates_after = torch.stack((scaled.ac_local, scaled.ac_transient), dim=1)
    strength_after = gains.ac[:, None] * gates_after

    # Then: the gate simplex fixes total AC mixture strength at g_AC.
    assert not torch.allclose(strength_after, strength_before)
    torch.testing.assert_close(gates_after.sum(dim=1), torch.ones(3))


def test_ac_structural_clamp_masks_normalized_gates_without_renormalizing() -> None:
    # Given: unequal AC logits and the normalized normal group gates.
    amacrine = AmacrinePathways(_config(), _CELL_TYPES, _POLARITIES)
    gates = PathwayGates(
        0.5, group_index=amacrine.group_index, trainable=True
    )
    with torch.no_grad():
        gates.ac_local.fill_(1.0)
        gates.ac_transient.fill_(-1.0)
    normal = gates.values(frozenset())

    # When: local alone and then both AC pathways are structurally clamped.
    local_off = gates.values(frozenset({PathwayClamp.AMACRINE_LOCAL}))
    both_off = gates.values(
        frozenset(
            {
                PathwayClamp.AMACRINE_LOCAL,
                PathwayClamp.AMACRINE_TRANSIENT,
            }
        )
    )
    presynaptic = torch.rand(2, 7, len(_CELL_TYPES), 2)
    output = amacrine(
        presynaptic,
        local_gate=both_off.ac_local,
        transient_gate=both_off.ac_transient,
    )

    # Then: masking is exact zero and the surviving gate keeps its pre-clamp value.
    assert torch.count_nonzero(local_off.ac_local) == 0
    torch.testing.assert_close(local_off.ac_transient, normal.ac_transient)
    assert bool(((local_off.ac_local + local_off.ac_transient) < 1.0).all())
    assert torch.count_nonzero(both_off.ac_local) == 0
    assert torch.count_nonzero(both_off.ac_transient) == 0
    assert torch.count_nonzero(output.local_current) == 0
    assert torch.count_nonzero(output.transient_current) == 0


def test_downstream_ac_mixture_scales_shared_bc_states_with_inhibitory_sign() -> None:
    # Given: broader BC signals and nonuniform group-specific AC mixtures.
    torch.manual_seed(17)
    amacrine = AmacrinePathways(_config(), _CELL_TYPES, _POLARITIES)
    gates = PathwayGates(
        0.5, group_index=amacrine.group_index, trainable=True
    )
    gains = CellSpecificPathwayGains(len(_CELL_TYPES))
    presynaptic = torch.rand(2, 9, len(_CELL_TYPES), 2)
    with torch.no_grad():
        gates.ac_local.copy_(torch.tensor((0.7, -0.1, 1.0, 0.2)))
        gates.ac_transient.copy_(torch.tensor((-0.3, 0.4, -0.2, 0.9)))
        gains.log_ac.copy_(torch.log(torch.tensor((0.7, 1.1, 1.4, 0.9))))

    # When: downstream AC dynamics and the final inhibitory gain are evaluated.
    gate_values = gates.values(frozenset())
    output = amacrine(
        presynaptic,
        local_gate=gate_values.ac_local,
        transient_gate=gate_values.ac_transient,
    )
    new_current = torch.stack(
        (output.local_current, output.transient_current), dim=-1
    ) * gains.ac[None, None, :, None]

    states = amacrine.presynaptic_states(presynaptic)
    mixture = torch.stack((gate_values.ac_local, gate_values.ac_transient), dim=-1)

    # Then: only the corresponding BC state, AC mixture and g_AC set each current.
    expected = -states * mixture[None, None] * gains.ac[None, None, :, None]
    torch.testing.assert_close(new_current, expected)
    assert bool((new_current <= 0).all())
