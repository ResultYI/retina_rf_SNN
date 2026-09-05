#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0"]
# ///
# How to run: imported by analyze.py in this artifact directory.
from __future__ import annotations

from typing import Final

import torch

from models.mechanistic_retina.causal_contract import CANONICAL_CAUSAL_CONTRACT
from models.mechanistic_retina.contracts import PathwayClamp
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina

type CheckValues = dict[str, bool | float | int]
type ReplayTensors = dict[str, dict[str, torch.Tensor]]
CLAMPS: Final = {
    "H1_off": frozenset({PathwayClamp.H1}),
    "direct_BC_off": frozenset({PathwayClamp.DIRECT_BC_SUSTAINED, PathwayClamp.DIRECT_BC_TRANSIENT}),
    "AC_off": frozenset({PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}),
}
FIELDS: Final = (
    "h1_state", "h1_surround_contribution", "bc_direct_presynaptic", "bc_broad_presynaptic",
    "bc_sustained_current", "bc_transient_current", "amacrine_local_state",
    "amacrine_transient_state", "amacrine_local_current", "amacrine_transient_current",
    "logits", "spike_probability",
)


def replay(
    model: MechanisticGraphTemporalRetina, cones: torch.Tensor, history: torch.Tensor,
) -> tuple[CheckValues, ReplayTensors]:
    """Replay only the requested causal boundaries on unchanged model tensors."""
    before = {name: value.clone() for name, value in model.state_dict().items()}
    encoder_ids: list[int] = []

    def capture(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        encoder_ids.append(id(module.get_parameter("raw_weights")))

    handle = model.bipolar.register_forward_pre_hook(capture)
    try:
        normal = model.forward_sequence(cones, observed_counts=history)
    finally:
        handle.remove()
    ac = normal.amacrine_local_current + normal.amacrine_transient_current
    dependence = torch.autograd.grad(ac.sum(), normal.bc_broad_presynaptic)[0]
    gates = model.gates.values(frozenset())
    with torch.no_grad():
        h1 = model.h1(cones, amplitude=gates.h1).modulated_cones
        features = model.shared_subunits(model.feature_bank(h1))
        weights = model.bipolar.positive_weights()[None, None]
        reconstructed = [
            (features[:, :, :, start:start + 2] * weights).sum((-1, -2))
            for start in (0, 2)
        ]
        view_errors = [float((actual - expected).abs().max()) for actual, expected in zip(
            (normal.bc_direct_presynaptic, normal.bc_broad_presynaptic), reconstructed, strict=True
        )]
        bank = model.feature_bank
        supports = (bank.bc_support, bank.bc_support, bank.ac_support, bank.ac_support)
        spatial = torch.stack([bank.spatial_basis * mask[:, None] for mask in supports], dim=1)
        spatial = spatial / spatial.sum(-1, keepdim=True)
        expected_basis = torch.einsum(
            "n,prl,npsc->npsrlc", bank.polarity_sign, bank.temporal_basis.repeat(2, 1, 1), spatial
        )
        basis_error = float((expected_basis - bank.basis_kernels()).abs().max())

    def detach_broad(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return (inputs[0].detach(),)

    stimulus = cones.clone().requires_grad_(True)
    handle = model.amacrine.register_forward_pre_hook(detach_broad)
    try:
        detached = model.forward_sequence(stimulus, observed_counts=history)
        detached_ac = detached.amacrine_local_current + detached.amacrine_transient_current
        bypass = torch.autograd.grad(detached_ac.sum(), stimulus, allow_unused=True)[0]
    finally:
        handle.remove()
    outputs = {"normal": normal}
    with torch.no_grad():
        for name, clamp in CLAMPS.items():
            outputs[name] = model.forward_sequence(cones, observed_counts=history, clamps=clamp)
    direct, ac_off, h1_off = (outputs[name] for name in ("direct_BC_off", "AC_off", "H1_off"))
    direct_unchanged = ("bc_broad_presynaptic", "amacrine_local_state", "amacrine_transient_state",
                        "amacrine_local_current", "amacrine_transient_current")
    ac_unchanged = ("h1_state", "h1_surround_contribution", "bc_direct_presynaptic", "bc_broad_presynaptic")
    propagated = ("bc_direct_presynaptic", "bc_broad_presynaptic", "amacrine_local_state",
                  "amacrine_transient_state", "amacrine_local_current", "amacrine_transient_current")
    checks: CheckValues = {
        "identity_current": model.config.causal_contract == CANONICAL_CAUSAL_CONTRACT,
        "shared_encoder_parameter_identity": encoder_ids == [id(model.bipolar.raw_weights)] * 2,
        "views_reconstruct_from_same_weights": max(view_errors) <= 1e-7,
        "views_differ_only_by_support": basis_error <= 1e-7,
        "one_BC_temporal_encoder": bank.raw_tau.shape == (2, 3) and bank.raw_delay.shape == (2,),
        "AC_has_only_downstream_parameters": set(dict(model.amacrine.named_parameters())) == {"raw_tau", "raw_delay"},
        "AC_depends_on_BC_broad": bool(torch.isfinite(dependence).all() and torch.count_nonzero(dependence)),
        "AC_without_BC_has_no_stimulus_gradient": bypass is None or bool(torch.count_nonzero(bypass) == 0),
        "direct_BC_off_preserves_BC_broad_and_AC": all(torch.equal(getattr(normal, n), getattr(direct, n)) for n in direct_unchanged),
        "AC_off_preserves_H1_and_BC_views": all(torch.equal(getattr(normal, n), getattr(ac_off, n)) for n in ac_unchanged),
        "H1_off_propagates_to_BC_and_AC": all(torch.count_nonzero(getattr(normal, n) - getattr(h1_off, n)) > 0 for n in propagated),
        "H1_exact_zero": torch.count_nonzero(h1_off.h1_surround_contribution).item() == 0,
        "direct_BC_exact_zero": all(torch.count_nonzero(getattr(direct, n)).item() == 0 for n in FIELDS[4:6]),
        "AC_exact_zero": all(torch.count_nonzero(getattr(ac_off, n)).item() == 0 for n in FIELDS[8:10]),
        "all_outputs_finite": all(bool(torch.isfinite(t).all()) for out in outputs.values() for t in out.tensors()),
        "state_dict_unchanged": all(torch.equal(before[n], v) for n, v in model.state_dict().items()),
    }
    checks["all_passed"] = all(checks.values())
    checks["shared_view_max_absolute_error"] = max(view_errors)
    checks["shared_basis_max_absolute_error"] = basis_error
    checks["AC_to_BC_broad_gradient_norm"] = float(dependence.norm())
    for name in propagated:
        checks[f"H1_off_{name}_change_norm"] = float((getattr(normal, name) - getattr(h1_off, name)).detach().norm())
    tensors = {condition: {name: getattr(out, name).detach() for name in FIELDS} for condition, out in outputs.items()}
    return checks, tensors


