#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "matplotlib", "numpy"]
# ///
# How to run: imported by run.py in the frozen repository runtime.
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Final, TypeAlias, TypedDict

import torch

OUT: Final = Path(__file__).resolve().parent
ROOT: Final = OUT.parents[2]
SOURCE: Final = OUT.parent / "schottdorf_canonical_v1_shared_bc_development_22cell_20260830"
TEMPORAL: Final = OUT.parent / "schottdorf_canonical_v1_h1_ac_temporal_dissociation_20260830"
ILLUSION: Final = OUT.parent / "schottdorf_canonical_v1_overlapping_support_visual_illusions_20260830"
DIAGNOSTIC: Final = OUT.parent / "schottdorf_ln_r4_illusion_diagnostics_20260830"
ORIGINAL: Final = OUT.parent / "schottdorf_r4_dev_visual_illusions_20260830"
for folder in (ROOT, TEMPORAL, ORIGINAL):
    sys.path.insert(0, str(folder))

from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig, PathwayClamp
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina, build_mechanistic_retina

CLAMPS: Final = {
    "normal": frozenset(), "H1_off": frozenset({PathwayClamp.H1}),
    "direct_BC_off": frozenset({PathwayClamp.DIRECT_BC_SUSTAINED, PathwayClamp.DIRECT_BC_TRANSIENT}),
    "AC_off": frozenset({PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}),
}
Responses: TypeAlias = dict[str, dict[str, torch.Tensor]]
Checks: TypeAlias = dict[str, bool | float]
VerificationRow: TypeAlias = dict[str, str | bool | float]


class CellIdentity(TypedDict):
    cell_id: str
    retinal_class: str
    polarity: str


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(cell_id: str) -> MechanisticGraphTemporalRetina:
    path = SOURCE / "cells" / cell_id.replace("#", "_") / "model-trained.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["cell_id"] == cell_id and checkpoint["stage"] == "trained"
    reference_grid = torch.load(ORIGINAL / "stimuli.pt", weights_only=True)["cone_positions_degs"]
    assert torch.allclose(checkpoint["cone_positions_degs"], reference_grid, atol=1e-7, rtol=0)
    config = MechanisticRetinaConfig(**(checkpoint["model_config"] | {
        "architecture_mode": ArchitectureMode.MECHANISM_IDENTIFIABLE}))
    assert config.causal_contract == "h1-shared-bc-direct-broad-ac"
    assert config.dt_ms == 1000 / 150
    model = build_mechanistic_retina(config, checkpoint["cone_positions_degs"],
        checkpoint["cell_positions_degs"], checkpoint["cell_types"], checkpoint["polarities"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def infer(model: MechanisticGraphTemporalRetina, drive: torch.Tensor, history: torch.Tensor) -> tuple[Responses, Checks]:
    before = {k: v.clone() for k, v in model.state_dict().items()}
    ac_inputs: list[torch.Tensor] = []

    def capture(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        assert len(inputs) == 1
        ac_inputs.append(inputs[0].clone())

    hook = model.amacrine.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            outputs = {mode: model.forward_sequence(drive, observed_counts=history, clamps=clamps)
                       for mode, clamps in CLAMPS.items()}
            repeat = model.forward_sequence(drive, observed_counts=history)
    finally:
        hook.remove()
    normal, direct, ac_off, h1_off = (outputs[k] for k in ("normal", "direct_BC_off", "AC_off", "H1_off"))
    direct_fields = ("bc_broad_presynaptic", "amacrine_local_state", "amacrine_transient_state",
                     "amacrine_local_current", "amacrine_transient_current")
    ac_fields = ("h1_state", "h1_surround_contribution", "bc_direct_presynaptic", "bc_broad_presynaptic",
                 "bc_sustained_current", "bc_transient_current")
    propagated = ("bc_direct_presynaptic", "bc_broad_presynaptic", "amacrine_local_state", "amacrine_transient_state")
    checks: Checks = {
        "AC_input_equals_BC_broad": all(torch.equal(x, y.bc_broad_presynaptic) for x, y in zip(ac_inputs, (*outputs.values(), repeat), strict=True)),
        "H1_clamp_exact_zero": bool(torch.count_nonzero(h1_off.h1_surround_contribution) == 0),
        "direct_BC_clamp_exact_zero": bool(torch.count_nonzero(direct.bc_sustained_current) == 0 and torch.count_nonzero(direct.bc_transient_current) == 0),
        "AC_clamp_exact_zero": bool(torch.count_nonzero(ac_off.amacrine_local_current) == 0 and torch.count_nonzero(ac_off.amacrine_transient_current) == 0),
        "direct_BC_off_preserves_BC_broad_AC": all(torch.equal(getattr(normal, n), getattr(direct, n)) for n in direct_fields),
        "AC_off_preserves_H1_BC_views": all(torch.equal(getattr(normal, n), getattr(ac_off, n)) for n in ac_fields),
        "H1_off_propagates_downstream": all(bool(torch.count_nonzero(getattr(normal, n) - getattr(h1_off, n))) for n in propagated),
        "all_outputs_finite": all(bool(torch.isfinite(t).all()) for o in outputs.values() for t in o.tensors()),
        "state_unchanged": all(torch.equal(v, before[k]) for k, v in model.state_dict().items()),
        "eval_mode_unchanged": not model.training,
        "parameter_gradients_absent": all(p.grad is None for p in model.parameters()),
        "normal_reentry_bitwise_equal": all(torch.equal(a, b) for a, b in zip(normal.tensors(), repeat.tensors(), strict=True)),
    }
    assert all(checks.values())
    for n in propagated:
        checks[f"H1_off_{n}_change_norm"] = float((getattr(normal, n) - getattr(h1_off, n)).norm())
    responses = {mode: {"logit": o.logits[..., 0], "probability": o.spike_probability[..., 0],
                       "ac_local": o.amacrine_local_current[..., 0], "ac_transient": o.amacrine_transient_current[..., 0]}
                 for mode, o in outputs.items()}
    return responses, checks
