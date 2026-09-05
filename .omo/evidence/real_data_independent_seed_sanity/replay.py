#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy", "pydantic"]
# ///
# How to run: imported by run.py in the frozen D:/anaconda/python.exe runtime.
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

from source import APPLICATION, Cell
from common import CLAMPS
from metrics import response_metrics
from data.retinal_recording import RealSequenceSplit
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina

SIGNATURES: Final = {
    "ramp_minus_matched_uniform_x-04": "Mach dark",
    "ramp_minus_matched_uniform_x+04": "Mach bright",
    "bright_surround_minus_dark_surround": "SBC",
    "intersection_minus_corridor": "Hermann",
    "on_bright_bar_minus_on_dark_bar": "White",
}
ZERO: Final = {"H1_off": ("h1_surround_contribution",),
    "direct_BC_off": ("bc_sustained_current", "bc_transient_current"),
    "AC_off": ("amacrine_local_current", "amacrine_transient_current")}


@dataclass(frozen=True, slots=True)
class FitIdentity:
    cell: Cell
    label: str
    seed: int


@torch.no_grad()
def pathway_effects(model: MechanisticGraphTemporalRetina, validation: RealSequenceSplit) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    before = {k: v.clone() for k, v in model.state_dict().items()}
    responses = {}
    for mode, clamps in CLAMPS.items():
        pieces = []
        for start in range(0, validation.cone_drive.shape[0], 8):
            output = model.forward_sequence(validation.cone_drive[start:start + 8],
                observed_counts=validation.spike_events[start:start + 8], clamps=clamps)
            assert all(bool(torch.isfinite(t).all()) for t in output.tensors())
            assert all(int(torch.count_nonzero(getattr(output, n))) == 0 for n in ZERO.get(mode, ()))
            pieces.append(output.logits)
        responses[mode] = torch.cat(pieces)
    effects = {mode: float((values - responses["normal"])[validation.valid_mask].abs().double().mean())
               for mode, values in responses.items() if mode != "normal"}
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    return effects, responses


@torch.no_grad()
def illusion(model: MechanisticGraphTemporalRetina, identity: FitIdentity) -> tuple[list, dict[str, torch.Tensor]]:
    saved = torch.load(APPLICATION / "illusion/inputs.pt", weights_only=True)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    values = {}
    for mode in ("normal", "AC_off"):
        output = model.forward_sequence(saved["cone_drive"], observed_counts=saved["history"], clamps=CLAMPS[mode])
        assert all(bool(torch.isfinite(t).all()) for t in output.tensors())
        assert all(int(torch.count_nonzero(getattr(output, n))) == 0 for n in ZERO.get(mode, ()))
        values[mode] = output.logits[..., 0]
    rows = []
    for pair in saved["pairs"]:
        if pair["name"] in SIGNATURES or pair["control"]:
            a, b = pair["a"], pair["b"]
            normal = response_metrics(values["normal"][a] - values["normal"][b], saved["time_ms"])
            off = response_metrics(values["AC_off"][a] - values["AC_off"][b], saved["time_ms"])
            normal_mean, off_mean = normal["mean_on"], off["mean_on"]
            assert normal_mean is not None and off_mean is not None
            sign_normal = int(normal_mean > 1e-9) - int(normal_mean < -1e-9)
            sign_off = int(off_mean > 1e-9) - int(off_mean < -1e-9)
            rows.append({"cell_id": identity.cell.cell_id, "group": identity.cell.group,
                "fit": identity.label, "seed": identity.seed, "signature": SIGNATURES.get(pair["name"], pair["name"]),
                "pair_name": pair["name"], "control": pair["control"], "pair_a": a, "pair_b": b,
                "normal_paired_logit": normal_mean, "AC_off_paired_logit": off_mean,
                "normal_A": response_metrics(values["normal"][a], saved["time_ms"])["mean_on"],
                "normal_B": response_metrics(values["normal"][b], saved["time_ms"])["mean_on"],
                "AC_off_A": response_metrics(values["AC_off"][a], saved["time_ms"])["mean_on"],
                "AC_off_B": response_metrics(values["AC_off"][b], saved["time_ms"])["mean_on"],
                "normal_sign": sign_normal, "AC_off_sign": sign_off,
                "AC_off_reverses_normal": sign_normal * sign_off < 0,
                **{f"normal_{k}": v for k, v in normal.items()}, **{f"AC_off_{k}": v for k, v in off.items()}})
            if pair["control"]:
                assert torch.equal(values["normal"][a], values["normal"][b])
                assert torch.equal(values["AC_off"][a], values["AC_off"][b])
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert not model.training and all(p.grad is None for p in model.parameters())
    return rows, values
