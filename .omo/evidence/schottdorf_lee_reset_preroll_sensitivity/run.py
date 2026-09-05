#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy", "pydantic", "opencv-python"]
# ///
# How to run: D:/anaconda/python.exe -B -u run.py
from __future__ import annotations

import csv
from enum import StrEnum
import json
from pathlib import Path
import time
from typing import Final, TypedDict, assert_never

import torch

from inputs import OUT, SOURCE, Inputs, load_cell, prepare_source, sha
from models.mechanistic_retina.contracts import PathwayClamp
from training.mechanistic_retina.losses import expected_bernoulli_nll


class Mode(StrEnum):
    PRODUCTION = "production"
    PREROLL = "preroll_400ms"
    CONTINUOUS = "continuous"


CLAMPS: Final = {
    "normal": frozenset(), "H1_off": frozenset({PathwayClamp.H1}),
    "direct_BC_off": frozenset({PathwayClamp.DIRECT_BC_SUSTAINED, PathwayClamp.DIRECT_BC_TRANSIENT}),
    "AC_off": frozenset({PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}),
}
ZERO_FIELDS: Final = {"normal": (), "H1_off": ("h1_surround_contribution",),
    "direct_BC_off": ("bc_sustained_current", "bc_transient_current"),
    "AC_off": ("amacrine_local_current", "amacrine_transient_current")}
type Responses = dict[str, torch.Tensor]


class Delta(TypedDict):
    mean_abs_delta_logit: float
    p95_abs_delta_logit: float
    max_abs_delta_logit: float


def delta_summary(values: torch.Tensor) -> Delta:
    absolute = values.double().abs().flatten()
    return Delta(mean_abs_delta_logit=float(absolute.mean()),
                 p95_abs_delta_logit=float(torch.quantile(absolute, .95)),
                 max_abs_delta_logit=float(absolute.max()))


@torch.no_grad()
def replay(inputs: Inputs, mode: Mode) -> Responses:
    match mode:
        case Mode.PRODUCTION:
            cones, history = inputs.validation.cone_drive, inputs.validation.spike_events
        case Mode.PREROLL:
            cones, history = inputs.preroll_cones, inputs.preroll_history
        case Mode.CONTINUOUS:
            cones, history = inputs.continuous_cones, inputs.continuous_history
        case unreachable:
            assert_never(unreachable)
    before = {k: v.clone() for k, v in inputs.model.state_dict().items()}
    result = {}
    for label, clamps in CLAMPS.items():
        pieces = []
        for start in range(0, cones.shape[0], 8):
            output = inputs.model.forward_sequence(cones[start:start + 8], observed_counts=history[start:start + 8], clamps=clamps)
            assert all(bool(torch.isfinite(t).all()) for t in output.tensors()), "STOP: nonfinite output"
            assert all(int(torch.count_nonzero(getattr(output, name))) == 0 for name in ZERO_FIELDS[label])
            pieces.append(output.logits)
        full = torch.cat(pieces)
        match mode:
            case Mode.PRODUCTION:
                if label == "normal":
                    assert torch.equal(full, inputs.expected_logits), "STOP: production checkpoint replay differs"
                scored = full[:, 30:]
            case Mode.PREROLL:
                scored = full[:, 60:]
            case Mode.CONTINUOUS:
                scored = full[inputs.stream_indices[:, None], inputs.scored_indices]
            case unreachable:
                assert_never(unreachable)
        assert scored.shape == inputs.validation.spike_events[:, 30:].shape, "STOP: evaluation bins differ"
        result[label] = scored
    assert all(torch.equal(before[k], v) for k, v in inputs.model.state_dict().items())
    assert not inputs.model.training and all(p.grad is None for p in inputs.model.parameters())
    return result


def nll(logits: torch.Tensor, inputs: Inputs) -> float:
    full = torch.zeros_like(inputs.validation.spike_events)
    full[:, 30:] = logits
    return float(expected_bernoulli_nll(full, inputs.validation.spike_events, inputs.validation.valid_mask))


def main() -> None:
    assert not (OUT / "summary.md").exists(), "fresh evidence output required"
    test = delta_summary(torch.tensor([-1., 0., 1.]))
    assert test["mean_abs_delta_logit"] == 2 / 3 and test["max_abs_delta_logit"] == 1
    torch.set_num_threads(2)
    source = prepare_source()
    rows, pathways, checks, tensors = [], [], [], {}
    deltas = {mode: [] for mode in Mode}
    pathway_values = {(mode, label): [] for mode in Mode for label in CLAMPS if label != "normal"}
    for index, cell in enumerate(source.snapshot.cells):
        started = time.monotonic()
        inputs = load_cell(source, index)
        normal = replay(inputs, Mode.PRODUCTION)
        expected_nll = json.loads((SOURCE / "results.json").read_text())["cells"][index]["validation_nll_trained"]
        assert nll(normal["normal"], inputs) == expected_nll
        outputs = {Mode.PRODUCTION: normal}
        print(f"{index + 1}/22 {cell.cell_id} production EXACT; streams={len(inputs.stream_ids)}", flush=True)
        for mode in (Mode.PREROLL, Mode.CONTINUOUS):
            outputs[mode] = replay(inputs, mode)
            print(f"  {mode} complete", flush=True)
        base_nll = nll(normal["normal"], inputs)
        for mode in Mode:
            response = outputs[mode]
            delta = response["normal"] - normal["normal"]
            deltas[mode].append(delta.flatten())
            value = nll(response["normal"], inputs)
            rows.append(dict(cell_id=cell.cell_id, group=f"{cell.retinal_class} {cell.polarity}", mode=mode.value,
                valid_bins=cell.validation_valid_bins, validation_nll=value, delta_nll=value - base_nll, **delta_summary(delta)))
            for label in CLAMPS:
                if label == "normal":
                    continue
                effect = (response[label] - response["normal"]).double().abs()
                base = (normal[label] - normal["normal"]).double().abs()
                pathway_values[(mode, label)].append(effect.flatten())
                pathways.append(dict(cell_id=cell.cell_id, group=f"{cell.retinal_class} {cell.polarity}", mode=mode.value,
                    pathway=label, valid_bins=effect.numel(), mean_abs_delta_logit=float(effect.mean()),
                    change_from_production=float(effect.mean() - base.mean())))
        checks.append(dict(cell_id=cell.cell_id, valid_bins=cell.validation_valid_bins,
            checkpoint_strict_load=True, production_logits_bitwise_equal_saved=True,
            original_target_mask_identity_exact=True, all_modes_same_scored_stimulus_target_bins=True,
            no_cross_recording_trial_or_gap=True, real_observed_history=True,
            exact_zero_clamps=True, model_state_dict_unchanged=True, all_outputs_finite=True,
            no_parameter_gradients=True, scored_identity_sha256=inputs.scored_identity_sha256,
            stream_ids=inputs.stream_ids))
        tensors[cell.cell_id] = dict(target=inputs.validation.spike_events[:, 30:],
            original_source_ids=inputs.validation.source_image_ids, stream_ids=inputs.stream_ids,
            stream_indices=inputs.stream_indices, scored_live_bin_indices=inputs.scored_indices,
            logits={mode.value: output for mode, output in outputs.items()})
        print(f"DONE {cell.cell_id} elapsed={time.monotonic() - started:.1f}s", flush=True)
    population = []
    for mode in Mode:
        selected = [r for r in rows if r["mode"] == mode.value]
        total = sum(r["valid_bins"] for r in selected)
        population.append(dict(mode=mode.value, cells=22, valid_bins=total,
            validation_nll_cell_mean=sum(r["validation_nll"] for r in selected) / 22,
            delta_nll_cell_mean=sum(r["delta_nll"] for r in selected) / 22,
            validation_nll_bin_weighted=sum(r["validation_nll"] * r["valid_bins"] for r in selected) / total,
            delta_nll_bin_weighted=sum(r["delta_nll"] * r["valid_bins"] for r in selected) / total,
            **delta_summary(torch.cat(deltas[mode]))))
    population_pathways = []
    for mode in Mode:
        for label in CLAMPS:
            if label == "normal":
                continue
            base = float(torch.cat(pathway_values[(Mode.PRODUCTION, label)]).mean())
            value = float(torch.cat(pathway_values[(mode, label)]).mean())
            population_pathways.append(dict(mode=mode.value, pathway=label,
                mean_abs_delta_logit=value, change_from_production=value - base))
    assert sum(c["valid_bins"] for c in checks) == 65760
    assert all(sha(Path(p)) == h for p, h in source.hashes.items()), "STOP: input/source changed during inference"
    for name, content in (("per_cell.csv", rows), ("pathway_effects.csv", pathways),
                          ("population.csv", population), ("population_pathways.csv", population_pathways)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(content[0]))
            writer.writeheader()
            writer.writerows(content)
    torch.save(tensors, OUT / "evaluation_logits.pt")
    verification = dict(status="COMPLETED", cells=22, evaluation_bins_per_mode=65760,
        sensitivity_pass_threshold=None, training=False, model_modified=False,
        protocol=dict(production="reset each 150-bin segment; score local bins 30..149",
            preroll="60 real preceding bins before first scored bin; 180-bin input; score bins 60..179",
            continuous="one uninterrupted 3000-bin forward per recording/trial from live t=0; score original validation bins only",
            history="unchanged production binary observed spike history, including unscored preceding bins; strictly-past shift remains inside model",
            clamps="each off condition applies throughout that mode's complete context; no normal-state splice",
            aggregation="primary NLL: unweighted 22-cell mean; also report bin-weighted NLL; population logit quantiles/effects pool all 65760 scored bins"),
        core_files_different_from_training_manifest=source.core_source_drift,
        core_source_note="Current sources recorded without modification. Strict loading and all saved production logits/NLL must match exactly; no checkpoint conversion.",
        checks=checks, source_sha256=source.hashes)
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    from report import write_summary
    write_summary()
    print(json.dumps(population, indent=2), flush=True)


if __name__ == "__main__":
    main()
