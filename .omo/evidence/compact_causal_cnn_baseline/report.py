#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy==2.2.6"]
# ///
# How to run: D:/anaconda/python.exe -B .omo/evidence/compact_causal_cnn_baseline/report.py
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Final

import numpy as np
import torch

ROOT: Final = Path(__file__).resolve().parents[3]
OUT: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from baselines.center_surround_ln import LNError
from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import sha256_file
from training.mechanistic_retina.losses import expected_bernoulli_nll


def main() -> None:
    torch.set_num_threads(2)
    if not (OUT / "training_complete.json").exists():
        raise LNError("STOP: all 22 CNN cells must finish before aggregation")
    preflight = json.loads((OUT / "preflight.json").read_text())
    runtime = json.loads((OUT / "runtime.json").read_text())
    if sha256_file(OUT / "preflight.json") != runtime["preflight_sha256"]:
        raise LNError("STOP: report preflight differs from the fitted contract")
    if any(sha256_file(Path(p)) != h for p, h in preflight["source_sha256"].items()):
        raise LNError("STOP: a frozen source changed")
    rows, checks = [], []
    for cell in preflight["cells"]:
        name = cell["cell_id"].replace("#", "_")
        directory = OUT / "cells" / name
        fit = json.loads((directory / "results.json").read_text())
        saved = torch.load(directory / "validation-predictions.pt", map_location="cpu", weights_only=True)
        native = torch.load(OUT / "inputs" / f"{name}.pt", map_location="cpu", weights_only=True)["validation"]
        if not torch.equal(saved["target"], native["spike_events"]) or not torch.equal(saved["valid_mask"], native["valid_mask"]):
            raise LNError("STOP: CNN evaluation target or mask differs")
        if saved["source_image_ids"] != native["source_image_ids"] or saved["trial_indices"] != native["trial_indices"]:
            raise LNError("STOP: CNN sequence identity differs")
        values = [float(expected_bernoulli_nll(saved[k], saved["target"], saved["valid_mask"]))
                  for k in ("logits_raw", "logits_trained")]
        if not all(bool(torch.isfinite(saved[k]).all()) for k in ("logits_raw", "logits_trained")):
            raise LNError("STOP: CNN logits are nonfinite")
        row = dict(cell_id=cell["cell_id"], group=cell["group"], cnn_nll=values[1], cnn_raw_nll=values[0],
                   constant_nll=cell["constant_nll"], ln_nll=cell["ln_nll"], sc_adapted_nll=cell["sc_adapted_nll"],
                   canonical_v1_nll=cell["canonical_v1_nll"], selected_lr=fit["selected_lr"], best_step=fit["best_step"],
                   stopping_step=fit["stopping_step"], refit_steps=fit["refit_steps"],
                   parameter_count=fit["parameter_counts"]["total"], train_bins=cell["train_bins"], validation_bins=cell["validation_bins"])
        for baseline in ("constant", "ln", "sc_adapted", "canonical_v1"):
            row[f"cnn_minus_{baseline}"] = row["cnn_nll"] - row[f"{baseline}_nll"]
        rows.append(row)
        checks.append(dict(cell_id=cell["cell_id"], target_mask_order_exact=True, finite_outputs=True,
                           cpu_minus_gpu_nll=values[1]-fit["validation_nll_trained_gpu"],
                           source_checkpoint_sha256=sha256_file(directory / "cnn-trained.pt")))
    metrics = ("cnn_nll", "constant_nll", "ln_nll", "sc_adapted_nll", "canonical_v1_nll")
    summary = []
    for group in ("overall", "MC_ON", "MC_OFF", "PC_ON", "PC_OFF"):
        selected = rows if group == "overall" else [r for r in rows if r["group"] == group]
        summary.append(dict(group=group, cells=len(selected), **{k: float(np.mean([r[k] for r in selected])) for k in metrics}))
    for filename, values in (("per_cell.csv", rows), ("group_summary.csv", summary)):
        with (OUT / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    payload = dict(status="COMPLETED", cells=rows, summary=summary, contract=preflight["contract"],
                   reporting_runtime=dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__),
                   evaluation="native CPU runtime Bernoulli NLL on saved full-precision logits; equal cell weight",
                   source_hashes_unchanged=True, all_22_target_mask_order_exact=True, checks=checks)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Compact causal CNN: final 22-cell comparison", "", "STATUS: COMPLETED", "",
             "| Group | Cells | CNN | Constant | Center-surround LN | SC-adapted | Canonical V1 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['group']} | {r['cells']} | " + " | ".join(f"{r[k]:.12f}" for k in metrics) + " |" for r in summary]
    lines += ["", "## Per cell", "", "| Cell | Group | CNN | Constant | LN | SC-adapted | Canonical V1 | LR | Best / stop |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['cell_id']} | {r['group']} | " + " | ".join(f"{r[k]:.12f}" for k in metrics)
              + f" | {r['selected_lr']:g} | {r['best_step']} / {r['stopping_step']} |" for r in rows]
    lines += ["", "## Fixed architecture and training contract", "",
              "Conv3D 1->4 (12,5,5), ReLU; Conv3D 4->4 (9,3,3), temporal dilation 6, ReLU; learned 4x11x11 readout.",
              "Temporal left pads 11/48; no spatial padding; spatial sizes 17->13->11. Stimulus receptive field 60 bins (lags 0..59).",
              "LN head: z=readout+history_weight*strictly-past fixed history+bias. No additional output nonlinearity before Bernoulli logits.",
              "Adam, batch 8, seed 61001, lr candidates {0.001,0.0003}; no regularizer, no weight decay, no architecture sweep.",
              "Exact LN inner split and guard; max 1000, patience 200, min_delta 1e-7, dev evaluation every step; raw step zero eligible.",
              "Select lowest unpenalized inner-dev NLL, then fresh full-train refit for best-step count. Original validation never selects a model.",
              "Input tensors generated with the frozen native loader and verified against the corrected SC preflight. GPU training is float32 without AMP or TF32.",
              "Final NLL uses the original native CPU evaluation runtime and identical saved target/mask/order, averaged equally over cells.",
              "", "## Parameter accounting", "",
              "CNN: 1204 first-conv + 1300 second-conv + 484 readout + 2 bias/history = 2990 per cell; all trainable and optimizer-listed; 65780 across 22 independent fits.",
              "Constant: 1/cell. LN: 128/cell. SC-adapted: 64 inherited center coordinates + 4 fitted output parameters (68 raw bookkeeping, not functional DoF). Canonical V1: 129 total / 33 trainable per cell.",
              "Prediction/capacity comparison only; no matched-capacity claim. Existing baseline artifacts were not modified or refitted.",
              "", "## Evidence", "",
              "preflight.json: frozen definitions, tensor/source hashes, splits, comparison source values.",
              "runtime.json: GPU arithmetic/runtime identity. training_complete.json: 44 inner fits and 22 refits.",
              "cells/*: two inner checkpoints and full dev trajectories, raw/refit checkpoints, validation logits, fitting parameter counts.",
              "per_cell.csv, group_summary.csv, results.json: full-precision comparison results. verification.md: focused contract checks."]
    (OUT / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
