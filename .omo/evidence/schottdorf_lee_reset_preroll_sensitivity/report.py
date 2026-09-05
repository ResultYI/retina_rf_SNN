#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# How to run: D:/anaconda/python.exe -B report.py after frozen inference.
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Final

OUT: Final = Path(__file__).resolve().parent


def write_summary() -> None:
    tables = {}
    for name in ("per_cell", "population", "pathway_effects", "population_pathways"):
        with (OUT / f"{name}.csv").open(encoding="utf-8", newline="") as handle:
            tables[name] = list(csv.DictReader(handle))
    verification = json.loads((OUT / "verification.json").read_text())
    lines = ["# Schottdorf–Lee reset/pre-roll sensitivity", "", "STATUS: COMPLETED; no sensitivity pass/fail threshold.", "",
        "Final shared-BC 22 trained checkpoints; 37 recordings. Original validation: 65,760 identical scored bins in all modes. No training, model edits, new probes, or checkpoint conversion.", "",
        "## Exact evaluation contract", "",
        "- Production: reset at each 150-bin (1 s) segment, score local bins 30–149; first 200 ms is warmup.",
        "- 400 ms causal pre-roll: 60 actual preceding bins before each first scored bin, including the original 200 ms warmup. Input covers 180 bins; score only bins 60–179. This is not an additional 400 ms before the original segment.",
        "- Continuous: one uninterrupted 3000-bin (20 s) forward from live t=0 for each recording/trial, without resets between its 1 s segments. Only original validation bins are scored. There is no state-transfer API change: original forward evaluates the complete causal prefix.",
        "- Preceding stimulus and binary observed spikes are loaded from the original train/validation timeline. Observed history includes preceding unscored bins; the model's strictly-past shift remains unchanged. No cross-recording/trial or discontinuous context.",
        "- H1-off, direct-BC-off and AC-off apply throughout each condition's entire context, not by splicing normal state into a clamped continuation.", "",
        "Primary overall NLL is the unweighted mean across 22 cells, consistent with the prior 22-cell reporting. Also shown: bin-weighted NLL. Population |Δlogit| statistics and pathway magnitudes pool the 65,760 scored bins. ΔNLL and Δlogit are mode minus production. No pass thresholds are defined.", "",
        "## Population", "", "| Mode | Cell-mean NLL | ΔNLL | Bin-weighted NLL | ΔNLL weighted | Mean abs(Δlogit) | P95 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in tables["population"]:
        fields = ("validation_nll_cell_mean", "delta_nll_cell_mean", "validation_nll_bin_weighted", "delta_nll_bin_weighted",
                  "mean_abs_delta_logit", "p95_abs_delta_logit", "max_abs_delta_logit")
        lines.append("| " + row["mode"] + " | " + " | ".join(f"{float(row[k]):.9f}" for k in fields) + " |")
    lines += ["", "## Pathway-off effect on the same scored bins", "",
        "Each magnitude is mean |z_off − z_normal| within that mode, not the normal response's reset sensitivity.", "",
        "| Mode | Pathway | Mean abs(Δlogit) | Change from production |", "|---|---|---:|---:|"]
    for row in tables["population_pathways"]:
        lines.append(f"| {row['mode']} | {row['pathway']} | {float(row['mean_abs_delta_logit']):.9f} | {float(row['change_from_production']):+.9f} |")
    candidates = [r for r in tables["per_cell"] if r["mode"] != "production"]
    most_sensitive = sorted(candidates, key=lambda r: abs(float(r["delta_nll"])), reverse=True)
    lines += ["", "## Most sensitive cells", "", "Ranked by largest absolute ΔNLL across the two alternatives; each cell listed once.", "",
        "| Cell | Group | Mode | NLL | ΔNLL | Mean abs(Δlogit) | P95 | Max |", "|---|---|---|---:|---:|---:|---:|---:|"]
    seen = set()
    for row in most_sensitive:
        if row["cell_id"] in seen:
            continue
        seen.add(row["cell_id"])
        values = " | ".join(f"{float(row[k]):.9f}" for k in ("validation_nll", "delta_nll", "mean_abs_delta_logit", "p95_abs_delta_logit", "max_abs_delta_logit"))
        lines.append(f"| {row['cell_id']} | {row['group']} | {row['mode']} | {values} |")
        if len(seen) == 5:
            break
    lines += ["", "## Verification and provenance", "",
        "- 22 strict checkpoint loads; production logits bitwise equal to the saved validation logits and NLL exactly equal to the final recorded NLL.",
        "- Original target/mask/source order exact; all modes score the same 65,760 bins and use the same targets/stimulus at those bins.",
        "- Off contributions exact-zero; finite outputs; model state_dict unchanged and no parameter gradients.",
        "- Current core source hashes differ from the original training manifest for the files listed below. No sources were changed here. Strict checkpoint load and full production replay were required, without converting checkpoints. Current hashes are recorded in verification.json; historical source-byte identity is not claimed.", ""]
    lines += [f"- `{p}`" for p in verification["core_files_different_from_training_manifest"]]
    lines += ["", "## Artifacts", "", "- per_cell.csv: 66 cell/mode rows with NLL, ΔNLL and mean/P95/max |Δlogit|.",
        "- pathway_effects.csv: 198 cell/mode/pathway rows with absolute effect and change from production.",
        "- population.csv and population_pathways.csv: pooled/equal-cell aggregate definitions above.",
        "- evaluation_logits.pt: scored logits for all modes/clamps, targets and explicit recording/trial/live-bin identity mappings; not a model checkpoint.",
        "- verification.json: contracts, checkpoint/source hashes and per-cell checks.",
        "- inputs.py, run.py, report.py: evidence-only reproduction scripts.", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    write_summary()
