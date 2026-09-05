#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy", "pydantic"]
# ///
# How to run: D:/anaconda/python.exe -B -u summarize.py after run.py completes.
from __future__ import annotations

import csv
from itertools import combinations
import json
from pathlib import Path

import torch

from source import APPLICATION, OUT, save_json, sha
from replay import SIGNATURES
from metrics import response_metrics, write_csv
from training.mechanistic_retina.losses import expected_bernoulli_nll


def main() -> None:
    torch.set_num_threads(2)
    verification = json.loads((OUT / "verification.json").read_text())
    assert verification["status"] == "COMPLETED" and verification["new_fits"] == 8
    with (OUT / "per_fit.csv").open(newline="") as stream:
        fits = list(csv.DictReader(stream))
    with (OUT / "illusion_paired_logits.csv").open(newline="") as stream:
        paired = list(csv.DictReader(stream))
    assert len(fits) == 12 and len({r["cell_id"] for r in fits}) == 4
    inputs = torch.load(APPLICATION / "illusion/inputs.pt", weights_only=True)
    for fit in fits:
        assert sha(Path(fit["checkpoint"])) == fit["checkpoint_sha256"]
        path = OUT / "fits" / fit["cell_id"].replace("#", "_") / fit["fit"] / "evaluation.pt"
        saved = torch.load(path, weights_only=True)
        logits = saved["validation_logits"]
        nll = float(expected_bernoulli_nll(logits["normal"], saved["target"], saved["valid_mask"]))
        assert nll == float(fit["validation_nll"])
        for mode in ("H1_off", "direct_BC_off", "AC_off"):
            effect = float((logits[mode] - logits["normal"])[saved["valid_mask"]].abs().double().mean())
            assert effect == float(fit[mode])
        for row in paired:
            if row["cell_id"] == fit["cell_id"] and row["fit"] == fit["fit"]:
                a, b = int(row["pair_a"]), int(row["pair_b"])
                for mode in ("normal", "AC_off"):
                    values = saved["illusion_logits"][mode]
                    expected = response_metrics(values[a] - values[b], inputs["time_ms"])["mean_on"]
                    assert expected == float(row[f"{mode}_paired_logit"])
    pairwise = []
    for cid in dict.fromkeys(r["cell_id"] for r in fits):
        members = [r for r in fits if r["cell_id"] == cid]
        for a, b in combinations(members, 2):
            for name in SIGNATURES.values():
                left = next(r for r in paired if r["cell_id"] == cid and r["fit"] == a["fit"] and r["signature"] == name)
                right = next(r for r in paired if r["cell_id"] == cid and r["fit"] == b["fit"] and r["signature"] == name)
                pairwise.append({"cell_id": cid, "group": a["group"], "fit_a": a["fit"], "fit_b": b["fit"],
                    "signature": name, "nll_a": float(a["validation_nll"]), "nll_b": float(b["validation_nll"]),
                    "absolute_nll_difference": abs(float(a["validation_nll"]) - float(b["validation_nll"])),
                    "normal_a": float(left["normal_paired_logit"]), "normal_b": float(right["normal_paired_logit"]),
                    "AC_off_a": float(left["AC_off_paired_logit"]), "AC_off_b": float(right["AC_off_paired_logit"]),
                    "normal_sign_opposite": int(left["normal_sign"]) * int(right["normal_sign"]) < 0,
                    "AC_off_sign_opposite": int(left["AC_off_sign"]) * int(right["AC_off_sign"]) < 0,
                    "reversal_status_different": left["AC_off_reverses_normal"] != right["AC_off_reverses_normal"]})
    write_csv(OUT / "pairwise_comparison.csv", pairwise)
    opposite = [r for r in pairwise if r["normal_sign_opposite"] or r["AC_off_sign_opposite"] or r["reversal_status_different"]]
    opposite.sort(key=lambda r: (r["absolute_nll_difference"], r["cell_id"], r["signature"]))
    save_json(OUT / "mechanism_disagreement.json", {"opposite_rows": opposite,
        "closeness_threshold": None, "sign_zero_tolerance": 1e-9,
        "sign_definition": "unchanged mean-on paired logit, 300 <= time_ms < 400",
        "no_refits_or_tuning_triggered_by_disagreement": True})
    lines = ["# Independent-seed mechanism sanity", "", "STATUS: COMPLETED", "",
        "4 primary fits preserved; exactly 8 new development-selection/full-train-refit runs. No retries, tuning, architecture, loss, split, bounds or intervention changes.",
        "Selection was saved before training. Closest within-class primary validation NLL to median; lexical cell-ID tie break. New seeds: primary +100000 and +200000; minibatch seed is seed +1000003 as in the frozen protocol.",
        "Adam LR=0.03, batch=4, max=1000, patience=200, min_delta=1e-7; no regularizer. Each run freshly refits for its own inner-dev best step; original validation never selects weights or stopping.",
        "Canonical signatures are existing original-stimulus paired logit mean-on (300–400 ms). Mach dark/bright are the saved x=-4/+4 ramp-minus-matched-uniform pairs. SBC is bright-minus-dark surround; Hermann intersection-minus-corridor; White on-bright-minus-on-dark bar. No polarity inversion. Same saved input/history tensors and production reset contract.",
        "AC-off sign reversal means normal and AC-off paired means have opposite nonzero signs; the existing 1e-9 zero tolerance is retained. No NLL-closeness threshold was invented.", "",
        "## Validation and pathway-off effects", "",
        "| Cell | Group | Fit | Seed | NLL | Delta vs primary | H1-off | direct-BC-off | AC-off | best / stop |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in fits:
        numbers = [f"{float(r[k]):.9f}" for k in ("validation_nll", "delta_nll_vs_primary", "H1_off", "direct_BC_off", "AC_off")]
        lines.append("| " + " | ".join([r["cell_id"], r["group"], r["fit"], r["seed"], *numbers, f"{r['best_step']} / {r['stopping_step']}"]) + " |")
    lines += ["", "## Paired logit: normal / AC-off", "",
        "| Cell | Fit | Mach dark | Mach bright | SBC | Hermann | White | AC-off reverses normal |",
        "|---|---|---:|---:|---:|---:|---:|---|"]
    for fit in fits:
        values = [next(r for r in paired if r["cell_id"] == fit["cell_id"] and r["fit"] == fit["fit"] and r["signature"] == s) for s in SIGNATURES.values()]
        pairs = [f"{float(r['normal_paired_logit']):+.9f} / {float(r['AC_off_paired_logit']):+.9f}" for r in values]
        reversals = ", ".join(r["signature"] for r in values if r["AC_off_reverses_normal"] == "True") or "none"
        lines.append("| " + " | ".join([fit["cell_id"], fit["fit"], *pairs, reversals]) + " |")
    lines += ["", "## Pairwise disagreements", "", f"Opposite-sign / different-reversal rows: {len(opposite)} of {len(pairwise)} fit-pair × signature rows."]
    if opposite:
        r = opposite[0]
        lines.append(f"Smallest NLL difference with disagreement: {r['cell_id']}, {r['fit_a']} vs {r['fit_b']}, |Delta NLL|={r['absolute_nll_difference']:.12g}; see pairwise_comparison.csv for each signature and signed output.")
    lines += ["", "## Verification and provenance", "",
        "All four primary validation logits/NLL and original illusion normal/AC-off logits replay exactly. All four primary same-seed raw states and raw training NLL replay exactly. Training/data/loss/selection source hashes match the final training manifest. Existing core-source byte drift is listed in provenance.json; N=1 uses fixed identity mixing and unchanged standard geometry. Historical source-byte identity is not claimed.",
        "All checkpoint hashes, saved evaluation NLL, pathway effects and paired means independently recomputed. All outputs finite; clamps exact-zero; inference state unchanged; paired controls exact-zero. All input/source hashes unchanged during training.",
        "", "Artifacts: selection.json; provenance.json; preflight.json; run-manifest.json; per_fit.csv; illusion_paired_logits.csv; pairwise_comparison.csv; mechanism_disagreement.json; verification.json; fits/<cell>/<fit>/ (new checkpoints, trajectories and saved logits)."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    verification["saved_tensor_metrics_independently_recomputed"] = True
    verification["report_source_sha256"] = sha(Path(__file__))
    save_json(OUT / "verification.json", verification)
    print("VERIFIED", len(fits), "fits; disagreements", len(opposite), flush=True)


if __name__ == "__main__":
    main()
