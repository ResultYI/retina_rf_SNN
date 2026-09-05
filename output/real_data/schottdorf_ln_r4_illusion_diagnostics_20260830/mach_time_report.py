# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "matplotlib"]
# ///
# How to run: D:/anaconda/python.exe -B mach_time_report.py (saved responses only).
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from diagnostic_stimuli import OUT
from report import COLORS


def main() -> None:
    path = OUT / "responses.pt"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    saved = torch.load(path, weights_only=True)
    time, cells, metadata = saved["time_ms"], saved["cells"], saved["metadata"]
    groups = {cid.replace("#", "_"): [cid] for cid in cells}
    groups.update({group: [cid for cid in cells if group == "all" or metadata[cid]["group"] == group]
                   for group in ("all", "MC_ON", "MC_OFF", "PC_ON", "PC_OFF")})
    for title, ids in groups.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        for row, (label, idx) in enumerate((("dark junction", 8), ("bright junction", 16))):
            for col, channel in enumerate(("logit", "probability")):
                ax = axes[row, col]
                for mode, color in COLORS.items():
                    trace = torch.stack([cells[cid][mode][channel][idx] -
                                         cells[cid][mode][channel][idx+25] for cid in ids]).mean(0)
                    ax.plot(time, trace, color=color, label=mode)
                ax.axvspan(300, 400, color="gray", alpha=0.12)
                ax.axhline(0, color="gray", linewidth=0.5)
                ax.set_xlim(250, 1000)
                ax.set_title(f"Mach {label} | {channel}")
                ax.set_xlabel("time (ms)")
                ax.set_ylabel("ramp minus matched uniform")
                ax.legend(fontsize=8)
        fig.suptitle(f"{title} n={len(ids)} | unchanged original Mach stimuli and metrics")
        fig.savefig(OUT / "figures" / f"mach-time-{title}.png", dpi=150)
        plt.close(fig)
    with (OUT / "mach-boundary-extrema.csv").open(newline="", encoding="utf-8") as stream:
        extrema = list(csv.DictReader(stream))
    lines = ["# Original Mach replay: plateau excursions", "",
             "Unchanged original metric: per-cell maximum above/minimum below the two remote plateau mean-on responses, in fixed boundary regions x=-6..-2 and +2..+6 pixels. Counts use 1e-9 as in the original report. Values remain response units.", "",
             "| Group | Model/mode | Channel | Profile | Above cells | Below cells | Mean max above | Mean min below |",
             "|---|---|---|---|---:|---:|---:|---:|"]
    for group in ("all", "MC_ON", "MC_OFF", "PC_ON", "PC_OFF"):
        ids = groups[group]
        for mode in COLORS:
            for channel in ("logit", "probability"):
                for profile in ("ramp", "matched_uniform"):
                    selected = [r for r in extrema if r["mode"] == mode and r["channel"] == channel and r["profile"] == profile]
                    hi = [max(float(r["overshoot_above_plateaus"]) for r in selected if r["cell_id"] == cid) for cid in ids]
                    lo = [min(float(r["undershoot_below_plateaus"]) for r in selected if r["cell_id"] == cid) for cid in ids]
                    lines.append(f"| {group} | {mode} | {channel} | {profile} | {sum(v>1e-9 for v in hi)}/{len(ids)} | "
                                 f"{sum(v< -1e-9 for v in lo)}/{len(ids)} | {sum(hi)/len(ids):+.9f} | {sum(lo)/len(ids):+.9f} |")
    lines.extend(["", "Time courses: figures/mach-time-*.png. Full scan-position curves and metrics remain in responses.pt and per-cell-responses.csv."])
    (OUT / "MACH_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == after
    (OUT / "mach-supplement-provenance.json").write_text(json.dumps({
        "source_responses_sha256": before, "source_unchanged": True,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inference_or_training": False, "figures": len(groups)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
