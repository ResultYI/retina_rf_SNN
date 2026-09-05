from __future__ import annotations

import csv
from pathlib import Path
from typing import Final

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from diagnostic_stimuli import OUT, VARIANTS, diagnostic_bank
from metrics import Row

COLORS: Final = {"R4_normal": "#222222", "R4_H1_off": "#2477bb", "R4_AC_off": "#cd5134", "LN": "#8050ad"}


def stimulus_figure() -> None:
    bank = diagnostic_bank()
    fig, axes = plt.subplots(4, 4, figsize=(12, 12), constrained_layout=True)
    for j, (family, variant, a, b, _, _) in enumerate(VARIANTS):
        for control in (0, 1):
            row = 2 * j + control
            indices = (a + 2*control, b + 2*control, 63 + 4*j + 2*control, 64 + 4*j + 2*control)
            for col, idx in enumerate(indices):
                axes[row, col].imshow(bank.patches[idx] + 1, cmap="gray", vmin=0.75, vmax=1.25,
                                      extent=(-8.5, 8.5, -8.5, 8.5))
                axes[row, col].set_title(f"{family} {'control' if control else 'target'}\n"
                                         f"{'original' if col < 2 else 'diagnostic'} {'A' if col % 2 == 0 else 'B'}")
                axes[row, col].set_xlabel("x (pooled pixels)")
                axes[row, col].set_ylabel("y (pooled pixels)")
    fig.suptitle("Fixed diagnostic local views | target-centered exact-radius histograms preserved\n"
                 "Relative L+M 0.75 / 1 / 1.25; pitch 0.05390625 deg; identical original exposure", fontsize=12)
    fig.savefig(OUT / "diagnostic-stimuli.png", dpi=160)
    plt.close(fig)


def response_figures(saved: dict, ids: list[str], title: str) -> None:
    cells, time = saved["cells"], saved["time_ms"]
    values = {label: {channel: torch.stack([cells[cid][label][channel] for cid in ids]).mean(0)
                      for channel in ("logit", "probability")} for label in COLORS}
    active = (time >= 300) & (time < 400)
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), constrained_layout=True)
    for row, (family, a, b) in enumerate((("Mach", 0, 0), ("SBC", 50, 51), ("Hermann", 54, 55), ("White", 58, 59))):
        for col, channel in enumerate(("logit", "probability")):
            ax = axes[row, col]
            for label, color in COLORS.items():
                trace = values[label][channel]
                if row == 0:
                    for start, style in ((0, "-"), (25, "--")):
                        ax.plot(torch.arange(-12, 13) * 0.05390625,
                                trace[start:start + 25, active].mean(1) - trace[62, active].mean(),
                                style, color=color, label=label if start == 0 else None)
                    ax.set_xlabel("scan x (deg); dashed = uniform control")
                else:
                    ax.plot(time, trace[a] - trace[b], color=color, label=label)
                    ax.plot(time, trace[a + 2] - trace[b + 2], "--", color=color)
                    ax.set_xlim(250, 1000)
                    ax.axvspan(300, 400, color="gray", alpha=0.12)
                    ax.set_xlabel("time (ms); dashed = matched control")
                ax.set_title(f"{family} | {channel}")
                ax.set_ylabel("mean-on minus blank" if row == 0 else "A minus B response")
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.legend(fontsize=8)
    fig.suptitle(f"{title} | original replay | n={len(ids)}")
    fig.savefig(OUT / "figures" / f"original-{title}.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for j, (family, _, a, b, _, _) in enumerate(VARIANTS):
        for col, channel in enumerate(("logit", "probability")):
            ax = axes[j, col]
            for label, color in COLORS.items():
                trace = values[label][channel]
                ax.plot(time, trace[a] - trace[b], color=color, label=f"{label} original")
                ax.plot(time, trace[63+4*j] - trace[64+4*j], "--", color=color, label=f"{label} diagnostic")
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.axvspan(300, 400, color="gray", alpha=0.12)
            ax.set_xlim(250, 1000)
            ax.set_title(f"{family} | {channel}")
            ax.set_xlabel("time (ms)")
            ax.set_ylabel("A minus B response")
            ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f"{title} | original versus diagnostic | n={len(ids)}; controls A-B = 0")
    fig.savefig(OUT / "figures" / f"diagnostic-{title}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    saved = torch.load(OUT / "responses.pt", weights_only=True)
    (OUT / "figures").mkdir(exist_ok=True)
    for cid in saved["cells"]:
        response_figures(saved, [cid], cid.replace("#", "_"))
    for group in ("all", "MC_ON", "MC_OFF", "PC_ON", "PC_OFF"):
        ids = [cid for cid in saved["cells"] if group == "all" or saved["metadata"][cid]["group"] == group]
        response_figures(saved, ids, group)
    with (OUT / "group-responses.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    pairs = (("Mach dark", "ramp_minus_matched_uniform_x-04"), ("Mach bright", "ramp_minus_matched_uniform_x+04"),
             ("SBC", "bright_surround_minus_dark_surround"), ("Hermann original", "intersection_minus_corridor"),
             ("Hermann diagnostic", "contour_rearrangement_A_minus_B"), ("White original", "on_bright_bar_minus_on_dark_bar"),
             ("White diagnostic", "remote_contour_rearrangement_A_minus_B"))
    lines = ["# Frozen LN versus R4-dev illusion diagnostics", "",
        "22 cells: MC ON 5, MC OFF 4, PC ON 9, PC OFF 4. Original 63 input sequences, timing, controls, history and metrics reused unchanged. Diagnostic views use the same envelope and 150 Hz time axis. No training/model edits.", "",
        "Original R4 replay max absolute logit/probability error: 0. All learned states unchanged; H1/AC structural clamps exact-zero. Target-centered exact-radius luminance histograms are preserved separately for each target; this does not assert matching around learned off-center LN Gaussian centers.", "",
        "White custom remote-contour rearrangement is motivated by [Howe 2001](https://doi.org/10.1068/p3212) / [Howe 2005](https://doi.org/10.1068/p5414). Hermann custom contour rearrangement is motivated by [Geier et al. 2008](https://doi.org/10.1068/p5622). The 17x17 permutation produces fragmented contours, not a smooth curved grid. These are discrete diagnostic constructions, not exact reproductions or claims of a particular perceptual effect.", "",
        "Construction and all fixed constants: protocol.json. Original and diagnostic are separately centered local views. For every exact radius, angle-ordered pixel luminances are cyclically permuted; protected radii White=4 px and Hermann=2 px retain all target pixels. No response-dependent stimulus selection.", "",
        "## Mean-on response differences", "",
        "Mean over 300<=t<400 ms, then equal-cell mean. Mach entries are junction minus uniform-control responses. SBC is bright-surround minus dark-surround; Hermann is intersection minus corridor; White is bright-bar target minus dark-bar target. Signed response units, no perceptual score.", "",
        "| Group | Pair | Channel | LN | R4 normal | R4 H1-off | R4 AC-off |",
        "|---|---|---|---:|---:|---:|---:|"]
    for group in ("all", "MC_ON", "MC_OFF", "PC_ON", "PC_OFF"):
        for label, name in pairs:
            for channel in ("logit", "probability"):
                values = [next(float(r["mean_on"]) for r in rows if r["group"] == group and r["name"] == name
                               and r["mode"] == mode and r["channel"] == channel and r["kind"] == "pair_difference")
                          for mode in ("LN", "R4_normal", "R4_H1_off", "R4_AC_off")]
                lines.append(f"| {group} | {label} | {channel} | " + " | ".join(f"{v:+.9f}" for v in values) + " |")
    with (OUT / "annular-luminance.csv").open(encoding="utf-8", newline="") as stream:
        stats = list(csv.DictReader(stream))
    lines.extend(["", "## Annular matching", "",
        f"Maximum original-to-diagnostic mean luminance change: {max(abs(float(r['mean_delta'])) for r in stats):.12g}.",
        f"Maximum std change: {max(abs(float(r['std_delta'])) for r in stats):.12g}; maximum sorted histogram error: {max(float(r['sorted_histogram_max_abs_error']) for r in stats):.12g}.",
        "", "Per-ring original/diagnostic means, std, min/max, three luminance fractions, changed-pixel counts: annular-luminance.csv. Actual per-cell BC/AC support statistics: per-cell-spatial-support-statistics.csv.", "",
        "All original and diagnostic contextual paired controls A-B are exact-zero. Mach uniform-control profile and plateau excursions are in mach-boundary-extrema.csv.", "",
        "## Files", "", "- per-cell-responses.csv / group-responses.csv: signed mean, peak, latency, integral, onset/offset and direction counts.",
        "- per-cell-comparisons.csv / group-comparisons.csv: diagnostic minus original and each R4 mode minus LN, using identical response metrics.",
        "- responses.pt: raw logit/probability time courses, R4 AC currents, names and cell identities.",
        "- figures/: original and diagnostic time courses for every cell and all four groups; diagnostic-stimuli.png: input views.",
        "- verification.json: frozen checkpoint/source hashes and execution checks."])
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
