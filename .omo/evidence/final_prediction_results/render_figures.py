#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.2.6", "matplotlib==3.10.8", "pydantic==2.8.2"]
# ///
# How to run: D:/anaconda/python.exe -B .omo/evidence/final_prediction_results/render_figures.py
from __future__ import annotations

import csv
import json
from typing import Final

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from build_package import Cell, GROUPS, MODELS, OUT, PAIRS

COLORS: Final = {"MC_ON": "#0072B2", "MC_OFF": "#D55E00", "PC_ON": "#009E73", "PC_OFF": "#CC79A7"}
MARKERS: Final = {"MC_ON": "o", "MC_OFF": "s", "PC_ON": "^", "PC_OFF": "D"}


def save(figure: Figure, name: str) -> None:
    for extension in ("png", "pdf", "svg"):
        figure.savefig(OUT / "figures" / f"{name}.{extension}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    (OUT / "figures").mkdir(exist_ok=True)
    with (OUT / "per_cell_nll.csv").open(newline="", encoding="utf-8") as stream:
        cells = [Cell.model_validate(row) for row in csv.DictReader(stream)]
    analysis = json.loads((OUT / "analysis.json").read_text(encoding="utf-8"))
    data = np.array([c.values() for c in cells])
    legend = [Line2D([], [], color=COLORS[g], marker=MARKERS[g], linestyle="none", markersize=5,
                      label=f"{g.replace('_', ' ')} (n={sum(c.group == g for c in cells)})") for g in GROUPS]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=.08, right=.97, bottom=.24, top=.83, wspace=.55)
    ax = axes[0]
    for cell in cells:
        ax.plot(range(4), cell.values()[:4], color="#B9BEC3", linewidth=.55, alpha=.7, zorder=1)
        ax.scatter(range(4), cell.values()[:4], color=COLORS[cell.group], marker=MARKERS[cell.group], s=15, zorder=2)
    ax.plot(range(4), data[:, :4].mean(axis=0), color="#111111", marker="D", markersize=6, linewidth=2, zorder=3)
    ax.set_xticks(range(4), ["Constant", "LN", "CNN", "Canonical\nV1"])
    ax.set_ylabel("Validation Bernoulli NLL (nats/bin)")
    ax.set_title("A   Paired cell predictions", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=.18)
    ax.set_xlim(-.25, 3.25)
    ax.set_ylim(.28, .66)
    ax = axes[1]
    for index, row in enumerate(analysis["paired"]):
        ax.errorbar(row["mean"], 2-index,
                    xerr=[[row["mean"]-row["mean_ci_low"]], [row["mean_ci_high"]-row["mean"]]],
                    fmt="o", color="#252525", capsize=4, linewidth=1.6)
    ax.axvline(0, color="#9A9A9A", linestyle="--", linewidth=1)
    ax.set_yticks([2, 1, 0], ["CNN − LN", "CNN − Canonical V1", "Canonical V1 − LN"])
    ax.set_ylim(-.6, 2.6)
    ax.set_xlim(-.04, .035)
    ax.set_xlabel("Mean paired ΔNLL (first − second)")
    ax.set_title("B   Cell-paired bootstrap", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=.18)
    fig.suptitle("Frozen prediction comparison · 22 macaque cells", y=.97, fontsize=12)
    fig.legend(handles=legend, loc="lower center", ncol=4, bbox_to_anchor=(.5, .075), frameon=False)
    fig.text(.5, .018, "A: thin lines = cells; black diamonds = equal-cell means.  B: 95% percentile CI, 100,000 paired resamples.",
             ha="center", fontsize=8)
    save(fig, "main_prediction_paired")
    ordered = sorted(cells, key=lambda c: (GROUPS.index(c.group), cells.index(c)))
    fig, axes = plt.subplots(1, 3, figsize=(10.1, 8.6), sharey=True)
    fig.subplots_adjust(left=.20, right=.97, bottom=.12, top=.90, wspace=.13)
    limit = max(abs(c.values()[MODELS.index(a)]-c.values()[MODELS.index(b)]) for c in cells for a, b in PAIRS)*1.12
    for ax, (first, second), row in zip(axes, PAIRS, analysis["paired"], strict=True):
        for index, cell in enumerate(ordered):
            difference = cell.values()[MODELS.index(first)]-cell.values()[MODELS.index(second)]
            ax.hlines(index, 0, difference, color=COLORS[cell.group], linewidth=1.5)
            ax.scatter(difference, index, color=COLORS[cell.group], marker=MARKERS[cell.group], s=28, zorder=3)
        ax.axvline(0, color="#444444", linestyle="--", linewidth=.8)
        ax.set_xlim(-limit, limit)
        ax.set_xticks([-.10, -.05, 0, .05, .10])
        ax.set_xlabel("Paired ΔNLL (nats/bin)")
        ax.set_title(f"{first} − {second}\nWins: {row['first_wins']} / {row['second_wins']} (ties: {row['ties']})", fontsize=9)
        ax.grid(axis="x", alpha=.15)
        for boundary in (4.5, 8.5, 17.5):
            ax.axhline(boundary, color="#DDDDDD", linewidth=.7)
    axes[0].set_yticks(range(22), [f"{c.cell_id}   {c.group.replace('_', ' ')}" for c in ordered])
    axes[0].set_ylim(21.7, -.7)
    fig.suptitle("Per-cell paired differences", y=.98, fontsize=12)
    fig.text(.5, .03, "Negative = lower NLL for the first model. Each cell has one frozen fit per model; no per-cell uncertainty bars.",
             ha="center", fontsize=8)
    save(fig, "per_cell_paired_differences")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.7))
    fig.subplots_adjust(left=.10, right=.98, bottom=.24, top=.83, wspace=.36)
    for ax, comparator in zip(axes, ("LN", "Canonical V1"), strict=True):
        for cell in cells:
            ax.scatter(cell.values()[MODELS.index(comparator)], cell.sc_adapted_nll,
                       color=COLORS[cell.group], marker=MARKERS[cell.group], s=30)
        ax.plot([.28, .66], [.28, .66], "--", color="#888888", linewidth=1)
        ax.set(xlim=(.28, .66), ylim=(.28, .66), xlabel=f"{comparator} NLL (nats/bin)", ylabel="SC-adapted NLL (nats/bin)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"SC-adapted vs {comparator}")
    fig.suptitle("Supplement · frozen SC-adapted predictions", y=.97, fontsize=12)
    fig.legend(handles=legend, loc="lower center", ncol=4, bbox_to_anchor=(.5, .07), frameon=False)
    fig.text(.5, .015, "Identity line shown; points above it have higher SC-adapted NLL. Descriptive comparison only.", ha="center", fontsize=8)
    save(fig, "supplement_sc_adapted")
    print("Rendered 3 figures in PNG/PDF/SVG; no model or checkpoint loaded")


if __name__ == "__main__":
    main()
