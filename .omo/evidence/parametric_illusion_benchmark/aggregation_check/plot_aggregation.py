#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib"]
# ///

from __future__ import annotations

import csv
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt


OUT: Final = Path(__file__).resolve().parent
GROUPS: Final = ("MC_ON", "MC_OFF", "PC_ON", "PC_OFF")
COLORS: Final = {"normal": "#222222", "AC_off": "#d95f02"}


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def group_curve_figure() -> None:
    rows = read_csv("group_surface_curves.csv")
    signatures = (
        ("bright_minus_dark", "SBC bright - dark"),
        ("dark_ramp_minus_uniform", "Mach dark ramp - uniform"),
        ("bright_ramp_minus_uniform", "Mach bright ramp - uniform"),
    )
    figure, axes = plt.subplots(3, 4, figsize=(14, 8.5), sharex="row")
    for column, group in enumerate(GROUPS):
        for row_index, (signature, title) in enumerate(signatures):
            axis = axes[row_index, column]
            for condition in ("normal", "AC_off"):
                selected = sorted(
                    (row for row in rows if row["group"] == group and row["condition"] == condition
                     and row["signature"] == signature and float(row["contrast"]) == 0.5),
                    key=lambda row: int(row["extent_px"]),
                )
                x = [int(row["extent_px"]) for row in selected]
                mean = [float(row["mean"]) for row in selected]
                low = [float(row["ci95_low"]) for row in selected]
                high = [float(row["ci95_high"]) for row in selected]
                axis.plot(x, mean, marker="o", color=COLORS[condition], label=condition)
                axis.fill_between(x, low, high, color=COLORS[condition], alpha=0.12)
            axis.axhline(0, color="#999999", linewidth=0.7)
            if row_index == 0:
                axis.set_title(group.replace("_", " "))
            if column == 0:
                axis.set_ylabel(f"{title}\npaired logit")
            if row_index == 2:
                axis.set_xlabel("spatial extent (px)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Canonical V1 group curves at contrast 0.5")
    figure.tight_layout()
    figure.savefig(OUT / "group_normal_acoff_curves.png", dpi=180)
    plt.close(figure)


def interaction_figure() -> None:
    rows = [row for row in read_csv("mach_control_subtracted_groups.csv")
            if row["metric"] == "AC_effect_interaction" and float(row["contrast"]) == 0.5]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    signatures = (("dark_ramp_minus_uniform", "Mach dark"),
                  ("bright_ramp_minus_uniform", "Mach bright"))
    for axis, (signature, title) in zip(axes, signatures, strict=True):
        for group in GROUPS:
            selected = sorted((row for row in rows if row["group"] == group and row["signature"] == signature),
                              key=lambda row: int(row["extent_px"]))
            x = [int(row["extent_px"]) for row in selected]
            mean = [float(row["mean"]) for row in selected]
            low = [float(row["ci95_low"]) for row in selected]
            high = [float(row["ci95_high"]) for row in selected]
            axis.plot(x, mean, marker="o", label=group.replace("_", " "))
            axis.fill_between(x, low, high, alpha=0.08)
        axis.axhline(0, color="#777777", linewidth=0.8)
        axis.set_title(f"{title}, contrast 0.5")
        axis.set_xlabel("ramp width (px)")
    axes[0].set_ylabel("AC interaction vs width-0 (logit)")
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(OUT / "mach_ac_interaction_by_group.png", dpi=180)
    plt.close(figure)


def main() -> None:
    group_curve_figure()
    interaction_figure()


if __name__ == "__main__":
    main()
