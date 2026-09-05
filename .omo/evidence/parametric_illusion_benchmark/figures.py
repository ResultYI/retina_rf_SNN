from __future__ import annotations

import csv
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np


OUT: Final = Path(__file__).resolve().parent
FIGURES: Final = OUT / "figures"
MODELS: Final = ("canonical", "ln", "cnn")
LABELS: Final = {"canonical": "Canonical V1", "ln": "Center-surround LN", "cnn": "Causal CNN"}
FilterValue = str | int | float


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def subset(rows: list[dict[str, str]], **filters: FilterValue) -> list[dict[str, str]]:
    return [row for row in rows if all(row[key] == str(value) for key, value in filters.items())]


def plot_cohort_curves() -> None:
    rows = read_csv("cohort_curves.csv")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, 5))
    for axis, model in zip(axes, MODELS, strict=True):
        values = subset(rows, model=model, mode="normal", family="SBC", signature="bright_minus_dark")
        for color, extent in zip(colors, (0, 2, 4, 6, 8), strict=True):
            line = sorted(subset(values, extent_px=extent), key=lambda row: float(row["contrast"]))
            x = np.asarray([float(row["contrast"]) for row in line])
            y = np.asarray([float(row["mean_paired_logit"]) for row in line])
            lo = np.asarray([float(row["bootstrap_ci95_low"]) for row in line])
            hi = np.asarray([float(row["bootstrap_ci95_high"]) for row in line])
            axis.plot(x, y, marker="o", color=color, label=f"{extent} px")
            axis.fill_between(x, lo, hi, color=color, alpha=0.12)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_title(LABELS[model]); axis.set_xlabel("Surround contrast magnitude")
    axes[0].set_ylabel("Bright − dark surround paired logit")
    axes[-1].legend(title="Extent", fontsize=8)
    figure.suptitle("SBC cohort curves (22-cell paired bootstrap 95% CI)")
    figure.tight_layout(); figure.savefig(FIGURES / "cohort_sbc_normal.png", dpi=180); plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey="row")
    colors = plt.cm.plasma(np.linspace(0.05, 0.95, 6))
    signatures = ("dark_ramp_minus_uniform", "bright_ramp_minus_uniform")
    for row_index, signature in enumerate(signatures):
        for axis, model in zip(axes[row_index], MODELS, strict=True):
            values = subset(rows, model=model, mode="normal", family="Mach", signature=signature)
            for color, width in zip(colors, (0, 2, 4, 8, 12, 16), strict=True):
                line = sorted(subset(values, extent_px=width), key=lambda row: float(row["contrast"]))
                x = np.asarray([float(row["contrast"]) for row in line])
                y = np.asarray([float(row["mean_paired_logit"]) for row in line])
                lo = np.asarray([float(row["bootstrap_ci95_low"]) for row in line])
                hi = np.asarray([float(row["bootstrap_ci95_high"]) for row in line])
                axis.plot(x, y, marker="o", color=color, label=f"{width} px")
                axis.fill_between(x, lo, hi, color=color, alpha=0.1)
            axis.axhline(0, color="black", linewidth=0.7); axis.set_title(LABELS[model])
            axis.set_xlabel("Ramp contrast magnitude")
        axes[row_index, 0].set_ylabel(signature.replace("_", " ") + "\npaired logit")
    axes[0, -1].legend(title="Ramp width", fontsize=8)
    figure.suptitle("Mach cohort curves (22-cell paired bootstrap 95% CI)")
    figure.tight_layout(); figure.savefig(FIGURES / "cohort_mach_normal.png", dpi=180); plt.close(figure)


def grid(rows: list[dict[str, str]], family: str, signature: str, value: str) -> np.ndarray:
    contrasts = (0, 0.0625, 0.125, 0.25, 0.375, 0.5)
    extents = (0, 2, 4, 6, 8) if family == "SBC" else (0, 2, 4, 8, 12, 16)
    lookup = {(float(row["contrast"]), int(row["extent_px"])): float(row[value]) for row in rows
              if row["family"] == family and row["signature"] == signature}
    return np.asarray([[lookup[(contrast, extent)] for contrast in contrasts] for extent in extents])


def plot_clamp_surfaces() -> None:
    rows = read_csv("cohort_clamp_differences.csv")
    clamps = ("H1_off_minus_normal", "direct_BC_off_minus_normal", "AC_off_minus_normal")
    for family, signatures in (("SBC", ("bright_minus_dark",)),
                               ("Mach", ("dark_ramp_minus_uniform", "bright_ramp_minus_uniform"))):
        figure, axes = plt.subplots(len(signatures), 3, figsize=(13, 3.8 * len(signatures)), squeeze=False)
        matrices = [[grid(subset(rows, model=clamp), family, signature, "mean_paired_logit_difference")
                     for clamp in clamps] for signature in signatures]
        limit = max(float(np.abs(matrix).max()) for group in matrices for matrix in group)
        for row_index, signature in enumerate(signatures):
            for axis, clamp, matrix in zip(axes[row_index], clamps, matrices[row_index], strict=True):
                image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
                axis.set_title(clamp.replace("_minus_normal", "")); axis.set_xlabel("contrast index")
                axis.set_ylabel((signature + "\nextent index").replace("_", " "))
        figure.colorbar(image, ax=axes.ravel().tolist(), label="off − normal paired logit")
        figure.suptitle(f"Canonical {family} clamp effects across full parameter grid")
        figure.savefig(FIGURES / f"canonical_{family.lower()}_clamp_surfaces.png", dpi=180, bbox_inches="tight")
        plt.close(figure)


def plot_per_cell() -> None:
    rows = read_csv("per_cell_curves.csv")
    destination = FIGURES / "per_cell"; destination.mkdir(parents=True, exist_ok=True)
    cells = sorted({row["cell_id"] for row in rows})
    definitions = (("SBC", "bright_minus_dark"), ("Mach", "dark_ramp_minus_uniform"),
                   ("Mach", "bright_ramp_minus_uniform"))
    for cell in cells:
        figure, axes = plt.subplots(3, 3, figsize=(12, 10))
        for row_index, (family, signature) in enumerate(definitions):
            matrices = [grid(subset(rows, cell_id=cell, model=model, mode="normal"), family, signature,
                             "paired_logit") for model in MODELS]
            limit = max(float(np.abs(matrix).max()) for matrix in matrices)
            for axis, model, matrix in zip(axes[row_index], MODELS, matrices, strict=True):
                axis.imshow(matrix, origin="lower", aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
                axis.set_title(LABELS[model]); axis.set_xlabel("contrast index")
                axis.set_ylabel((family + " " + signature + "\nextent index").replace("_", " "))
        figure.suptitle(f"{cell}: paired-logit parameter surfaces")
        figure.tight_layout(); figure.savefig(destination / f"{cell.replace('#', '_')}.png", dpi=150); plt.close(figure)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    plot_cohort_curves(); plot_clamp_surfaces(); plot_per_cell()
    print("WROTE cohort and 22 per-cell figures")


if __name__ == "__main__":
    main()
