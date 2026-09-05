from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

from stimuli import DT_MS, ONSET_MS, DURATION_MS, PITCH_DEG, Stimuli, build_stimuli

COLORS = {"normal": "#222222", "H1_off": "#2477bb", "AC_off": "#cd5134"}
FAMILIES = ("Mach_bands", "SBC", "Hermann", "White")


def stimulus_figure(bank: Stimuli, out: Path) -> None:
    fig, axes = plt.subplots(4, 5, figsize=(15, 12), constrained_layout=True)
    for row, family in enumerate(FAMILIES):
        pairs = [p for p in bank.pairs if p.family == family]
        if family == "Mach_bands":
            ids = (8, 16, 33, 41)
        else:
            ids = (pairs[0].a, pairs[0].b, pairs[1].a, pairs[1].b)
        scene = bank.scenes[family]
        axes[row, 0].imshow(scene + 1, cmap="gray", vmin=0.75, vmax=1.25,
                             extent=(-32.5, 32.5, -32.5, 32.5))
        for index, (x, y) in enumerate(bank.crop_centers_pixels[family]):
            axes[row, 0].add_patch(Rectangle((x - 8.5, y - 8.5), 17, 17, fill=False,
                                           edgecolor=("#cd5134", "#2477bb")[index], linewidth=1))
        axes[row, 0].set_title(f"{family}: full scene")
        axes[row, 0].set_xlabel("scene x (pixels)")
        for column, idx in enumerate(ids, 1):
            axes[row, column].imshow(bank.patches[idx] + 1, cmap="gray", vmin=0.75, vmax=1.25,
                                      extent=(-8.5 * PITCH_DEG, 8.5 * PITCH_DEG,
                                              -8.5 * PITCH_DEG, 8.5 * PITCH_DEG))
            axes[row, column].set_title(bank.names[idx].replace("_", "\n"), fontsize=9)
            axes[row, column].set_xlabel("local x (deg)")
            axes[row, column].set_ylabel("local y (deg)")
    fig.suptitle("Fixed luminance probes and target-matched controls | relative L+M: 0.75 / 1 / 1.25\n"
                 "Each inference receives only the shown 17x17 local crop; 300-400 ms exposure", fontsize=13)
    fig.savefig(out / "stimuli-and-controls.png", dpi=160)
    plt.close(fig)


def response_figure(bank: Stimuli, responses: dict[str, dict[str, torch.Tensor]], time: torch.Tensor,
                    title: str, path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), constrained_layout=True)
    active = (time >= ONSET_MS) & (time < ONSET_MS + DURATION_MS)
    for row, family in enumerate(FAMILIES):
        pairs = [p for p in bank.pairs if p.family == family]
        for col, channel in enumerate(("logit", "probability")):
            ax = axes[row, col]
            for mode, color in COLORS.items():
                value = responses[mode][channel]
                if family == "Mach_bands":
                    x = torch.arange(-12, 13) * PITCH_DEG
                    baseline = value[-1, active].mean()
                    ax.plot(x, value[:25, active].mean(dim=1) - baseline, color=color, label=mode)
                    ax.plot(x, value[25:50, active].mean(dim=1) - baseline, "--", color=color)
                    for edge in (-4 * PITCH_DEG, 4 * PITCH_DEG):
                        ax.axvline(edge, color="gray", alpha=0.3, linewidth=0.5)
                    ax.set_xlabel("scan x (deg); solid ramp / dashed uniform control")
                else:
                    pair, control = pairs
                    ax.plot(time, value[pair.a] - value[pair.b], color=color, label=mode)
                    ax.plot(time, value[control.a] - value[control.b], "--", color=color)
                    ax.axvspan(ONSET_MS, ONSET_MS + DURATION_MS, color="gray", alpha=0.12)
                    ax.set_xlim(250, 1000)
                    ax.set_xlabel("time (ms); dashed = matched-control pair")
            ylabel = "mean-on response minus blank" if family == "Mach_bands" else "A minus B"
            ax.set_ylabel(f"{channel}: {ylabel}")
            ax.set_title(f"{family}: " + ("spatial profile" if family == "Mach_bands" else pairs[0].name))
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def mach_time_figure(cells: dict[str, dict[str, dict[str, torch.Tensor]]], metadata: dict[str, dict[str, str]],
                     time: torch.Tensor, out: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(12, 11), constrained_layout=True)
    for row, group in enumerate(("MC_ON", "MC_OFF", "PC_ON", "PC_OFF")):
        ids = [cid for cid in cells if metadata[cid]["group"] == group]
        for col, channel in enumerate(("logit", "probability")):
            ax = axes[row, col]
            for mode, color in COLORS.items():
                for index, label, style in ((8, "dark junction", "-"), (16, "bright junction", "--")):
                    trace = torch.stack([cells[cid][mode][channel][index] -
                                         cells[cid][mode][channel][index + 25] for cid in ids]).mean(dim=0)
                    ax.plot(time, trace, style, color=color, label=f"{mode}: {label}")
            ax.axvspan(300, 400, color="gray", alpha=0.12)
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.set_xlim(250, 1000)
            ax.set_title(f"{group} n={len(ids)} | {channel}")
            ax.set_xlabel("time (ms)")
            ax.set_ylabel("ramp minus matched uniform response")
            ax.legend(fontsize=7, ncol=2)
    fig.savefig(out / "mach-boundary-timecourses.png", dpi=150)
    plt.close(fig)


def report(out: Path) -> None:
    bank = build_stimuli()
    saved = torch.load(out / "response-tensors.pt", weights_only=True)
    cells, metadata, time = saved["cells"], saved["metadata"], saved["time_ms"]
    result = json.loads((out / "results.json").read_text())
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    for cid, responses in cells.items():
        response_figure(bank, responses, time, f"{cid} | {metadata[cid]['group']} | frozen R4-dev",
                        figdir / f"cell-{cid.replace('#', '_')}.png")
    for group in ("all", "MC_ON", "MC_OFF", "PC_ON", "PC_OFF"):
        ids = [cid for cid in cells if group == "all" or metadata[cid]["group"] == group]
        responses = {mode: {channel: torch.stack([cells[cid][mode][channel] for cid in ids]).mean(dim=0)
                            for channel in ("logit", "probability")} for mode in COLORS}
        response_figure(bank, responses, time, f"{group} | n={len(ids)} | frozen R4-dev",
                        figdir / f"group-{group}.png")
    mach_time_figure(cells, metadata, time, figdir)
    lines = ["# Frozen R4-dev visual-probe model outputs", "",
             "22 cells: MC ON 5, MC OFF 4, PC ON 9, PC OFF 4. 150 Hz; 17x17 L+M; 100 ms pulse at 300 ms; identical zero observed-spike history.",
             "", "No training or model changes. All 22 checkpoints/state dictionaries unchanged; H1/AC clamps exact-zero; normal re-entry bitwise equal.",
             "", "Differences below are mean over 300<=t<400 ms, then equal-cell mean. Positive/negative denotes A-B, not perceived brightness. Probability is per 6.667 ms bin.",
             "", "Mach numbers are junction response minus matched uniform luminance response (not themselves an overshoot claim). Other pairs: SBC bright-dark surround; Hermann intersection-corridor; White bright-bar target minus dark-bar target.",
             "", "## Mean-on signatures", "",
             "| Group | Stimulus/pair | Channel | normal | H1-off | AC-off |", "|---|---|---|---:|---:|---:|"]
    selected = (("Mach_bands", "ramp_minus_matched_uniform_x-04", "Mach dark junction"),
                ("Mach_bands", "ramp_minus_matched_uniform_x+04", "Mach bright junction"),
                ("SBC", "bright_surround_minus_dark_surround", "SBC"),
                ("Hermann", "intersection_minus_corridor", "Hermann"),
                ("White", "on_bright_bar_minus_on_dark_bar", "White"))
    for group in ("all", "MC_ON", "MC_OFF", "PC_ON", "PC_OFF"):
        for family, name, label in selected:
            for channel in ("logit", "probability"):
                values = [next(r["mean_on"] for r in result["group_metrics"] if r["group"] == group
                               and r["mode"] == mode and r["family"] == family and r["name"] == name
                               and r["channel"] == channel and r["kind"] == "pair_difference") for mode in COLORS]
                lines.append(f"| {group} | {label} | {channel} | " + " | ".join(f"{v:+.9f}" for v in values) + " |")
    lines.extend(["", "## Boundary excursions", "", "Computed in fixed x=-6..-2 / +2..+6-pixel regions against both remote plateau mean-on responses. Positive/negative excursions use response units, not an illusion score.",
                  "", "| Mode | Profile | Channel | Cells with above-plateau excursion | Cells with below-plateau excursion | Mean cell max above | Mean cell min below |",
                  "|---|---|---|---:|---:|---:|---:|"])
    for mode in COLORS:
        for profile in ("ramp", "matched_uniform"):
            for channel in ("logit", "probability"):
                rows = [r for r in result["mach_boundary_extrema"] if r["mode"] == mode and r["profile"] == profile and r["channel"] == channel]
                hi = [max(r["overshoot_above_plateaus"] for r in rows if r["cell_id"] == cid) for cid in cells]
                lo = [min(r["undershoot_below_plateaus"] for r in rows if r["cell_id"] == cid) for cid in cells]
                lines.append(f"| {mode} | {profile} | {channel} | {sum(v > 1e-9 for v in hi)}/22 | {sum(v < -1e-9 for v in lo)}/22 | {sum(hi)/22:.9f} | {sum(lo)/22:.9f} |")
    lines.extend(["", "## Controls and artifacts", "",
                  "SBC, Hermann and White control A-B traces are bitwise zero for every cell and mode. Mach uniform-control responses and their plateau excursions are reported separately.",
                  "", "Per-cell signed mean, signed/absolute peak, peak time, onset/offset, integral and clamp-minus-normal: per-cell-responses.csv. Group means and direction counts: group-responses.csv. Full raw logits/probabilities and AC currents: response-tensors.pt. All individual spatial profiles/time courses: figures/cell-*.png.",
                  "", "Stimulus definitions and matching boundaries: stimulus-contract.json and stimuli-and-controls.png. Reproducibility and immutable-source checks: verification.json."])
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    report(Path(__file__).resolve().parent)
