#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "matplotlib", "numpy"]
# ///
# How to run: D:/anaconda/python.exe -B figures.py after frozen replay.
from __future__ import annotations

from typing import Final

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from common import OUT, TEMPORAL, ILLUSION, CLAMPS
from probe import NAMES

COLORS: Final = {"normal": "#222222", "H1_off": "#2878b5", "direct_BC_off": "#3a9853", "AC_off": "#cf3e3e"}
PAIRS: Final = (("Mach dark", 8, 33), ("Mach bright", 16, 41), ("SBC", 50, 51),
               ("Hermann original", 54, 55), ("Hermann diagnostic", 67, 68),
               ("White original", 58, 59), ("White diagnostic", 63, 64))


def main() -> None:
    torch.set_num_threads(2)
    temporal = torch.load(OUT / "temporal/responses.pt", weights_only=True)
    temporal_old = torch.load(TEMPORAL / "responses.pt", weights_only=True)
    inputs = torch.load(OUT / "temporal/inputs.pt", weights_only=True)
    illusion = torch.load(OUT / "illusion/responses.pt", weights_only=True)
    illusion_old = torch.load(ILLUSION / "response-tensors.pt", weights_only=True)
    groups = temporal["groups"]
    selections = {"all": list(groups)} | {g: [c for c in groups if groups[c] == g]
                                         for g in ("MC_ON", "MC_OFF", "PC_ON", "PC_OFF")}
    selections |= {cid.replace("#", "_"): [cid] for cid in groups}
    folder = OUT / "comparison-figures"
    folder.mkdir(exist_ok=True)
    for name, ids in selections.items():
        fig, axes = plt.subplots(7, 2, figsize=(13, 16), sharex=True, layout="constrained")
        time = next(iter(inputs.values()))["time_ms"]
        for i, probe in enumerate(NAMES):
            for j, channel in enumerate(("logit", "probability")):
                ax = axes[i, j]
                for label, saved, style in (("shared-BC", temporal, "-"), ("overlap", temporal_old, "--")):
                    for mode in CLAMPS:
                        if mode == "normal" or mode not in saved["cells"][ids[0]]:
                            continue
                        trace = torch.stack([saved["cells"][cid][mode][channel][i] - saved["cells"][cid]["normal"][channel][i] for cid in ids]).mean(0)
                        ax.plot(time, trace, style, color=COLORS[mode], lw=1, label=f"{label} {mode}")
                ax.axhline(0, color="#999999", lw=.5)
                ax.axvspan(300, 2300, color="gray", alpha=.08)
                ax.set_title(f"{probe} | {channel}", fontsize=9)
                ax.set_ylabel("off - normal")
        for ax in axes[-1]:
            ax.set_xlabel("time (ms)")
        axes[0, 0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"{name} | n={len(ids)} | temporal frozen replay; no RMS normalization")
        fig.savefig(folder / f"temporal-{name}.png", dpi=120)
        plt.close(fig)
        time = illusion["time_ms"]
        means = {label: {mode: {channel: torch.stack([saved["cells"][cid][mode][channel] for cid in ids]).mean(0)
                                 for channel in ("logit", "probability")}
                        for mode in saved["cells"][ids[0]]}
                 for label, saved in (("shared-BC", illusion), ("overlap", illusion_old))}
        fig, axes = plt.subplots(7, 2, figsize=(13, 16), sharex=True, layout="constrained")
        for i, (pair_name, a, b) in enumerate(PAIRS):
            for j, channel in enumerate(("logit", "probability")):
                ax = axes[i, j]
                for label, style in (("shared-BC", "-"), ("overlap", "--")):
                    for mode, channels in means[label].items():
                        ax.plot(time, channels[channel][a] - channels[channel][b], style,
                                color=COLORS[mode], lw=1, label=f"{label} {mode}")
                ax.axhline(0, color="#999999", lw=.5)
                ax.axvspan(300, 400, color="gray", alpha=.1)
                ax.set_xlim(250, 1000)
                ax.set_title(f"{pair_name} | {channel}", fontsize=9)
                ax.set_ylabel("matched response difference")
        for ax in axes[-1]:
            ax.set_xlabel("time (ms)")
        axes[0, 0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"{name} | n={len(ids)} | shared-BC vs overlapping-support")
        fig.savefig(folder / f"illusion-{name}.png", dpi=120)
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, layout="constrained")
        for i, (family, a, b, da, db) in enumerate((("White", 58, 59, 63, 64), ("Hermann", 54, 55, 67, 68))):
            for j, channel in enumerate(("logit", "probability")):
                ax = axes[i, j]
                for label, style in (("shared-BC", "-"), ("overlap", "--")):
                    for mode, channels in means[label].items():
                        v = channels[channel]
                        ax.plot(time, (v[da] - v[db]) - (v[a] - v[b]), style,
                                color=COLORS[mode], lw=1, label=f"{label} {mode}")
                ax.axhline(0, color="#999999", lw=.5)
                ax.axvspan(300, 400, color="gray", alpha=.1)
                ax.set_xlim(250, 1000)
                ax.set_title(f"{family} | {channel}")
                ax.set_ylabel("diagnostic - original pair difference")
                ax.set_xlabel("time (ms)")
        axes[0, 0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"{name} | n={len(ids)} | diagnostic minus original")
        fig.savefig(folder / f"diagnostic-{name}.png", dpi=130)
        plt.close(fig)
        print(f"FIGURES {name}", flush=True)


if __name__ == "__main__":
    main()
