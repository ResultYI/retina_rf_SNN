#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "matplotlib", "numpy"]
# ///
# How to run: imported by run.py using the frozen repository environment.
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import statistics
from typing import Final, TypeAlias

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluation.mechanistic_retina.temporal_center_surround import ResponseSummary, summarize_response
from probe import ACTIVE_MS, CONTRAST, DT_MS, EVENT_MS, NAMES, ONSET_MS, REFERENCES, ProbeBank

Scalar: TypeAlias = str | int | float | None
Row: TypeAlias = dict[str, Scalar]
Responses: TypeAlias = dict[str, dict[str, dict[str, torch.Tensor]]]
COLORS: Final = {"normal": "#222222", "H1_off": "#2878b5", "AC_off": "#cf3e3e"}


def write_csv(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(trace: torch.Tensor, duration: float) -> ResponseSummary:
    return summarize_response(trace, dt_ms=DT_MS, center_onset_ms=ONSET_MS, surround_onset_ms=None,
                              pulse_duration_ms=duration, event_window_ms=EVENT_MS)


def grouped(rows: list[Row]) -> list[Row]:
    keys = ("group", "probe", "mode", "channel")
    buckets = {}
    for row in rows:
        for group in (row["group"], "all"):
            key = (group, row["probe"], row["mode"], row["channel"])
            buckets.setdefault(key, []).append(row)
    output = []
    for key, values in buckets.items():
        item = dict(zip(keys, key, strict=True)) | {"n_cells": len(values)}
        for name in rows[0]:
            if name not in {*keys, "cell_id"}:
                numbers = [v[name] for v in values if v[name] is not None]
                item[name] = statistics.fmean(numbers) if numbers else None
        output.append(item)
    return output


def create_report(out: Path, banks: dict[str, ProbeBank], responses: Responses) -> None:
    """Retain existing response metrics and report raw clamp differences."""
    groups = torch.load(out / "responses.pt", weights_only=True)["groups"]
    rows, events, stimulus_rows = [], [], []
    for cid, modes in responses.items():
        bank = banks[groups[cid]]
        active = (bank.time_ms >= ONSET_MS) & (bank.time_ms < ONSET_MS + ACTIVE_MS)
        for mode, values in modes.items():
            for channel in ("logit", "probability"):
                delta = values[channel] - values[channel][-1]
                for i, name in enumerate(NAMES):
                    duration = ACTIVE_MS if i < 6 else 50.0
                    summary = summarize(delta[i], duration)
                    center = summarize(delta[REFERENCES[i]], duration)
                    difference = values[channel][i] - modes["normal"][channel][i]
                    effect = summarize(difference, duration)
                    first_event = bank.intervals[i][0]
                    onset = summarize_response(delta[i], dt_ms=DT_MS, center_onset_ms=None,
                        surround_onset_ms=first_event.onset_ms, pulse_duration_ms=first_event.duration_ms, event_window_ms=EVENT_MS)
                    rows.append({"cell_id": cid, "group": groups[cid], "probe": name, "mode": mode, "channel": channel,
                        "peak_response": summary.peak_response, "peak_absolute_response": summary.peak_absolute_response,
                        "peak_latency_ms_from_trial_start": summary.peak_latency_ms,
                        "peak_latency_ms_from_onset": summary.peak_latency_ms - ONSET_MS,
                        "response_integral_seconds": summary.response_integral,
                        "peak_change_vs_center_only": summary.peak_response - center.peak_response,
                        "integral_change_vs_center_only": summary.response_integral - center.response_integral,
                        "center_onset_50ms": summary.center_onset_response, "center_offset_50ms": summary.center_offset_response,
                        "surround_first_onset_50ms": onset.surround_onset_response,
                        "surround_first_offset_50ms": onset.surround_offset_response,
                        "off_minus_normal_mean_abs_active": float(difference[active].abs().mean()),
                        "off_minus_normal_peak_signed": effect.peak_response,
                        "off_minus_normal_peak_absolute": effect.peak_absolute_response,
                        "off_minus_normal_integral_seconds": effect.response_integral})
                    for j, interval in enumerate(bank.intervals[i]):
                        event = summarize_response(delta[i], dt_ms=DT_MS, center_onset_ms=None,
                            surround_onset_ms=interval.onset_ms, pulse_duration_ms=interval.duration_ms, event_window_ms=EVENT_MS)
                        events.append({"cell_id": cid, "group": groups[cid], "probe": name, "mode": mode,
                            "channel": channel, "event_index": j, **asdict(interval),
                            "onset_response_50ms": event.surround_onset_response, "offset_response_50ms": event.surround_offset_response})
    for group, bank in banks.items():
        for i, name in enumerate(NAMES):
            wave = CONTRAST * bank.waveforms[i]
            stimulus_rows.append({"group": group, "probe": name, "center_cones": int(bank.center.sum()),
                "annulus_cones": int(bank.annulus.sum()), "large_field_cones": bank.center.numel(),
                "sampled_temporal_mean_contrast": float(wave.mean()), "sampled_min_contrast": float(wave.min()),
                "sampled_max_contrast": float(wave.max()), "sampled_rms_contrast_full_trial": float(wave.square().mean().sqrt()),
                "continuous_rms_contrast_full_trial": CONTRAST * (sum(e.duration_ms for e in bank.intervals[i]) / 3000) ** 0.5})
    summaries = grouped(rows)
    write_csv(out / "per-cell.csv", rows)
    write_csv(out / "group-summary.csv", summaries)
    write_csv(out / "per-event-onset-offset.csv", events)
    write_csv(out / "stimulus-statistics.csv", stimulus_rows)
    (out / "results.json").write_text(json.dumps({"cell_count": 22, "group_counts": {g: list(groups.values()).count(g) for g in banks},
        "groups": summaries, "stimulus_statistics": stimulus_rows, "per_cell": rows}, indent=2, allow_nan=False), encoding="utf-8")
    render_figures(out, banks, responses)
    lines = ["# Frozen H1 / AC temporal probe results", "", "Metric: mean absolute off-minus-normal response over 300–2300 ms.",
             "The large-field row is the requested spatial exception. Per-event onset/offset windows are 50 ms.", "",
             "| Group | Probe | H1 logit | AC logit | H1 probability | AC probability |", "|---|---|---:|---:|---:|---:|"]
    for group in ("all", "MC_ON", "MC_OFF", "PC_ON", "PC_OFF"):
        for name in NAMES:
            values = [next(r["off_minus_normal_mean_abs_active"] for r in summaries
                           if (r["group"], r["probe"], r["mode"], r["channel"]) == (group, name, mode, channel))
                      for channel in ("logit", "probability") for mode in ("H1_off", "AC_off")]
            lines.append(f"| {group} | {name} | " + " | ".join(f"{v:.9g}" for v in values) + " |")
    lines += ["", "[All per-cell metrics](per-cell.csv) · [Group metrics](group-summary.csv) · [Every event](per-event-onset-offset.csv)",
              "", "[Stimulus matching](stimulus-statistics.csv) · [Inputs](inputs.pt) · [Responses](responses.pt) · [Protocol](protocol.json)",
              "", "![Population effects](figures/effect-magnitude.png)", "", "![Population traces](figures/all.png)"]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def render_figures(out: Path, banks: dict[str, ProbeBank], responses: Responses) -> None:
    groups = torch.load(out / "responses.pt", weights_only=True)["groups"]
    figure_dir = out / "figures"
    figure_dir.mkdir()
    selections = {"all": list(responses)} | {g: [cid for cid in responses if groups[cid] == g] for g in banks}
    selections |= {cid.replace("#", "_"): [cid] for cid in responses}
    time = next(iter(banks.values())).time_ms.numpy()
    for name, ids in selections.items():
        fig, axes = plt.subplots(7, 3, figsize=(16, 16), sharex=True, layout="constrained")
        for i, probe_name in enumerate(NAMES):
            for mode, color in COLORS.items():
                for col, channel in enumerate(("logit", "probability")):
                    value = torch.stack([responses[cid][mode][channel][i] - responses[cid][mode][channel][-1] for cid in ids]).mean(0)
                    axes[i, col].plot(time, value.numpy(), color=color, lw=1, label=mode)
                effect = torch.stack([responses[cid][mode]["probability"][i] - responses[cid]["normal"]["probability"][i] for cid in ids]).mean(0)
                axes[i, 2].plot(time, effect.numpy(), color=color, lw=1, label=mode)
            for ax in axes[i]:
                ax.axhline(0, color="#888888", lw=0.4)
                ax.axvline(ONSET_MS, color="#aaaaaa", ls="--", lw=0.5)
                ax.set_title(probe_name, fontsize=9)
            axes[i, 0].set_ylabel("logit - blank")
            axes[i, 1].set_ylabel("probability - blank")
            axes[i, 2].set_ylabel("probability: off - normal")
        for ax in axes[-1]:
            ax.set_xlabel("time (ms)")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"{name} | n={len(ids)} | frozen Canonical V1")
        fig.savefig(figure_dir / f"{name}.png", dpi=130)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    active = (time >= ONSET_MS) & (time < ONSET_MS + ACTIVE_MS)
    x = np.arange(7)
    for ax, channel in zip(axes, ("logit", "probability"), strict=True):
        for j, mode in enumerate(("H1_off", "AC_off")):
            values = [float(torch.stack([(r[mode][channel][i] - r["normal"][channel][i])[active].abs().mean()
                                          for r in responses.values()]).mean()) for i in range(7)]
            ax.bar(x + (j - .5) * .35, values, .35, color=COLORS[mode], label=mode)
        ax.set_xticks(x, NAMES, rotation=35, ha="right")
        ax.set_ylabel(f"mean |off - normal| {channel}")
        ax.legend()
    fig.savefig(figure_dir / "effect-magnitude.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(7, 1, figsize=(12, 10), sharex=True, layout="constrained")
    bank = next(iter(banks.values()))
    for i, ax in enumerate(axes):
        ax.plot(time, CONTRAST * bank.waveforms[i].numpy(), lw=1)
        ax.set_ylabel(NAMES[i], fontsize=8)
        ax.set_ylim(-.28, .28)
    axes[-1].set_xlabel("time (ms); native-bin Weber contrast; polarity sign applied per cell")
    fig.savefig(figure_dir / "stimulus-waveforms.png", dpi=150)
    plt.close(fig)
