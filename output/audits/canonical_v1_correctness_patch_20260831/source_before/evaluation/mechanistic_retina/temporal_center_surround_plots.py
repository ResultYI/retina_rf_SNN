from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from evaluation.mechanistic_retina.temporal_center_surround import (
    CenterSurroundProbeConfig,
)
from evaluation.mechanistic_retina.temporal_center_surround_reporting import CellTrace


_COLORS: Final = {"normal": "#202020", "H1_off": "#377eb8", "AC_off": "#e41a1c"}


def save_figures(
    traces: list[CellTrace],
    output_dir: Path,
    config: CenterSurroundProbeConfig,
) -> None:
    cells_dir = output_dir / "cells"
    groups_dir = output_dir / "groups"
    cells_dir.mkdir(parents=True, exist_ok=True)
    groups_dir.mkdir(parents=True, exist_ok=True)
    for trace in traces:
        _save_trace_figure((trace,), cells_dir / f"{trace.cell_id.replace('#', '_')}.png", config)
    for group in sorted({trace.group for trace in traces}):
        selected = tuple(trace for trace in traces if trace.group == group)
        _save_trace_figure(selected, groups_dir / f"{group}.png", config)


def _save_trace_figure(
    traces: tuple[CellTrace, ...],
    path: Path,
    config: CenterSurroundProbeConfig,
) -> None:
    reference = traces[0]
    figure, axes = plt.subplots(
        len(reference.condition_names),
        1,
        figsize=(10, 14),
        sharex=True,
        constrained_layout=True,
    )
    for condition_index, (axis, condition) in enumerate(
        zip(axes, reference.condition_names, strict=True)
    ):
        for mode, color in _COLORS.items():
            values = torch.stack(
                tuple(trace.probability_delta[mode][condition_index] for trace in traces)
            )
            mean = values.mean(dim=0)
            axis.plot(reference.time_ms, mean, color=color, linewidth=1.5, label=mode)
            if len(traces) > 1:
                error = values.std(dim=0, unbiased=False)
                axis.fill_between(
                    reference.time_ms,
                    mean - error,
                    mean + error,
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
        center_onset, surround_onset = _event_onsets(condition, config)
        if center_onset is not None:
            axis.axvline(center_onset, color="#4daf4a", linestyle="--", linewidth=0.8)
            axis.axvline(
                center_onset + config.pulse_duration_ms,
                color="#4daf4a",
                linestyle=":",
                linewidth=0.8,
            )
        if surround_onset is not None:
            axis.axvline(surround_onset, color="#984ea3", linestyle="--", linewidth=0.8)
            axis.axvline(
                surround_onset + config.pulse_duration_ms,
                color="#984ea3",
                linestyle=":",
                linewidth=0.8,
            )
        axis.axhline(0, color="#888888", linewidth=0.5)
        axis.set_ylabel("Δp")
        axis.set_title(condition.replace("_", " "), fontsize=9)
    axes[0].legend(ncol=3, frameon=False, loc="upper right")
    axes[-1].set_xlabel("time (ms)")
    title = reference.cell_id if len(traces) == 1 else f"{reference.group} mean ± SD"
    figure.suptitle(f"{title}: frozen temporal center-surround perturbation")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _event_onsets(
    condition: str,
    config: CenterSurroundProbeConfig,
) -> tuple[float | None, float | None]:
    center = config.center_onset_ms
    events = {
        "center_only": (center, None),
        "surround_only": (None, center),
        "surround_then_center_100ms": (center, center - 100.0),
        "surround_then_center_50ms": (center, center - 50.0),
        "center_surround_simultaneous": (center, center),
        "center_then_surround_50ms": (center, center + 50.0),
        "center_then_surround_100ms": (center, center + 100.0),
    }
    return events[condition]


__all__ = ["save_figures"]
