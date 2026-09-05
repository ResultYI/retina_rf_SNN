from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Final, TypeAlias

from common import OUT, TEMPORAL, ILLUSION, CLAMPS, sha
from probe import NAMES

Row: TypeAlias = dict[str, str]
GROUPS: Final = ("all", "MC_ON", "MC_OFF", "PC_ON", "PC_OFF")
SIGNATURES: Final = (
    ("Mach dark", "ramp_minus_matched_uniform_x-04"),
    ("Mach bright", "ramp_minus_matched_uniform_x+04"),
    ("SBC", "bright_surround_minus_dark_surround"),
    ("Hermann original", "intersection_minus_corridor"),
    ("Hermann diagnostic", "contour_rearrangement_A_minus_B"),
    ("White original", "on_bright_bar_minus_on_dark_bar"),
    ("White diagnostic", "remote_contour_rearrangement_A_minus_B"),
)


def read(path: Path) -> list[Row]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select(rows: list[Row], **keys: str) -> Row:
    matches = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    assert len(matches) == 1, keys
    return matches[0]


def number(value: str) -> str:
    return f"{float(value):.9g}" if value else "NA"


def table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers),
                      *("| " + " | ".join(row) + " |" for row in rows)])


def compare_csv(current: list[Row], previous: list[Row], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[Row]:
    old = {tuple(r[k] for k in keys): r for r in previous}
    rows = []
    for row in current:
        key = tuple(row[k] for k in keys)
        if key not in old:
            continue
        item = {k: row[k] for k in keys}
        for metric in metrics:
            item[f"current_{metric}"] = row[metric]
            item[f"previous_{metric}"] = old[key][metric]
            item[f"delta_{metric}"] = str(float(row[metric]) - float(old[key][metric]))
        rows.append(item)
    return rows


def supplemental_tables() -> None:
    keys = ("cell_id", "group", "probe", "mode", "channel", "event_index", "onset_ms", "duration_ms", "sign")
    events = compare_csv(read(OUT / "temporal/per-event-onset-offset.csv"),
        read(TEMPORAL / "per-event-onset-offset.csv"), keys, ("onset_response_50ms", "offset_response_50ms"))
    write(OUT / "temporal/per-event-vs-overlapping.csv", events)
    current = read(OUT / "illusion/mach-boundary-extrema.csv")
    previous = read(ILLUSION / "mach-boundary-extrema.csv")
    keys = ("cell_id", "group", "mode", "channel", "profile", "region")
    metrics = tuple(k for k in current[0] if k not in keys)
    differences = compare_csv(current, previous, keys, metrics)
    write(OUT / "illusion/mach-boundary-differences.csv", differences)
    for filename, source in (("mach-boundary-group.csv", current), ("mach-boundary-group-differences.csv", differences)):
        buckets: dict[tuple[str, ...], list[Row]] = {}
        for r in source:
            for group in (r["group"], "all"):
                buckets.setdefault((group, *(r[k] for k in keys[2:])), []).append(r)
        groups = []
        for key, rows in buckets.items():
            group_row = dict(zip(keys[1:], key, strict=True)) | {"n_cells": str(len(rows))}
            group_row.update({k: str(fmean(float(r[k]) for r in rows)) for k in rows[0] if k not in keys})
            groups.append(group_row)
        write(OUT / "illusion" / filename, groups)


def main() -> None:
    supplemental_tables()
    temporal = read(OUT / "temporal/group-summary.csv")
    temporal_delta = read(OUT / "temporal/group-vs-overlapping.csv")
    illusion = read(OUT / "illusion/group-responses.csv")
    illusion_delta = read(OUT / "illusion/group-metric-differences.csv")
    diagnostic = read(OUT / "illusion/group-comparisons.csv")
    lines = ["# Canonical V1 shared-BC frozen application replay", "",
        "22 cells: MC ON 5; MC OFF 4; PC ON 9; PC OFF 4. All group/population values are equal-cell means.", "",
        "Original saved stimulus tensors were loaded. Temporal: 7 probes + 2 existing center references + blank, 450 bins at 150 Hz. Illusion: 72 saved sequences, 150 bins at 150 Hz. No training, optimizer, new stimulus family or RMS normalization.", "",
        "Temporal mean absolute clamp effect uses the unchanged 300–2300 ms window. Signed peak/integral suppression–facilitation is the unchanged condition-minus-same-clamp-center-only difference. Illusion signed mean-on uses 300–400 ms. Peak, integral, latency and onset/offset definitions are unchanged.", "",
        "All version deltas below are shared-BC minus overlapping-support. direct-BC-off has no previous counterpart. Tau, explicit pathway delay, RF lag window and strictly-past history shift are unchanged and are not estimated here.", ""]
    for group in GROUPS:
        lines += [f"## {group}", "", "### Temporal clamp effects", ""]
        for channel in ("logit", "probability"):
            rows = []
            for probe in NAMES:
                vals = [number(select(temporal, group=group, probe=probe, mode=m, channel=channel)["off_minus_normal_mean_abs_active"])
                        for m in ("H1_off", "direct_BC_off", "AC_off")]
                diffs = [number(select(temporal_delta, group=group, probe=probe, mode=m, channel=channel)["delta_off_minus_normal_mean_abs_active"])
                         for m in ("H1_off", "AC_off")]
                rows.append([probe, *vals, *diffs])
            lines += [f"{channel}: mean absolute off−normal", "", table(["Probe", "H1-off", "direct-BC-off", "AC-off", "H1 effect Δ vs old", "AC effect Δ vs old"], rows), ""]
        lines += ["### Illusion signed response signatures", ""]
        for channel in ("logit", "probability"):
            rows = []
            for label, name in SIGNATURES:
                vals = [number(select(illusion, group=group, name=name, mode=m, channel=channel, kind="pair_difference")["mean_on"]) for m in CLAMPS]
                diffs = [number(select(illusion_delta, group=group, name=name, mode=m, channel=channel, kind="pair_difference")["delta_mean_on"])
                         for m in ("normal", "H1_off", "AC_off")]
                rows.append([label, *vals, *diffs])
            lines += [f"{channel}: signed mean-on pair difference", "", table(["Signature", "normal", "H1-off", "direct-BC-off", "AC-off", "normal Δ vs old", "H1-off Δ vs old", "AC-off Δ vs old"], rows), ""]
        lines += ["### Diagnostic minus original", ""]
        rows = []
        for family, name in (("White", "remote_contour_rearrangement"), ("Hermann", "contour_rearrangement")):
            for channel in ("logit", "probability"):
                vals = [number(select(diagnostic, group=group, name=name, mode=m, channel=channel, kind="diagnostic_minus_original")["mean_on"]) for m in CLAMPS]
                diffs = [number(select(diagnostic, group=group, name=name, mode=m, channel=channel, kind="shared_minus_overlapping_diagnostic_change")["mean_on"])
                         for m in ("normal", "H1_off", "AC_off")]
                rows.append([family, channel, *vals, *diffs])
        lines += [table(["Family", "Channel", "normal", "H1-off", "direct-BC-off", "AC-off", "normal Δ vs old", "H1-off Δ vs old", "AC-off Δ vs old"], rows), "",
            f"[Temporal figure](comparison-figures/temporal-{group}.png) · [Illusion figure](comparison-figures/illusion-{group}.png) · [Diagnostic figure](comparison-figures/diagnostic-{group}.png)", ""]
    controls = [r for r in illusion if r["kind"] == "pair_difference" and "control" in r["name"]]
    control_max = max(abs(float(r[k])) for r in controls for k in ("mean_on", "peak_absolute", "integral_response_seconds"))
    assert control_max == 0
    lines += ["## Controls and verification", "", f"All five paired contextual controls, all 22 cells and all four conditions: exact-zero difference (maximum |mean-on|, peak and integral = {control_max:g}). Mach matched-uniform controls and boundary extrema are retained separately in the full tables.", "",
        "22/22 checkpoint/source hashes matched. Both applications: 22/22 exact-zero H1/direct-BC/AC clamps; direct-BC-off preserves BC_broad and AC; AC-off preserves H1 and both BC views; H1-off propagates downstream; inference leaves state unchanged; all outputs finite. Normal reentry is bitwise identical. Previous illusion metric replay matched for 22/22 cells.", "",
        "## Complete numerical artifacts", "",
        "- Temporal: [per-cell all metrics](temporal/per-cell.csv), [population/four classes](temporal/group-summary.csv), [per-cell version deltas](temporal/per-cell-vs-overlapping.csv), [group version deltas](temporal/group-vs-overlapping.csv), [each onset/offset event](temporal/per-event-onset-offset.csv), [event version deltas](temporal/per-event-vs-overlapping.csv), [responses](temporal/responses.pt), [stimuli](temporal/inputs.pt).",
        "- Illusions: [per-cell signatures and clamp changes](illusion/per-cell-responses.csv), [population/four classes](illusion/group-responses.csv), [per-cell metric deltas](illusion/per-cell-metric-differences.csv), [group metric deltas](illusion/group-metric-differences.csv), [diagnostic differences](illusion/per-cell-comparisons.csv), [group diagnostic differences](illusion/group-comparisons.csv), [Mach boundary extrema](illusion/mach-boundary-extrema.csv), [Mach group extrema](illusion/mach-boundary-group.csv), [Mach version deltas](illusion/mach-boundary-differences.csv), [Mach group deltas](illusion/mach-boundary-group-differences.csv), [responses](illusion/responses.pt), [stimuli](illusion/inputs.pt).",
        "- Figures: [all population/group/per-cell figures](comparison-figures). Solid lines: shared-BC; dashed: previous overlapping-support.",
        "- [Verification](verification.json), [input/checkpoint manifest](input-manifest.json), [report provenance](report-manifest.json).", ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    paths = [*OUT.glob("*.py"), OUT / "RESULTS.md", *OUT.glob("temporal/*.csv"), *OUT.glob("illusion/*.csv")]
    (OUT / "report-manifest.json").write_text(json.dumps({"source_sha256": {str(p): sha(p) for p in paths},
        "inference_rerun": False, "training": False, "new_metrics": False}, indent=2))
    print("RESULTS.md and supplemental unchanged-metric comparison tables saved")


if __name__ == "__main__":
    main()
