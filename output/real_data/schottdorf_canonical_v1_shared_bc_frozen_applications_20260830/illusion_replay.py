from __future__ import annotations

import csv
import json

import torch

from common import OUT, ILLUSION, DIAGNOSTIC, ORIGINAL, CellIdentity, VerificationRow, infer, load_model
from metrics import Row, aggregate, cell_rows, mach_rows, response_metrics, write_csv
from stimuli import Pair, Stimuli


def metric_differences(current: list[Row], previous: list[Row]) -> list[Row]:
    identity = ("cell_id", "group", "mode", "channel", "family", "name", "kind")
    reference = {tuple(r.get(k) for k in identity): r for r in previous}
    names = ("mean_on", "peak_signed", "peak_absolute", "peak_latency_from_onset_ms",
             "integral_response_seconds", "onset_50ms_mean", "offset_50ms_mean")
    rows = []
    for r in current:
        key = tuple(r.get(k) for k in identity)
        if key not in reference:
            continue
        old = reference[key]
        row = {k: r[k] for k in identity if k in r}
        for metric in names:
            row[f"current_{metric}"] = r[metric]
            row[f"previous_{metric}"] = old[metric]
            row[f"delta_{metric}"] = None if r[metric] is None or old[metric] is None else r[metric] - old[metric]
        rows.append(row)
    return rows


def run_illusion(metadata: list[CellIdentity]) -> list[VerificationRow]:
    output = OUT / "illusion"
    output.mkdir()
    inputs = torch.load(DIAGNOSTIC / "inputs.pt", weights_only=True)
    original = torch.load(ORIGINAL / "input-tensors.pt", weights_only=True)
    scenes = torch.load(ORIGINAL / "stimuli.pt", weights_only=True)
    contract = json.loads((ORIGINAL / "stimulus-contract.json").read_text())
    protocol = json.loads((DIAGNOSTIC / "protocol.json").read_text())
    bank = Stimuli(tuple(inputs["names"]), inputs["patches"], tuple(Pair(**p) for p in protocol["pairs"]),
                   scenes["scenes"], contract["crop_centers_pixels"])
    assert torch.equal(inputs["original_drive"], original["cone_drive"])
    assert torch.equal(inputs["original_history"], original["history"])
    drive = torch.cat((inputs["original_drive"], inputs["diagnostic_drive"]))
    history = torch.cat((inputs["original_history"], inputs["diagnostic_history"]))
    time = inputs["time_ms"]
    old = torch.load(ILLUSION / "response-tensors.pt", weights_only=True)
    assert tuple(old["names"]) == bank.names and torch.equal(old["time_ms"], time)
    assert drive.shape == (72, 150, 289) and history.shape == (72, 150, 1)
    cells, rows, old_rows, comparisons, boundaries, checks = {}, [], [], [], [], []
    with (ILLUSION / "per-cell-responses.csv").open(newline="") as stream:
        saved_rows = list(csv.DictReader(stream))
    row_keys = ("cell_id", "mode", "channel", "name", "kind")
    old_lookup = {tuple(r[k] for k in row_keys): r for r in saved_rows}
    for cell in metadata:
        cid, group = cell["cell_id"], f"{cell['retinal_class']}_{cell['polarity']}"
        assert old["metadata"][cid]["group"] == group
        model = load_model(cid)
        response, check = infer(model, drive, history)
        for values in response.values():
            assert all(torch.equal(values["logit"][p.a], values["logit"][p.b]) for p in bank.pairs if p.control)
        cells[cid] = response
        rows.extend(cell_rows(cid, group, bank, response, time))
        reference_rows = cell_rows(cid, group, bank, old["cells"][cid], time)
        for r in reference_rows:
            saved = old_lookup[tuple(r[k] for k in row_keys)]
            for k in response_metrics(time.new_zeros(time.shape), time):
                assert (r[k] is None and saved[k] == "") or (r[k] is not None and abs(r[k] - float(saved[k])) <= 1e-12)
        old_rows.extend(reference_rows)
        boundaries.extend(mach_rows(cid, group, bank, response, time))
        for mode, values in response.items():
            for variant in protocol["variants"]:
                family, name = variant["family"], variant["variant"]
                for control in (False, True):
                    suffix = "control_A_minus_B" if control else "A_minus_B"
                    pair = next(p for p in bank.pairs if p.name == f"{name}_{suffix}")
                    a, b = variant["original_A"] + 2 * control, variant["original_B"] + 2 * control
                    for channel in ("logit", "probability"):
                        trace = values[channel][pair.a] - values[channel][pair.b] - (values[channel][a] - values[channel][b])
                        common = {"cell_id": cid, "group": group, "mode": mode, "channel": channel,
                                  "family": family, "name": name + ("_control" if control else ""), "control": control}
                        comparisons.append(common | {"kind": "diagnostic_minus_original"} | response_metrics(trace, time))
                        if mode in old["cells"][cid]:
                            old_trace = old["cells"][cid][mode][channel]
                            previous = old_trace[pair.a] - old_trace[pair.b] - (old_trace[a] - old_trace[b])
                            comparisons.append(common | {"kind": "previous_diagnostic_minus_original"} | response_metrics(previous, time))
                            comparisons.append(common | {"kind": "shared_minus_overlapping_diagnostic_change"} | response_metrics(trace - previous, time))
            if mode in old["cells"][cid]:
                for p in bank.pairs:
                    for channel in ("logit", "probability"):
                        previous = old["cells"][cid][mode][channel]
                        trace = values[channel][p.a] - values[channel][p.b] - (previous[p.a] - previous[p.b])
                        comparisons.append({"cell_id": cid, "group": group, "mode": mode, "channel": channel,
                            "family": p.family, "name": p.name, "kind": "shared_minus_overlapping_pair_trace"} | response_metrics(trace, time))
        checks.append({"cell_id": cid, "application": "illusion", "saved_inputs_used_exactly": True,
                       "old_metric_replay_matched": True, "paired_controls_exact_zero": True, **check})
        print(f"ILLUSION {len(cells)}/22 {cid}", flush=True)
    groups = aggregate(rows)
    previous_groups = aggregate(old_rows)
    comparison_groups = aggregate(comparisons)
    for name, values in (("per-cell-responses.csv", rows), ("group-responses.csv", groups),
                         ("per-cell-comparisons.csv", comparisons), ("group-comparisons.csv", comparison_groups),
                         ("mach-boundary-extrema.csv", boundaries),
                         ("per-cell-metric-differences.csv", metric_differences(rows, old_rows)),
                         ("group-metric-differences.csv", metric_differences(groups, previous_groups))):
        write_csv(output / name, values)
    torch.save({"cells": cells, "metadata": old["metadata"], "time_ms": time, "names": bank.names}, output / "responses.pt")
    torch.save({"cone_drive": drive, "history": history, "time_ms": time, "names": bank.names,
                "patches": bank.patches, "pairs": protocol["pairs"]}, output / "inputs.pt")
    (output / "results.json").write_text(json.dumps({"groups": groups, "comparisons": comparison_groups,
        "verification": checks, "conditions": list(next(iter(cells.values()))), "generated_stimuli": 0,
        "original_stimulus_contract": str(ORIGINAL / "stimulus-contract.json"),
        "diagnostic_contract": str(DIAGNOSTIC / "protocol.json")}, indent=2, allow_nan=False))
    return checks
