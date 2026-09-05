from __future__ import annotations

from dataclasses import asdict
import json
import runpy

import torch

from common import OUT, SOURCE, TEMPORAL, CellIdentity, VerificationRow, infer, load_model
from probe import Interval, ProbeBank, verify_bank


def run_temporal(metadata: list[CellIdentity]) -> list[VerificationRow]:
    output = OUT / "temporal"
    output.mkdir()
    saved = torch.load(TEMPORAL / "inputs.pt", weights_only=True)
    old = torch.load(TEMPORAL / "responses.pt", weights_only=True)
    banks = {g: ProbeBank(**(b | {"intervals": tuple(tuple(Interval(**i) for i in events)
                                                    for events in b["intervals"])})) for g, b in saved.items()}
    responses, groups, checks = {}, {}, []
    for cell in metadata:
        cid, group = cell["cell_id"], f"{cell['retinal_class']}_{cell['polarity']}"
        bank = banks[group]
        verify_bank(bank)
        model = load_model(cid)
        assert torch.equal(bank.center, model.feature_bank.bc_support[0].bool())
        assert torch.equal(bank.annulus, model.feature_bank.ac_support[0].bool() & ~bank.center)
        history = torch.zeros(bank.drive.shape[:2] + (1,))
        response, check = infer(model, bank.drive, history)
        assert old["groups"][cid] == group
        responses[cid], groups[cid] = response, group
        checks.append({"cell_id": cid, "application": "temporal", "saved_inputs_used_exactly": True, **check})
        print(f"TEMPORAL {len(responses)}/22 {cid}", flush=True)
    torch.save({"cells": responses, "groups": groups}, output / "responses.pt")
    torch.save({g: asdict(b) | {"intervals": [[asdict(e) for e in es] for es in b.intervals]}
                for g, b in banks.items()}, output / "inputs.pt")
    report = runpy.run_path(str(TEMPORAL / "report.py"))
    report["COLORS"]["direct_BC_off"] = "#3a9853"
    report["create_report"](output, banks, responses)
    current = json.loads((output / "results.json").read_text())
    previous = json.loads((TEMPORAL / "results.json").read_text())
    keys = ("cell_id", "group", "probe", "mode", "channel")
    lookup = {tuple(r[k] for k in keys): r for r in previous["per_cell"]}
    numeric = tuple(k for k in previous["per_cell"][0] if k not in keys)
    comparisons = []
    for row in current["per_cell"]:
        key = tuple(row[k] for k in keys)
        if key not in lookup:
            continue
        reference = lookup[key]
        comparison = {k: row[k] for k in keys}
        for metric in numeric:
            comparison[f"current_{metric}"] = row[metric]
            comparison[f"previous_{metric}"] = reference[metric]
            comparison[f"delta_{metric}"] = None if row[metric] is None or reference[metric] is None else row[metric] - reference[metric]
        comparisons.append(comparison)
    group_comparisons = report["grouped"](comparisons)
    report["write_csv"](output / "per-cell-vs-overlapping.csv", comparisons)
    report["write_csv"](output / "group-vs-overlapping.csv", group_comparisons)
    protocol = json.loads((TEMPORAL / "protocol.json").read_text())
    protocol.update({"replayed_from_saved_inputs": str(TEMPORAL / "inputs.pt"),
                     "source": str(SOURCE),
                     "conditions": list(next(iter(responses.values()))), "generated_probe_families": 0,
                     "RMS_normalization": False, "delta_definition": "shared-BC minus overlapping-support"})
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (output / "comparison.json").write_text(json.dumps({"groups": group_comparisons,
        "per_cell": comparisons, "verification": checks}, indent=2, allow_nan=False))
    return checks
