from __future__ import annotations

import csv
import json
import statistics

import torch

from replay import OLD, OUT


def norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.double()))


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = norm(left) * norm(right)
    return float((left.double() * right.double()).sum()) / denominator if denominator else 0.0


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_rows(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    output = []
    for group in ("ALL", "MC_ON", "MC_OFF", "PC_ON", "PC_OFF"):
        subset = rows if group == "ALL" else [r for r in rows if r["group"] == group]
        for combination in sorted({tuple(row[key] for key in keys) for row in subset}):
            selected = [r for r in subset if tuple(r[key] for key in keys) == combination]
            summary = {"group": group, **dict(zip(keys, combination)), "cells": len(selected)}
            for key, value in selected[0].items():
                if key not in {"group", "cell_id", *keys} and isinstance(value, (float, int)):
                    summary[key] = statistics.fmean(r[key] for r in selected if r[key] is not None)
                elif value is None:
                    summary[key] = None
            output.append(summary)
    return output


def main() -> None:
    replay = json.loads((OUT / "replay-results.json").read_text())
    source = json.loads((OLD / "results.json").read_text())
    groups = {r["cell_id"]: f"{r['retinal_class']}_{r['polarity']}" for r in source["cells"]}
    old_rf = torch.load(OLD / "evaluation/rf-tensors.pt", weights_only=True)
    new_rf = torch.load(OUT / "rf-tensors.pt", weights_only=True)
    old_p = torch.load(OLD / "evaluation/perturbation-tensors.pt", weights_only=True)
    new_p = torch.load(OUT / "perturbation-tensors.pt", weights_only=True)
    old_t = json.loads((OLD / "temporal_center_surround_perturbation/results.json").read_text())["rows"]
    new_t = replay["rows"]
    rf_rows, perturbation_rows, temporal_rows = [], [], []
    for cid, group in groups.items():
        assert torch.allclose(sum(new_rf[cid][p] for p in ("H1", "BC", "AC")), new_rf[cid]["global"], atol=1e-7)
        for pathway, left in old_rf[cid].items():
            right = new_rf[cid][pathway]
            rf_rows.append({"cell_id": cid, "group": group, "pathway": pathway,
                            "old_norm": norm(left), "new_norm": norm(right),
                            "change_norm": norm(right-left), "old_new_cosine": cosine(left, right),
                            "new_old_norm_ratio": norm(right)/norm(left)})
        for clamp, old in old_p[cid].items():
            new = new_p[cid][clamp]
            row = {"cell_id": cid, "group": group, "clamp": clamp}
            for label, tensors, rf in (("old", old, old_rf), ("new", new, new_rf)):
                for response in ("logit", "probability"):
                    delta = tensors[f"{response}_delta"].double()
                    row[f"{label}_signed_{response}"] = float(delta.mean())
                    row[f"{label}_abs_{response}"] = float(delta.abs().mean())
                row[f"{label}_rf_change_norm"] = norm(tensors["rf_delta"])
                row[f"{label}_normal_clamp_rf_cosine"] = cosine(rf[cid]["global"], tensors["clamped_rf"])
            row["old_new_logit_effect_cosine"] = cosine(old["logit_delta"], new["logit_delta"])
            perturbation_rows.append(row)
    old_lookup = {(r["cell_id"], r["condition"], r["mode"]): r for r in old_t}
    for new in new_t:
        old = old_lookup[new["cell_id"], new["condition"], new["mode"]]
        row = {key: new[key] for key in ("cell_id", "group", "condition", "mode")}
        for key, value in new.items():
            if isinstance(value, (int, float)) and key != "offset_ms":
                row[f"old_{key}"] = old[key]
                row[f"new_{key}"] = value
                row[f"change_{key}"] = value-old[key]
            elif key not in {*row, "offset_ms"}:
                row[f"old_{key}"] = row[f"new_{key}"] = row[f"change_{key}"] = None
        temporal_rows.append(row)
    rf_summary = mean_rows(rf_rows, ("pathway",))
    perturbation_summary = mean_rows(perturbation_rows, ("clamp",))
    temporal_summary = mean_rows(temporal_rows, ("condition", "mode"))
    for name, rows in (("rf-comparison.csv", rf_rows), ("rf-group-comparison.csv", rf_summary),
                       ("perturbation-comparison.csv", perturbation_rows), ("perturbation-group-comparison.csv", perturbation_summary),
                       ("temporal-comparison.csv", temporal_rows), ("temporal-group-comparison.csv", temporal_summary)):
        write_csv(name, rows)
    pattern_rows = []
    for mode in ("normal", "H1_off", "AC_off"):
        for condition in sorted({r["condition"] for r in new_t}):
            selected = [r for r in temporal_rows if r["mode"] == mode and r["condition"] == condition]
            for metric in ("peak_response_probability", "integral_change_vs_center_only"):
                pattern_rows.append({"mode": mode, "condition": condition, "metric": metric,
                                     **{f"{label}_{sign}": sum(test(r[f"{label}_{metric}"]) for r in selected)
                                        for label in ("old", "new")
                                        for sign, test in (("positive", lambda x: x > 0), ("negative", lambda x: x < 0), ("zero", lambda x: x == 0))},
                                     "sign_reversal_cells": [r["cell_id"] for r in selected if r[f"old_{metric}"]*r[f"new_{metric}"] < 0]})
    clamp_sign = {clamp: {"signed_logit_reversal_cells": [r["cell_id"] for r in perturbation_rows
                    if r["clamp"] == clamp and r["old_signed_logit"]*r["new_signed_logit"] < 0],
                    "abs_effect_increased_cells": sum(r["new_abs_logit"] > r["old_abs_logit"] for r in perturbation_rows if r["clamp"] == clamp)}
                  for clamp in ("H1_off", "BC_off", "AC_off")}
    payload = {"comparison": "historical 50-step R4 versus R4-dev; equal cell weighting",
               "rf": rf_summary, "perturbation": perturbation_summary, "temporal": temporal_summary,
               "temporal_sign_counts": pattern_rows, "clamp_sign_changes": clamp_sign,
               "rf_nonzero_cells": {p: {label: sum(norm(rf[cid][p]) > 0 for cid in groups)
                   for label, rf in (("old", old_rf), ("new", new_rf))} for p in ("global", "H1", "BC", "AC")},
               "old_replay_checks": replay["checks"]}
    (OUT / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"rf": [r for r in rf_summary if r["group"] == "ALL"],
                      "perturbation": [r for r in perturbation_summary if r["group"] == "ALL"],
                      "clamp_sign_changes": clamp_sign,
                      "temporal_sign_counts": [r for r in pattern_rows if r["mode"] == "normal"]}, indent=2))


if __name__ == "__main__":
    main()
