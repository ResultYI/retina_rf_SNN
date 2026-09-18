from __future__ import annotations

import csv
import itertools
import math
import statistics

import torch

from run import OUT, SEEDS, CONDITIONS, INTERVENTIONS, BRANCHES, EPS, checked_protocol, read_json, sha, write_json, utc


def records(name):
    with (OUT / name).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def match(actual, expected):
    assert math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12), (actual, expected)


def rms(value):
    return value.double().square().mean().sqrt().item()


def correlation(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum().div(a.norm() * b.norm()).item()


def verify():
    checked_protocol()
    summary = read_json(OUT / "summary.json")
    assert summary["status"] == "COMPLETE"
    root = OUT.parents[2]
    for path, digest in read_json(OUT / "evaluation_manifest.json").items():
        assert sha(root / path) == digest
    norm = read_json(OUT / "normalization.json")
    target = {key: torch.load(OUT / f"datasets/D_{key}_heldout.pt", weights_only=True)["target"][:, 30:].double() for key in ("H", "B", "R")}
    p = torch.load(OUT / "datasets/teacher_probabilities/D_R_heldout.pt", weights_only=True)[:, 30:].double()
    teacher = torch.load(OUT / "evaluation_arrays/teacher_mechanism.pt", weights_only=True)
    arrays = {(c, s): torch.load(OUT / "evaluation_arrays" / f"{c}_{s}.pt", weights_only=True) for c in CONDITIONS for s in SEEDS}
    pred, latent, interventions, ambiguity = (records(name) for name in (
        "heldout_prediction.csv", "latent_recovery.csv", "intervention_recovery.csv", "ambiguity.csv"))
    assert len(pred) == 15 and len(latent) == 90 and len(interventions) == 60 and len(ambiguity) == 200
    for row in pred:
        logits = arrays[row["condition"], int(row["seed"])]["RGC_logits"][:, 30:].double()
        match(row["sampled_spike_NLL"], torch.nn.functional.binary_cross_entropy_with_logits(logits, target["R"]))
        match(row["expected_Bernoulli_CE"], torch.nn.functional.binary_cross_entropy_with_logits(logits, p))
    for row in latent:
        a = arrays[row["condition"], int(row["seed"])]
        if row["channel"] == "H1":
            values = a["H1_heldout"][:, 30:].double()
            nr = rms(values - target["H"]) / norm["H"]["std"][0]
            cr = correlation(values, target["H"])
        else:
            values = a["BC_heldout"][:, 30:].double()
            nr = [rms(values[..., i] - target["B"][..., i]) / norm["B"]["std"][i] for i in range(4)]
            cr = [correlation(values[..., i], target["B"][..., i]) for i in range(4)]
            if row["channel"] == "BC_mean4":
                nr, cr = statistics.mean(nr), statistics.mean(cr)
            else:
                index = BRANCHES.index(row["channel"])
                nr, cr = nr[index], cr[index]
        match(row["normalized_RMSE"], nr)
        match(row["correlation"], cr)
    for row in interventions:
        name = row["intervention"]
        a = arrays[row["condition"], int(row["seed"])]["mechanism_" + name][:, 30:]
        truth = teacher[name][:, 30:]
        match(row["normalized_intervention_error"], rms(a - truth) / (rms(truth) + EPS))
        match(row["absolute_RMS_error_logit"], rms(a - truth))
    for row in ambiguity:
        name, condition = row["measure"], row["condition"]
        seed_pairs = list(itertools.combinations(SEEDS, 2)) if row["aggregation"] == "mean_3_pairs" else [(int(row["seed_i"]), int(row["seed_j"]))]
        values = []
        for first, second in seed_pairs:
            a, b = arrays[condition, first], arrays[condition, second]
            if name in INTERVENTIONS:
                value = rms(a["mechanism_" + name][:, 30:] - b["mechanism_" + name][:, 30:]) / (rms(teacher[name][:, 30:]) + EPS)
            elif name == "H1":
                value = rms(a["mechanism_H1"][:, 30:] - b["mechanism_H1"][:, 30:]) / norm["H"]["std"][0]
            else:
                per_channel = [rms(a["mechanism_BC"][:, 30:, i] - b["mechanism_BC"][:, 30:, i]) / norm["B"]["std"][i] for i in range(4)]
                value = statistics.mean(per_channel) if name == "BC_mean4" else per_channel[BRANCHES.index(name)]
            values.append(value)
        match(row["pairwise_normalized_RMS"], statistics.mean(values))
    curves = records("training_curves.csv")
    assert len(curves) == 1815
    for condition in CONDITIONS:
        for seed in SEEDS:
            rows = [r for r in curves if r["condition"] == condition and int(r["seed"]) == seed]
            assert [int(r["step"]) for r in rows] == [1, *range(25, 3001, 25)]
            for r in rows:
                match(r["total_loss"], sum(float(r[k]) for k in ("L_H", "L_B", "L_R") if r[k]))
    manifest = read_json(OUT / "dataset_manifest.json")
    for name, row in manifest.items():
        assert sha(OUT / f"datasets/{name}.pt") == row["file_sha256"]
    result = {"status": "PASS", "verified_utc": utc(), "protocol_sha256": sha(OUT / "protocol.json"),
        "checks": ["saved-array replay of all prediction, latent, intervention and ambiguity CSV entries",
            "15fits x121 curve rows, final3000 and exact sum of active losses", "dataset and evidence hashes unchanged",
            "current frozen model/source/teacher hashes match"],
        "review": "executor separate formula replay; not independent-person audit"}
    write_json(OUT / "verification/final.json", result)
    print(result)


if __name__ == "__main__":
    torch.set_num_threads(1)
    verify()
