from __future__ import annotations

import csv
import itertools
import math

import numpy as np
import torch

from run import (ROOT, OUT, CONDITIONS, SEEDS, EPS, read_json, write_json, checked_protocol,
                 load_s0, sha, utc)


def verify():
    checked_protocol()
    for name, digest in read_json(OUT / "evaluation_manifest.json").items():
        assert sha(ROOT / name) == digest, name
    norm = read_json(OUT / "normalization.json")
    truth = torch.load(OUT / "evaluation_arrays/teacher.pt", weights_only=True)
    all_arrays = {(c, s): torch.load(OUT / "evaluation_arrays" / f"{c}_{s}.pt", weights_only=True)
                  for c in CONDITIONS for s in SEEDS}
    def arr(tensor):
        return tensor[:, 30:].double().numpy()
    def rms(x):
        return float(np.sqrt(np.mean(np.square(x))))
    def equal(actual, expected):
        assert math.isclose(float(actual), float(expected), rel_tol=2e-6, abs_tol=2e-8), (actual, expected)
    def rows(name):
        with (OUT / name).open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    target_rms = rms(arr(truth["mechanism"]["delta"]))
    for row in rows("heldout_state_feedback.csv"):
        c, seed, key = row["condition"], int(row["seed"]), row["observable"]
        pred, target = arr(all_arrays[c, seed]["heldout"][key]), arr(truth["heldout"][key])
        equal(row["normalized_RMSE"], rms(pred-target)/norm[key]["std"][0])
        equal(row["correlation"], np.corrcoef(pred.ravel(), target.ravel())[0, 1])
    for row in rows("intervention_recovery.csv"):
        pred = arr(all_arrays[row["condition"], int(row["seed"])]["mechanism"]["delta"])
        error = rms(pred-arr(truth["mechanism"]["delta"]))
        equal(row["normalized_intervention_error"], error/(target_rms+EPS))
        equal(row["absolute_RMS_error_logit"], error)
        equal(row["teacher_Delta_RMS_logit"], target_rms)
    r = load_s0("D_R_heldout")
    for row in rows("prediction.csv"):
        logits = arr(all_arrays[row["condition"], int(row["seed"])]["RGC_logits"])
        for field, target in (("sampled_Bernoulli_NLL", r["target"]), ("expected_CE", truth["RGC_probabilities"])):
            equal(row[field], np.mean(np.logaddexp(0, logits)-arr(target)*logits))
    for row in rows("h1_parameter_recovery.csv"):
        key = row["parameter"]
        target = truth["parameters"][key]
        value = all_arrays[row["condition"], int(row["seed"])]["parameters"][key]
        equal(row["teacher"], target)
        equal(row["student"], value)
        equal(row["absolute_error"], abs(value-target))
        equal(row["relative_absolute_error"], abs(value-target)/(abs(target)+EPS))
    ambiguity = rows("ambiguity.csv")
    for row in ambiguity:
        c, key = row["condition"], row["measure"]
        pairs = itertools.combinations(SEEDS, 2) if row["aggregation"] == "mean_3_pairs" else [(int(row["seed_i"]), int(row["seed_j"]))]
        values = []
        for first, second in pairs:
            left, right = (arr(all_arrays[c, seed]["mechanism"][key]) for seed in (first, second))
            denominator = target_rms+EPS if key == "delta" else norm[key]["std"][0]
            values.append(rms(left-right)/denominator)
        equal(row["pairwise_normalized_RMS"], np.mean(values))
    curves = rows("training_curves.csv")
    for c in CONDITIONS:
        for seed in SEEDS:
            values = [r for r in curves if r["condition"] == c and int(r["seed"]) == seed]
            assert len(values) == 121 and int(values[-1]["step"]) == 3000
            for row in values:
                expected = sum(float(row[k]) for k in ("L_state", "L_feedback", "L_R") if row[k])
                equal(row["total_loss"], expected)
                assert bool(row["L_state"]) == ("state" in CONDITIONS[c])
                assert bool(row["L_feedback"]) == ("feedback" in CONDITIONS[c])
    summary = read_json(OUT / "summary.json")
    for c in CONDITIONS:
        select = lambda name: [r for r in rows(name) if r["condition"] == c]
        checks = {"state_nRMSE": [float(r["normalized_RMSE"]) for r in select("heldout_state_feedback.csv") if r["observable"] == "state"],
                  "feedback_nRMSE": [float(r["normalized_RMSE"]) for r in select("heldout_state_feedback.csv") if r["observable"] == "feedback"],
                  "H1_intervention_error": [float(r["normalized_intervention_error"]) for r in select("intervention_recovery.csv")],
                  "expected_CE": [float(r["expected_CE"]) for r in select("prediction.csv")]}
        for key, values in checks.items():
            equal(summary["conditions"][c][key]["mean"], np.mean(values))
    a, b, c, d = (summary["conditions"][key] for key in CONDITIONS)
    def clear(left, right, key):
        return (left[key]["mean"] <= .90*right[key]["mean"] and
                all(left[key]["values_by_seed"][str(s)] < right[key]["values_by_seed"][str(s)] for s in SEEDS))
    sufficient = all(clear(b, a, k) for k in ("state_nRMSE", "feedback_nRMSE", "H1_intervention_error"))
    required = ((clear(b, a, "state_nRMSE") or b["ambiguity"]["state"] <= .90*a["ambiguity"]["state"]) and
                all(clear(c, ref, k) and clear(d, ref, k) for ref in (a, b) for k in ("feedback_nRMSE", "H1_intervention_error")) and
                all(d[k]["mean"] <= 1.05*c[k]["mean"] for k in ("feedback_nRMSE", "H1_intervention_error")))
    verdict = "STATE_SUPERVISION_SUFFICIENT" if sufficient else "FEEDBACK_OBSERVATION_REQUIRED" if required else "MIXED_OR_OPTIMIZATION_LIMITED"
    assert summary["verdict"] == verdict
    result = {"status": "PASS", "verified_utc": utc(), "protocol_sha256": sha(OUT / "protocol.json"),
              "checks": ["saved-array NumPy formula replay of all six CSVs", "12 fits x121 curve rows, final3000, active loss sum",
                         "summary aggregates and frozen verdict independently recomputed", "S0 inputs and frozen source hashes unchanged"],
              "review": "executor separate formula replay; not independent-person review"}
    write_json(OUT / "verification/final.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    verify()
