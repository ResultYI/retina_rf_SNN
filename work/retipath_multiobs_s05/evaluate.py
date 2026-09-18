from __future__ import annotations

import csv
import itertools
import statistics
from pathlib import Path

import torch

from run import (ROOT, OUT, SEEDS, CONDITIONS, UPDATES, EPS, s0, read_json, write_json,
                 sha, write_csv, save_tensor, state_sha, utc, build, configure_runtime,
                 checked_protocol, load_s0, observations)


def rms(value):
    return float(value.double().square().mean().sqrt())


def corr(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    denominator = a.norm() * b.norm()
    return float((a * b).sum() / denominator) if denominator > 0 else None


def h_parameters(model):
    return {"H1_tau_ms": float(model.h1.tau_ms), "H1_delay_ms": float(model.h1.delay_ms),
            "H1_amplitude": float(model.gates.h1)}


def predict(model, data):
    return torch.cat([model(x, observed_counts=y).logits for x, y in
                      zip(data["x"].split(8), data["target"].split(8), strict=True)])


def latent(model, x, p):
    results = [observations(model, chunk, p) for chunk in x.split(8)]
    return {key: torch.cat([r[key] for r in results]) for key in ("state", "feedback")}


def mechanism(model, x, p):
    result = {key: [] for key in ("state", "feedback", "normal", "delta")}
    for chunk in x.split(8):
        zeros = torch.zeros((*chunk.shape[:2], 1))
        normal = s0.observe_mechanism(model, chunk, observed_counts=zeros,
                                     intervention=s0.InterventionSpec(s0.Intervention.FIX_HISTORY_ZERO))
        blocked = s0.observe_mechanism(model, chunk, observed_counts=zeros,
                                      intervention=s0.InterventionSpec(s0.Intervention.BLOCK_H1_FEEDBACK))
        for trace in (normal, blocked):
            assert not torch.count_nonzero(trace.tensors["history_state"])
        node, pixel = p["observation"]["state_node"], p["observation"]["feedback_pixel"]
        result["state"].append(normal.tensors["h1_state"][..., node:node+1])
        result["feedback"].append(normal.tensors["h1_feedback"][..., pixel:pixel+1])
        result["normal"].append(normal.logits)
        result["delta"].append(normal.logits - blocked.logits)
    return {key: torch.cat(value) for key, value in result.items()}


def aggregate(rows, field):
    values = [row[field] for row in rows]
    assert len(values) == 3 and all(v is not None for v in values)
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values),
            "values_by_seed": {str(row["seed"]): row[field] for row in rows}}


def comparison(first, second):
    result = {}
    for measure in ("state_nRMSE", "feedback_nRMSE", "H1_intervention_error", "expected_CE"):
        a, b = first[measure], second[measure]
        differences = {str(seed): a["values_by_seed"][str(seed)] - b["values_by_seed"][str(seed)] for seed in SEEDS}
        ratio = a["mean"] / b["mean"]
        result[measure] = {"mean_ratio": ratio, "mean_relative_reduction": 1 - ratio,
                           "paired_differences": differences, "all3_lower": all(v < 0 for v in differences.values()),
                           "clear_improvement": ratio <= .90 and all(v < 0 for v in differences.values())}
    return result


def adjudicate(conditions):
    a, b, c, d = (conditions[name] for name in CONDITIONS)
    comparisons = {name: comparison(left, right) for name, left, right in (
        ("B_vs_A", b, a), ("C_vs_A", c, a), ("C_vs_B", c, b),
        ("D_vs_A", d, a), ("D_vs_B", d, b), ("D_vs_C", d, c))}
    state_sufficient = all(comparisons["B_vs_A"][key]["clear_improvement"]
                           for key in ("state_nRMSE", "feedback_nRMSE", "H1_intervention_error"))
    b_state_constraint = (comparisons["B_vs_A"]["state_nRMSE"]["clear_improvement"] or
                          b["ambiguity"]["state"] <= .90 * a["ambiguity"]["state"])
    c_improves = all(comparisons[cmp][key]["clear_improvement"] for cmp in ("C_vs_A", "C_vs_B")
                     for key in ("feedback_nRMSE", "H1_intervention_error"))
    d_keeps = (all(comparisons[cmp][key]["clear_improvement"] for cmp in ("D_vs_A", "D_vs_B")
                  for key in ("feedback_nRMSE", "H1_intervention_error")) and
               all(d[key]["mean"] <= 1.05 * c[key]["mean"] for key in ("feedback_nRMSE", "H1_intervention_error")))
    feedback_required = b_state_constraint and c_improves and d_keeps
    verdict = ("STATE_SUPERVISION_SUFFICIENT" if state_sufficient else
               "FEEDBACK_OBSERVATION_REQUIRED" if feedback_required else "MIXED_OR_OPTIMIZATION_LIMITED")
    return verdict, comparisons, {"state_sufficient": state_sufficient, "B_state_constraint": b_state_constraint,
                                  "C_clearly_improves_feedback_and_intervention": c_improves,
                                  "D_retains_improvements_within_5percent_of_C": d_keeps}


def evaluate():
    p = checked_protocol()
    protocol_sha = sha(OUT / "protocol.json")
    complete = read_json(OUT / "training_complete.json")
    assert len(complete["fits"]) == 12 and complete["total_optimizer_updates"] == 36000
    assert complete["protocol_sha256"] == protocol_sha
    if (OUT / "EVALUATION_STARTED.json").exists():
        raise FileExistsError("Evaluation already started; no blind repeat or overwrite")
    finals, replications = {}, {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            path = OUT / "checkpoints" / f"{condition}_{seed}_final.pt"
            record = next(r for r in complete["fits"] if r["condition"] == condition and r["seed"] == seed)
            assert sha(path) == record["sha256"]
            cp = torch.load(path, weights_only=True)
            assert cp["step"] == UPDATES and cp["protocol_sha256"] == protocol_sha
            assert state_sha(cp["model"]) == cp["state_sha256"]
            assert {int(v["step"]) for v in cp["optimizer"]["state"].values()} == {UPDATES}
            finals[condition, seed] = cp
            if condition in ("RGC_ONLY", "RGC_STATE"):
                old_condition = "RGC_ONLY" if condition == "RGC_ONLY" else "RGC_H1"
                old_cp = torch.load(s0.OUT / "checkpoints" / f"{old_condition}_{seed}_final.pt", weights_only=True)
                replications[f"{condition}_{seed}"] = state_sha(old_cp["model"]) == cp["state_sha256"]
    assert all(replications.values()), "S0-identical A/B implementation did not exactly replicate"
    write_json(OUT / "EVALUATION_STARTED.json", {"started_utc": utc(), "all12_finals_frozen": True,
        "protocol_sha256": protocol_sha, "six_A_B_S0_replications_exact": replications})
    meta = read_json(s0.OUT / "teacher_metadata.json")
    teacher = build(meta, torch.load(ROOT / p["teacher"]["path"], weights_only=True)["model"]).eval().requires_grad_(False)
    before = state_sha(teacher.state_dict())
    norm = read_json(OUT / "normalization.json")
    h, r = load_s0("D_H_heldout"), load_s0("D_R_heldout")
    probability = torch.load(s0.OUT / "datasets/teacher_probabilities/D_R_heldout.pt", weights_only=True)
    bank = torch.cat([load_s0("mechanism_" + key)["x"] for key in ("H", "B", "R")])
    latent_rows, parameter_rows, intervention_rows, prediction_rows = [], [], [], []
    bank_responses = {}
    with torch.no_grad():
        target = latent(teacher, h["x"], p)
        assert torch.equal(target["state"], h["target"])
        reference_logits = predict(teacher, r)
        assert torch.equal(reference_logits.sigmoid(), probability)
        truth = mechanism(teacher, bank, p)
        teacher_parameters = h_parameters(teacher)
        target_rms = rms(truth["delta"][:, 30:])
        assert target_rms > EPS
        entropy = float(s0.expected_bernoulli_nll(reference_logits.double(), probability.double(), r["mask"]))
        teacher_nll = float(s0.expected_bernoulli_nll(reference_logits.double(), r["target"].double(), r["mask"]))
        save_tensor(OUT / "evaluation_arrays/teacher.pt", {"heldout": target, "mechanism": truth,
                    "RGC_logits": reference_logits, "RGC_probabilities": probability, "parameters": teacher_parameters})
        for condition in CONDITIONS:
            for seed in SEEDS:
                cp = finals[condition, seed]
                model = build(meta, cp["model"]).eval().requires_grad_(False)
                obs = latent(model, h["x"], p)
                response = mechanism(model, bank, p)
                bank_responses[condition, seed] = response
                logits = predict(model, r)
                for key in ("state", "feedback"):
                    pred, ref = obs[key][:, 30:], target[key][:, 30:]
                    latent_rows.append({"condition": condition, "seed": seed, "observable": key,
                        "normalized_RMSE": rms(pred - ref) / norm[key]["std"][0],
                        "correlation": corr(pred, ref), "dataset": "D_H_heldout", "scored_bins": 3840})
                values = h_parameters(model)
                for name, value in values.items():
                    ref = teacher_parameters[name]
                    parameter_rows.append({"condition": condition, "seed": seed, "parameter": name,
                        "teacher": ref, "student": value, "signed_error": value-ref,
                        "absolute_error": abs(value-ref), "relative_absolute_error": abs(value-ref)/(abs(ref)+EPS),
                        "role": "secondary_not_causal_mediation_proof"})
                error = rms(response["delta"][:, 30:] - truth["delta"][:, 30:])
                intervention_rows.append({"condition": condition, "seed": seed, "intervention": "BLOCK_H1_FEEDBACK",
                    "normalized_intervention_error": error/(target_rms+EPS), "absolute_RMS_error_logit": error,
                    "teacher_Delta_RMS_logit": target_rms, "student_Delta_RMS_logit": rms(response["delta"][:, 30:]),
                    "history": "FIX_HISTORY_ZERO", "role": "primary", "scored_bins": 11520})
                nll = float(s0.expected_bernoulli_nll(logits.double(), r["target"].double(), r["mask"]))
                ce = float(s0.expected_bernoulli_nll(logits.double(), probability.double(), r["mask"]))
                prediction_rows.append({"condition": condition, "seed": seed, "sampled_Bernoulli_NLL": nll,
                    "expected_CE": ce, "teacher_NLL": teacher_nll, "teacher_entropy": entropy,
                    "excess_CE": ce-entropy, "scored_bins": 3840})
                save_tensor(OUT / "evaluation_arrays" / f"{condition}_{seed}.pt", {
                    "heldout": obs, "mechanism": response, "RGC_logits": logits, "parameters": values})
                assert state_sha(model.state_dict()) == cp["state_sha256"]
                print("evaluated", condition, seed, flush=True)
        assert state_sha(teacher.state_dict()) == before
    ambiguity = []
    for condition in CONDITIONS:
        for first, second in itertools.combinations(SEEDS, 2):
            left, right = bank_responses[condition, first], bank_responses[condition, second]
            for key in ("state", "feedback", "delta"):
                scale = target_rms+EPS if key == "delta" else norm[key]["std"][0]
                ambiguity.append({"condition": condition, "measure": key, "seed_i": first, "seed_j": second,
                    "pairwise_normalized_RMS": rms(left[key][:, 30:]-right[key][:, 30:])/scale,
                    "aggregation": "single_pair", "bank": "same96sequence_mechanism_bank"})
        for key in ("state", "feedback", "delta"):
            rows = [r for r in ambiguity if r["condition"] == condition and r["measure"] == key]
            ambiguity.append({**rows[0], "seed_i": "", "seed_j": "", "aggregation": "mean_3_pairs",
                              "pairwise_normalized_RMS": statistics.mean(r["pairwise_normalized_RMS"] for r in rows)})
    summaries = {}
    for condition in CONDITIONS:
        lat = [r for r in latent_rows if r["condition"] == condition]
        iv = [r for r in intervention_rows if r["condition"] == condition]
        pred = [r for r in prediction_rows if r["condition"] == condition]
        params = [r for r in parameter_rows if r["condition"] == condition]
        summaries[condition] = {
            **{f"{key}_nRMSE": aggregate([r for r in lat if r["observable"] == key], "normalized_RMSE") for key in ("state", "feedback")},
            **{f"{key}_correlation": aggregate([r for r in lat if r["observable"] == key], "correlation") for key in ("state", "feedback")},
            "H1_intervention_error": aggregate(iv, "normalized_intervention_error"),
            "sampled_NLL": aggregate(pred, "sampled_Bernoulli_NLL"), "expected_CE": aggregate(pred, "expected_CE"),
            "parameters": {name: {"value": aggregate([r for r in params if r["parameter"] == name], "student"),
                "absolute_error": aggregate([r for r in params if r["parameter"] == name], "absolute_error")} for name in teacher_parameters},
            "ambiguity": {r["measure"]: r["pairwise_normalized_RMS"] for r in ambiguity
                          if r["condition"] == condition and r["aggregation"] == "mean_3_pairs"}}
    verdict, comparisons, gates = adjudicate(summaries)
    curves = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            with (OUT / "curves" / f"{condition}_{seed}.csv").open(encoding="utf-8") as stream:
                curves.extend(csv.DictReader(stream))
    outputs = {"training_curves.csv": curves, "heldout_state_feedback.csv": latent_rows,
        "h1_parameter_recovery.csv": parameter_rows, "intervention_recovery.csv": intervention_rows,
        "ambiguity.csv": ambiguity, "prediction.csv": prediction_rows}
    for name, rows in outputs.items():
        write_csv(OUT / name, rows)
    summary = {"status": "COMPLETE", "completed_utc": utc(), "protocol_sha256": protocol_sha,
        "verdict": verdict, "verdict_gate_results": gates, "conditions": summaries, "comparisons": comparisons,
        "prediction_preserved": {name: value["expected_CE"]["mean"] <= 1.01*summaries["RGC_ONLY"]["expected_CE"]["mean"]
                                 for name, value in summaries.items()},
        "teacher": {"parameters": teacher_parameters, "sampled_NLL": teacher_nll,
                    "expected_CE_entropy": entropy, "H1_Delta_RMS_logit": target_rms, "history_gate": float(teacher.gates.history)},
        "verification": {"12_fits_step3000": True, "36000_optimizer_updates": True,
            "six_S0_A_B_exact_replications": replications, "teacher_and_frozen_sources_unchanged": True,
            "teacher_RGC_probability_replay_exact": True, "new_stimuli_generated": 0,
            "D_B_or_D_R_EXTRA_loaded": False, "secondary_interventions_run": 0},
        "limits": ["one synthetic teacher,3 seeds, fixed finite budget", "not biological ground truth",
            "state target has no direct amplitude dependence; RGC loss can still constrain amplitude indirectly",
            "scalar feedback loss constrains amplitude together with distributed state return mapping; not amplitude-only supervision",
            "parameter-error/intervention-error association is not causal mediation evidence",
            "frozen96sequence mechanism bank retains its B-family evaluation slice; no D_B observation data used",
            "D supervises two scalar traces on the same stimulus, versus one in B/C; not information-matched",
            "no S1 or real physiology; no structural uniqueness claim from finite optimization"]}
    checked_protocol()
    write_json(OUT / "summary.json", summary)
    paths = [OUT / name for name in outputs] + [OUT / "summary.json"] + list((OUT / "evaluation_arrays").glob("*.pt"))
    write_json(OUT / "evaluation_manifest.json", {str(path.relative_to(ROOT)).replace(chr(92), "/"): sha(path) for path in paths})
    print("EVALUATION_COMPLETE", verdict, flush=True)


if __name__ == "__main__":
    configure_runtime()
    evaluate()
