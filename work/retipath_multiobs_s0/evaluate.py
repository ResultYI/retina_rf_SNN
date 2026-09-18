from __future__ import annotations

import csv
import itertools
import statistics
from pathlib import Path

import torch

from run import (ROOT, OUT, SEEDS, CONDITIONS, INTERVENTIONS, BRANCHES, EPS, UPDATES,
    build, checked_protocol, read_json, write_json, write_csv, save_tensor, sha, state_sha,
    h_observation, b_observation, configure_runtime, utc,
    observe_mechanism, Intervention, InterventionSpec)
from training.mechanistic_retina.losses import expected_bernoulli_nll
from evaluation.mechanistic_retina.clean_sampled_data import _sample_spikes


def rms(value):
    return float(value.double().square().mean().sqrt())


def corr(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    denominator = a.norm() * b.norm()
    return float((a * b).sum() / denominator) if denominator > 0 else None


def parameters(model):
    values = {"H1_tau_ms": float(model.h1.tau_ms), "H1_delay_ms": float(model.h1.delay_ms),
        "H1_amplitude": float(model.gates.h1), "alpha": float(model.alpha),
        "G_E": float(model.canonical_gains.G_E), "G_I": float(model.canonical_gains.G_I),
        "w_E_sustained": float(model.canonical_gains.w_E[0, 0]),
        "w_I_local": float(model.canonical_gains.w_I[0, 0])}
    for branch, name in enumerate(("sustained", "transient")):
        for basis in range(3):
            values[f"BC_{name}_tau{basis+1}_ms"] = float(model.feature_bank.tau_ms[branch, basis])
        values[f"BC_{name}_delay_ms"] = float(model.feature_bank.delay_ms[branch])
    for index, name in enumerate(("local", "transient")):
        values[f"AC_{name}_tau_ms"] = float(model.amacrine.tau_ms.flatten()[index])
        values[f"AC_{name}_delay_ms"] = float(model.amacrine.delay_ms.flatten()[index])
    return values


def predict(model, data):
    return torch.cat([model(x, observed_counts=y).logits
        for x, y in zip(data["x"].split(8), data["target"].split(8), strict=True)])


def mechanism(model, x, node):
    output = {"H1": [], "BC": [], "NORMAL": [], **{name: [] for name in INTERVENTIONS}}
    for chunk in x.split(8):
        zeros = torch.zeros((*chunk.shape[:2], 1))
        normal = observe_mechanism(model, chunk, observed_counts=zeros,
                                  intervention=InterventionSpec(Intervention.FIX_HISTORY_ZERO))
        assert not torch.count_nonzero(normal.tensors["history_state"])
        output["H1"].append(normal.tensors["h1_state"][..., node:node+1])
        output["BC"].append(torch.cat((normal.tensors["bc_direct_effective_drive"].sum(-2),
                                      normal.tensors["bc_broad_effective_drive"].sum(-2)), -1).squeeze(2))
        output["NORMAL"].append(normal.logits)
        for name in INTERVENTIONS:
            altered = observe_mechanism(model, chunk, observed_counts=zeros,
                                       intervention=InterventionSpec(Intervention(name)))
            assert not torch.count_nonzero(altered.tensors["history_state"])
            output[name].append(normal.logits - altered.logits)
    return {key: torch.cat(value) for key, value in output.items()}


def latent_rows(condition, seed, label, prediction, target, std):
    prediction, target = prediction[:, 30:].double(), target[:, 30:].double()
    names = ("H1",) if label == "H1" else BRANCHES
    rows = []
    for channel, name in enumerate(names):
        rows.append({"condition": condition, "seed": seed, "dataset": f"D_{'H' if label == 'H1' else 'B'}_heldout",
            "channel": name, "normalized_RMSE": rms(prediction[..., channel] - target[..., channel]) / float(std[channel]),
            "correlation": corr(prediction[..., channel], target[..., channel]), "scored_bins": 32 * 120})
    if label == "BC":
        rows.append({**rows[0], "channel": "BC_mean4", "normalized_RMSE": statistics.mean(r["normalized_RMSE"] for r in rows),
                     "correlation": statistics.mean(r["correlation"] for r in rows) if all(r["correlation"] is not None for r in rows) else None})
    return rows


def evaluate():
    p = checked_protocol()
    complete = read_json(OUT / "training_complete.json")
    assert len(complete["fits"]) == 15 and complete["total_optimizer_updates"] == 45000
    protocol_sha = sha(OUT / "protocol.json")
    assert complete["protocol_sha256"] == protocol_sha
    if (OUT / "EVALUATION_STARTED.json").exists():
        raise FileExistsError("Evaluation already started; inspect existing artifacts before any repeat")
    finals = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            path = OUT / "checkpoints" / f"{condition}_{seed}_final.pt"
            expected = next(r["sha256"] for r in complete["fits"] if r["condition"] == condition and r["seed"] == seed)
            assert sha(path) == expected
            cp = torch.load(path, weights_only=True)
            assert cp["step"] == UPDATES and cp["protocol_sha256"] == protocol_sha
            assert state_sha(cp["model"]) == cp["state_sha256"]
            assert {int(value["step"]) for value in cp["optimizer"]["state"].values()} == {UPDATES}
            finals[condition, seed] = cp
    write_json(OUT / "EVALUATION_STARTED.json", {"started_utc": utc(), "all_15_final_checkpoints_frozen": True,
        "protocol_sha256": protocol_sha, "evaluation_code_sha256": sha(Path(__file__))})
    meta = read_json(OUT / "teacher_metadata.json")
    tcp = torch.load(ROOT / p["teacher"]["path"], weights_only=True)
    teacher = build(meta, tcp["model"]).eval().requires_grad_(False)
    before = state_sha(teacher.state_dict())
    node = p["observation"]["h1_node_index"]
    norm = {key: torch.tensor(value["std"], dtype=torch.float64) for key, value in read_json(OUT / "normalization.json").items()}
    data = {key: torch.load(OUT / f"datasets/D_{key}_heldout.pt", weights_only=True) for key in ("H", "B", "R")}
    probability = torch.load(OUT / "datasets/teacher_probabilities/D_R_heldout.pt", weights_only=True)
    bank = torch.cat([torch.load(OUT / f"datasets/mechanism_{key}.pt", weights_only=True)["x"] for key in ("H", "B", "R")])
    evaluation_path = OUT / "evaluation_arrays"
    evaluation_path.mkdir()
    prediction_rows, latent, parameter_rows, interventions = [], [], [], []
    all_mechanism = {}
    with torch.no_grad():
        reference_logits = predict(teacher, data["R"])
        assert torch.equal(reference_logits.sigmoid(), probability)
        teacher_nll = float(expected_bernoulli_nll(reference_logits.double(), data["R"]["target"].double(), data["R"]["mask"]))
        entropy = float(expected_bernoulli_nll(reference_logits.double(), probability.double(), data["R"]["mask"]))
        target_mechanism = mechanism(teacher, bank, node)
        save_tensor(evaluation_path / "teacher_mechanism.pt", target_mechanism)
        target_parameters = parameters(teacher)
        probe = build(meta, torch.load(OUT / "initial_states" / f"{SEEDS[0]}.pt", weights_only=True)["model"]).eval()
        probe_x = data["R"]["x"][:2]
        sample = _sample_spikes(probe, probe_x, trials=1, seed=887766)[:, 0]
        q = probe(probe_x, observed_counts=sample).spike_probability
        rng = torch.Generator().manual_seed(887766)
        replay = torch.stack([(torch.rand((2, 1), generator=rng) < q[:, t]).float() for t in range(150)], 1)
        assert float(probe.gates.history) > 0 and torch.equal(sample, replay)
        for condition in CONDITIONS:
            for seed in SEEDS:
                cp = finals[condition, seed]
                model = build(meta, cp["model"]).eval().requires_grad_(False)
                logits = predict(model, data["R"])
                nll = float(expected_bernoulli_nll(logits.double(), data["R"]["target"].double(), data["R"]["mask"]))
                ce = float(expected_bernoulli_nll(logits.double(), probability.double(), data["R"]["mask"]))
                prediction_rows.append({"condition": condition, "seed": seed, "sampled_spike_NLL": nll,
                    "expected_Bernoulli_CE": ce, "teacher_sampled_NLL": teacher_nll,
                    "teacher_expected_CE_entropy": entropy, "excess_CE_over_teacher": ce - entropy,
                    "scored_bins": 32 * 120, "history": "same teacher-sampled observed past"})
                h = torch.cat([h_observation(model, z, node) for z in data["H"]["x"].split(8)])
                b = torch.cat([b_observation(model, z) for z in data["B"]["x"].split(8)])
                latent.extend(latent_rows(condition, seed, "H1", h, data["H"]["target"], norm["H"]))
                latent.extend(latent_rows(condition, seed, "BC", b, data["B"]["target"], norm["B"]))
                for name, value in parameters(model).items():
                    truth = target_parameters[name]
                    parameter_rows.append({"condition": condition, "seed": seed, "parameter": name,
                        "teacher": truth, "student": value, "signed_error": value - truth,
                        "absolute_error": abs(value - truth), "relative_absolute_error": abs(value - truth) / (abs(truth) + EPS),
                        "role": "SECONDARY_CANONICAL_COORDINATE_NOT_IDENTIFIABILITY_PROOF"})
                response = mechanism(model, bank, node)
                all_mechanism[condition, seed] = response
                save_tensor(evaluation_path / f"{condition}_{seed}.pt", {"RGC_logits": logits, "H1_heldout": h,
                    "BC_heldout": b, **{"mechanism_" + key: value for key, value in response.items()}})
                for name in INTERVENTIONS:
                    target = target_mechanism[name][:, 30:]
                    delta = response[name][:, 30:]
                    error, target_rms = rms(delta - target), rms(target)
                    interventions.append({"condition": condition, "seed": seed, "intervention": name,
                        "normalized_intervention_error": error / (target_rms + EPS), "absolute_RMS_error_logit": error,
                        "teacher_Delta_RMS_logit": target_rms, "student_Delta_RMS_logit": rms(delta),
                        "teacher_denominator_above_epsilon": target_rms > EPS, "scored_bins": 96 * 120,
                        "history": "FIX_HISTORY_ZERO"})
                assert state_sha(model.state_dict()) == cp["state_sha256"]
                print("evaluated", condition, seed, flush=True)
        assert state_sha(teacher.state_dict()) == before
    ambiguity = []
    for condition in CONDITIONS:
        pairs = []
        for first, second in itertools.combinations(SEEDS, 2):
            a, b = all_mechanism[condition, first], all_mechanism[condition, second]
            values = {"H1": rms(a["H1"][:, 30:] - b["H1"][:, 30:]) / float(norm["H"][0])}
            for index, branch in enumerate(BRANCHES):
                values[branch] = rms(a["BC"][:, 30:, index] - b["BC"][:, 30:, index]) / float(norm["B"][index])
            values["BC_mean4"] = statistics.mean(values[branch] for branch in BRANCHES)
            for name in INTERVENTIONS:
                values[name] = rms(a[name][:, 30:] - b[name][:, 30:]) / (rms(target_mechanism[name][:, 30:]) + EPS)
            for measure, value in values.items():
                row = {"condition": condition, "kind": "A_intervention" if measure in INTERVENTIONS else "A_latent",
                    "measure": measure, "seed_i": first, "seed_j": second, "pairwise_normalized_RMS": value,
                    "aggregation": "single_pair", "bank": "same96sequence_mechanism_bank"}
                pairs.append(row)
                ambiguity.append(row)
        for measure in values:
            matching = [row for row in pairs if row["measure"] == measure]
            ambiguity.append({**matching[0], "seed_i": "", "seed_j": "",
                "pairwise_normalized_RMS": statistics.mean(row["pairwise_normalized_RMS"] for row in matching),
                "aggregation": "mean_3_pairs"})
    outputs = {"heldout_prediction.csv": prediction_rows, "latent_recovery.csv": latent,
               "parameter_recovery.csv": parameter_rows, "intervention_recovery.csv": interventions,
               "ambiguity.csv": ambiguity}
    curves = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            with (OUT / "curves" / f"{condition}_{seed}.csv").open(encoding="utf-8") as stream:
                curves.extend(csv.DictReader(stream))
    outputs["training_curves.csv"] = curves
    for name, rows in outputs.items():
        write_csv(OUT / name, rows)
    summaries = {}
    def aggregate(rows, field):
        values = [row[field] for row in rows]
        return {"mean": statistics.mean(values), "min": min(values), "max": max(values),
                "values_by_seed": dict(zip(SEEDS, values, strict=True))}
    for condition in CONDITIONS:
        pred = [r for r in prediction_rows if r["condition"] == condition]
        lat = [r for r in latent if r["condition"] == condition]
        iv = [r for r in interventions if r["condition"] == condition]
        amb = [r for r in ambiguity if r["condition"] == condition and r["aggregation"] == "mean_3_pairs"]
        summaries[condition] = {"sampled_NLL": aggregate(pred, "sampled_spike_NLL"),
            "expected_CE": aggregate(pred, "expected_Bernoulli_CE"),
            "excess_CE": aggregate(pred, "excess_CE_over_teacher"),
            "H1_nRMSE": aggregate([r for r in lat if r["channel"] == "H1"], "normalized_RMSE"),
            "BC_nRMSE": aggregate([r for r in lat if r["channel"] == "BC_mean4"], "normalized_RMSE"),
            "interventions": {name: aggregate([r for r in iv if r["intervention"] == name], "normalized_intervention_error") for name in INTERVENTIONS},
            "mean4_intervention_error": statistics.mean(r["normalized_intervention_error"] for r in iv),
            "ambiguity": {r["measure"]: r["pairwise_normalized_RMS"] for r in amb},
            "mean4_intervention_ambiguity": statistics.mean(r["pairwise_normalized_RMS"] for r in amb if r["kind"] == "A_intervention")}
    comparisons = {}
    for other in ("RGC_ONLY", "EXTRA_RGC_CONTROL"):
        d, a = summaries["RGC_H1_BC"], summaries[other]
        comparisons["RGC_H1_BC_vs_" + other] = {
            "expected_CE_difference": d["expected_CE"]["mean"] - a["expected_CE"]["mean"],
            "expected_CE_relative_change": d["expected_CE"]["mean"] / a["expected_CE"]["mean"] - 1,
            "prediction_preserved_within_predefined_1percent_descriptive_margin": d["expected_CE"]["mean"] <= 1.01 * a["expected_CE"]["mean"],
            "mean4_intervention_error_ratio": d["mean4_intervention_error"] / a["mean4_intervention_error"],
            "mean4_intervention_ambiguity_ratio": d["mean4_intervention_ambiguity"] / a["mean4_intervention_ambiguity"],
            "per_intervention_error_difference": {name: d["interventions"][name]["mean"] - a["interventions"][name]["mean"] for name in INTERVENTIONS}}
    summary = {"status": "COMPLETE", "finished_utc": utc(), "protocol_sha256": protocol_sha,
        "late_temporal_contract_clarification": {"user_confirmed": "complete current frozen S0 and disclose difference",
            "sampling_Hz": 150, "existing_rho_H_B_R": [.85, .30, .92],
            "existing_noise_correlation_tau_ms_H_B_R": [41.020862537480234, 5.537223633883582, 79.95368225071985],
            "new_suggested_tau_ms_not_applied": [50, 20, 40], "in_flight_protocol_changed": False},
        "invalid_predecessor": p["invalid_predecessor"],
        "conditions": summaries, "primary_comparisons": comparisons,
        "teacher": {"sampled_NLL": teacher_nll, "expected_CE_entropy": entropy,
                    "history_gate": float(teacher.gates.history), "parameters": target_parameters,
                    "heldout_probability_scored_mean": float(probability[:, 30:].double().mean()),
                    "heldout_probability_scored_std": float(probability[:, 30:].double().std(correction=0)),
                    "heldout_probability_scored_min": float(probability[:, 30:].min()),
                    "heldout_probability_scored_max": float(probability[:, 30:].max())},
        "verification": {"15_fits_final_step3000": True, "45000_optimizer_updates": True,
            "all_current_source_hashes_match_protocol": True, "teacher_checkpoint_and_state_unchanged": True,
            "heldout_read_after_all_training": True, "teacher_probability_replay_exact": True,
            "nonzero_history_initial_model_sampler_replay": True,
            "per_fit_gradient_seen": {f"{c}_{s}": cp["gradient_nonzero_seen"] for (c, s), cp in finals.items()},
            "per_fit_parameter_changed": {f"{c}_{s}": cp["parameter_changed"] for (c, s), cp in finals.items()}},
        "limits": ["one frozen fitted-model teacher, not biological ground truth", "3 seeds: descriptive only",
            "fixed finite budget does not distinguish finite-data/optimization limitations from structural non-identifiability",
            "teacher history gate0 despite correct causal sampler", "no AC supervision",
            "E has192 independent RGC training sequences, A-D RGC has64; same updates/batch, hence not equal data exposure or compute across multi-loss conditions",
            "noise-free identity H1/BC observations are more informative than stochastic Bernoulli samples; S0 does not claim information-matched control",
            "no S1, architecture search, real physiology or population training executed"]}
    checked_protocol()
    write_json(OUT / "summary.json", summary)
    files = [OUT / name for name in outputs] + [OUT / "summary.json", Path(__file__)]
    files += list(evaluation_path.glob("*.pt"))
    write_json(OUT / "evaluation_manifest.json", {str(path.relative_to(ROOT)).replace('\\', '/'): sha(path) for path in files})
    print("EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    configure_runtime()
    evaluate()
