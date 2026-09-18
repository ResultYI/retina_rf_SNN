from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from work.retipath_multiobs_s0 import run as s0
import torch

OUT = ROOT / "output/experiments/retipath_multiobs_s05_h1_observability_20260917"
SEEDS, UPDATES, EPS = s0.SEEDS, s0.UPDATES, s0.EPS
CONDITIONS = {"RGC_ONLY": (), "RGC_STATE": ("state",),
              "RGC_FEEDBACK": ("feedback",), "RGC_STATE_FEEDBACK": ("state", "feedback")}
read_json, write_json, sha = s0.read_json, s0.write_json, s0.sha
save_tensor, write_csv, state_sha, utc = s0.save_tensor, s0.write_csv, s0.state_sha, s0.utc
build, configure_runtime = s0.build, s0.configure_runtime


def observations(model, x, p):
    h = model.h1(x, amplitude=model.gates.h1)
    node, pixel = p["observation"]["state_node"], p["observation"]["feedback_pixel"]
    return {"state": h.state[..., node:node+1], "feedback": h.surround[..., pixel:pixel+1]}


def normalized_loss(prediction, target, norm):
    mean, std = (torch.tensor(norm[k], dtype=torch.float32) for k in ("mean", "std"))
    return (((prediction[:, 30:] - mean) / std - (target[:, 30:] - mean) / std)**2).mean()


def source_files():
    return [Path(__file__), Path(__file__).with_name("launch.py"),
            Path(__file__).with_name("evaluate.py"), Path(__file__).with_name("verify_results.py")]


def checked_protocol():
    p = read_json(OUT / "protocol.json")
    s0.checked_protocol()
    for name, digest in p["source_sha256"].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in p["input_sha256"].items():
        assert sha(ROOT / name) == digest, name
    return p


def load_s0(name):
    allowed = {"D_H_train", "D_H_heldout", "D_R_train", "D_R_heldout",
               "mechanism_H", "mechanism_B", "mechanism_R"}
    assert name in allowed
    return torch.load(s0.OUT / "datasets" / f"{name}.pt", weights_only=True, map_location="cpu")


def prepare():
    if OUT.exists():
        raise FileExistsError("Never overwrite or restart an existing S0.5 run")
    old = s0.checked_protocol()
    meta = read_json(s0.OUT / "teacher_metadata.json")
    positions = torch.tensor(meta["cone_positions_degs"])
    center = torch.tensor(meta["cell_positions_degs"])[0]
    distance = (positions - center).square().sum(-1)
    pixel = int(distance.argmin())
    assert old["observation"]["h1_node_index"] == 146
    inputs = [s0.OUT / name for name in ("protocol.json", "teacher_metadata.json",
              "normalization.json", "preparation_complete.json", "dataset_manifest.json")]
    names = ("D_H_train", "D_H_heldout", "D_R_train", "D_R_heldout",
             "mechanism_H", "mechanism_B", "mechanism_R")
    inventory = read_json(s0.OUT / "dataset_manifest.json")
    for name in names:
        path = s0.OUT / "datasets" / f"{name}.pt"
        assert sha(path) == inventory[name]["file_sha256"]
        inputs.append(path)
    inputs += [s0.OUT / "initial_states" / f"{seed}.pt" for seed in SEEDS]
    inputs += [s0.OUT / "datasets/teacher_probabilities/D_R_heldout.pt"]
    p = {"task": "Synthetic S0.5 H1 Observation-Mechanism Alignment", "frozen_utc": utc(),
         "teacher": old["teacher"], "S0_protocol_sha256": sha(s0.OUT / "protocol.json"),
         "student_seeds": list(SEEDS), "conditions": CONDITIONS,
         "sequence": old["sequence"], "stimulus": "Reuse exact saved S0 tensors; no generation or AR changes",
         "observation": {"state_node": 146, "state_normalization": "exact frozen S0 H mean/std",
             "feedback_pixel": pixel, "feedback_pixel_deg": positions[pixel].tolist(),
             "functional_center_deg": center.tolist(), "squared_distance_deg": float(distance[pixel]),
             "selection_rule": "argmin squared Euclidean distance in saved input geometry to frozen functional center; first index tie break; selected before any student response",
             "feedback_definition": "formal H1Pathway.forward(..., amplitude=model.gates.h1).surround[..., pixel:pixel+1]; amplitude * graph.transpose_apply(state)",
             "feedback_normalization": "teacher D_H TRAIN scored bins only, float64 population mean/std; applied float32 in training",
             "head": "identity", "scalar_traces_per_observable": 1, "noise_scale_offset_filter": None},
         "training": {**{k: v for k, v in old["training"].items() if k not in ("H", "B")},
             "total_fits": 12, "total_optimizer_updates": 36000,
             "schedule": "same S0 H and R minibatch indices; one H forward per active step; D sums state+feedback on SAME H batch then backward; R backward then single optimizer step",
             "loss_coefficients": {"state": 1, "feedback": 1, "R": 1},
             "initialization": "strict load identical saved S0 fresh initial state per seed; no final checkpoint warm-start",
             "allowed_training_files": ["S0 datasets/D_H_train.pt", "S0 datasets/D_R_train.pt", "S05 feedback_train.pt"],
             "forbidden": ["D_B", "D_R_EXTRA", "teacher probability training targets", "intervention loss"]},
         "evaluation": {"after_all_12_final_checkpoints": True,
             "latent": "same D_H heldout; separate state/feedback normalized RMSE and correlation",
             "primary": "BLOCK_H1_FEEDBACK only; S0 formal definition; NORMAL and intervention use FIX_HISTORY_ZERO",
             "mechanism_bank": "exact same96sequence bank concatenated mechanism_H/B/R; mechanism_B is immutable evaluation slice, NOT D_B observation data",
             "intervention_error": "RMS(student Delta-teacher Delta)/(RMS(teacher Delta)+1e-8), bins30:150",
             "ambiguity": "all three unordered seed pairs on SAME96sequence mechanism bank; state/feedback RMS difference divided by respective teacher TRAIN std; intervention divided by teacher Delta RMS+1e-8",
             "parameters": ["H1_tau_ms", "H1_delay_ms", "H1_amplitude"],
             "RGC": "same D_R heldout NLL and expected CE against saved teacher p; identical sampled past conditioning",
             "prediction_preservation": "expected CE mean <=1.01*RGC_ONLY mean; descriptive only",
             "optional_secondary_interventions": [], "epsilon": EPS},
         "verdict_rules": {"user_confirmed": True,
             "clear_improvement": "mean normalized error <=0.90*comparator mean AND strictly lower error in all3 paired seeds",
             "STATE_SUPERVISION_SUFFICIENT": "B clearly improves state, feedback AND intervention versus A",
             "FEEDBACK_OBSERVATION_REQUIRED": "B state recovery clearly improves OR A_state mean pairwise ambiguity <=0.90*A; C clearly improves feedback AND intervention versus BOTH A and B; D preserves clear improvements versus BOTH A and B and each mean feedback/intervention error <=1.05*C",
             "MIXED_OR_OPTIMIZATION_LIMITED": "neither preceding rule satisfied, including unstable paired seed directions",
             "precedence": ["STATE_SUPERVISION_SUFFICIENT", "FEEDBACK_OBSERVATION_REQUIRED", "MIXED_OR_OPTIMIZATION_LIMITED"],
             "inference": "one known synthetic teacher,3 seeds; no statistical population or biological inference; no post-result criterion changes"},
         "runtime": old["runtime"],
         "source_sha256": {str(path.relative_to(ROOT)).replace(chr(92), "/"): sha(path) for path in source_files()},
         "input_sha256": {str(path.relative_to(ROOT)).replace(chr(92), "/"): sha(path) for path in inputs},
         "boundaries": ["no S0 writes", "no stimulus regeneration", "no architecture changes", "no additional seeds or steps", "no S1", "no real data"]}
    OUT.mkdir(parents=True)
    for folder in ("checkpoints", "curves", "verification", "evaluation_arrays"):
        (OUT / folder).mkdir()
    write_json(OUT / "protocol.json", p)
    tcp = torch.load(ROOT / p["teacher"]["path"], weights_only=True, map_location="cpu")
    teacher = build(meta, tcp["model"]).eval().requires_grad_(False)
    before = state_sha(teacher.state_dict())
    data = load_s0("D_H_train")
    with torch.no_grad():
        feedback = torch.cat([observations(teacher, x, p)["feedback"] for x in data["x"].split(8)])
    scored = feedback[:, 30:].double().reshape(-1, 1)
    norm = {"state": read_json(s0.OUT / "normalization.json")["H"],
            "feedback": {"mean": scored.mean(0).tolist(), "std": scored.std(0, correction=0).tolist(),
                         "bins": len(scored), "source": "teacher D_H_train bins30:150"}}
    assert norm["feedback"]["std"][0] > EPS
    assert torch.isfinite(feedback).all() and state_sha(teacher.state_dict()) == before
    save_tensor(OUT / "feedback_train.pt", feedback)
    write_json(OUT / "normalization.json", norm)
    write_json(OUT / "preparation_complete.json", {"completed_utc": utc(),
        "protocol_sha256": sha(OUT / "protocol.json"), "normalization_sha256": sha(OUT / "normalization.json"),
        "feedback_train_sha256": sha(OUT / "feedback_train.pt"), "teacher_state_sha256": before,
        "student_responses_computed": 0, "optimizer_updates": 0})
    print("PREPARED", "pixel", pixel, "feedback_std", norm["feedback"]["std"], flush=True)


def verify():
    p = checked_protocol()
    meta = read_json(s0.OUT / "teacher_metadata.json")
    teacher = build(meta, torch.load(ROOT / p["teacher"]["path"], weights_only=True)["model"]).eval().requires_grad_(False)
    before = state_sha(teacher.state_dict())
    h, r = load_s0("D_H_train"), load_s0("D_R_train")
    feedback = torch.load(OUT / "feedback_train.pt", weights_only=True)
    norm = read_json(OUT / "normalization.json")
    checks = {}
    with torch.no_grad():
        x = h["x"][:4]
        obs = observations(teacher, x, p)
        trace = s0.observe_mechanism(teacher, x, observed_counts=torch.zeros(4, 150, 1))
        for key, name, index in (("state", "h1_state", 146), ("feedback", "h1_feedback", p["observation"]["feedback_pixel"])):
            assert torch.equal(obs[key], trace.tensors[name][..., index:index+1])
        torch.testing.assert_close(obs["state"], h["target"][:4], rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(obs["feedback"], feedback[:4], rtol=1e-5, atol=2e-6)
        checks["scalar_state_feedback_match_formal_forward"] = True
        expected = teacher.gates.h1 * teacher.h1.graph.transpose_apply(trace.tensors["h1_state"])
        assert torch.equal(expected, trace.tensors["h1_feedback"])
        blocked = s0.observe_mechanism(teacher, x, observed_counts=torch.zeros(4, 150, 1),
                                     intervention=s0.InterventionSpec(s0.Intervention.BLOCK_H1_FEEDBACK))
        assert not torch.count_nonzero(blocked.tensors["h1_feedback"])
        assert torch.equal(blocked.tensors["h1_state"], trace.tensors["h1_state"])
        checks["return_mapping_amplitude_and_official_feedback_block"] = True
        for name, target in (("state", h["target"]), ("feedback", feedback)):
            z = target[:, 30:].double().reshape(-1, 1)
            assert z.mean(0).tolist() == norm[name]["mean"]
            assert z.std(0, correction=0).tolist() == norm[name]["std"]
        checks["train_only_normalization_exact_and_finite"] = True
    old_prep = read_json(s0.OUT / "preparation_complete.json")
    for seed in SEEDS:
        initial = torch.load(s0.OUT / "initial_states" / f"{seed}.pt", weights_only=True)
        assert state_sha(initial["model"]) == old_prep["initial_sha256"][str(seed)]
        for dataset in ("H", "R"):
            assert s0.tensor_sha(s0.schedule(seed, dataset, 64)) == old_prep["schedule_sha256"][str(seed)][dataset]
    checks["exact_S0_initial_states_and_H_R_batch_schedules"] = True
    model = build(meta, initial["model"])
    output = observations(model, h["x"][:4], p)
    amplitude = model.gates.raw_h1_amplitude
    state_grad = torch.autograd.grad(output["state"].square().mean(), amplitude, allow_unused=True, retain_graph=True)[0]
    feedback_grad = torch.autograd.grad(output["feedback"].square().mean(), amplitude)[0]
    assert state_grad is None and torch.isfinite(feedback_grad).all() and torch.count_nonzero(feedback_grad)
    checks["state_has_no_direct_amplitude_gradient_feedback_has_nonzero_gradient"] = True
    def losses():
        z = observations(model, h["x"][:4], p)
        state_loss = normalized_loss(z["state"], h["target"][:4], norm["state"])
        feedback_loss = normalized_loss(z["feedback"], feedback[:4], norm["feedback"])
        rgc_loss = s0.expected_bernoulli_nll(model(r["x"][:4], observed_counts=r["target"][:4]).logits,
                                          r["target"][:4], r["mask"][:4])
        return state_loss + feedback_loss, rgc_loss
    for loss in losses():
        loss.backward()
    gradients = {k: v.grad.clone() for k, v in model.named_parameters() if v.requires_grad}
    model.zero_grad()
    sum(losses()).backward()
    for key, parameter in model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, gradients[key], rtol=2e-5, atol=2e-6)
    checks["D_same_H_batch_two_losses_single_macro_gradient_sum"] = True
    assert state_sha(teacher.state_dict()) == before
    checks["teacher_unchanged_zero_optimizer_updates"] = True
    checked_protocol()
    result = {"status": "PASS", "verified_utc": utc(), "protocol_sha256": sha(OUT / "protocol.json"),
              "checks": checks, "optimizer_updates": 0, "review": "executor self-check"}
    write_json(OUT / "verification/pretraining.json", result)
    print(result, flush=True)


def train(condition, seed):
    p = checked_protocol()
    protocol_sha = sha(OUT / "protocol.json")
    v = read_json(OUT / "verification/pretraining.json")
    assert v["status"] == "PASS" and v["protocol_sha256"] == protocol_sha
    assert condition in CONDITIONS and seed in SEEDS
    final_path = OUT / "checkpoints" / f"{condition}_{seed}_final.pt"
    start_path = OUT / "checkpoints" / f"{condition}_{seed}_started.json"
    if final_path.exists() or start_path.exists():
        raise FileExistsError("No restart, overwrite, selection or extension")
    prep = read_json(OUT / "preparation_complete.json")
    assert sha(OUT / "normalization.json") == prep["normalization_sha256"]
    assert sha(OUT / "feedback_train.pt") == prep["feedback_train_sha256"]
    meta = read_json(s0.OUT / "teacher_metadata.json")
    initial = torch.load(s0.OUT / "initial_states" / f"{seed}.pt", weights_only=True)
    assert state_sha(initial["model"]) == initial["initial_sha256"]
    model = build(meta, initial["model"]).train()
    params = [v for v in model.parameters() if v.requires_grad]
    optimizer = torch.optim.Adam(params, lr=.003, betas=(.9, .999), eps=1e-8, weight_decay=0)
    r = load_s0("D_R_train")
    r_batches = s0.schedule(seed, "R", 64)
    h = load_s0("D_H_train") if CONDITIONS[condition] else None
    feedback = torch.load(OUT / "feedback_train.pt", weights_only=True) if "feedback" in CONDITIONS[condition] else None
    h_batches = s0.schedule(seed, "H", 64)
    norm = read_json(OUT / "normalization.json")
    write_json(start_path, {"started_utc": utc(), "condition": condition, "seed": seed,
        "protocol_sha256": protocol_sha, "initial_sha256": initial["initial_sha256"],
        "target_updates": UPDATES, "training_datasets": ["D_R_train"] + (["D_H_train"] if h is not None else []),
        "heldout_teacher_probabilities_interventions_read": [], "S0_schedule_sha256": {
            "R": s0.tensor_sha(r_batches), "H": s0.tensor_sha(h_batches)}})
    started, curves = time.perf_counter(), []
    gradient_seen = {k: False for k, v in model.named_parameters() if v.requires_grad}
    for step in range(1, UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        values = {}
        if h is not None:
            idx = h_batches[step-1]
            obs = observations(model, h["x"][idx], p)
            hlosses = []
            for key in CONDITIONS[condition]:
                target = h["target"][idx] if key == "state" else feedback[idx]
                loss = normalized_loss(obs[key], target, norm[key])
                hlosses.append(loss)
                values[key] = float(loss.detach())
            h_loss = sum(hlosses)
            if not torch.isfinite(h_loss):
                raise FloatingPointError(f"Nonfinite H loss {condition}/{seed}/{step}")
            h_loss.backward()
        idx = r_batches[step-1]
        logits = model(r["x"][idx], observed_counts=r["target"][idx]).logits
        loss = s0.expected_bernoulli_nll(logits, r["target"][idx], r["mask"][idx])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite R loss {condition}/{seed}/{step}")
        loss.backward()
        values["R"] = float(loss.detach())
        for key, parameter in model.named_parameters():
            if parameter.requires_grad:
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"Missing/nonfinite gradient {key}")
                gradient_seen[key] |= bool(torch.count_nonzero(parameter.grad))
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 5, error_if_nonfinite=True))
        optimizer.step()
        model.project_mechanism_parameters()
        if step == 1 or step % 25 == 0:
            curves.append({"condition": condition, "seed": seed, "step": step,
                "L_state": values.get("state"), "L_feedback": values.get("feedback"), "L_R": values["R"],
                "total_loss": sum(values.values()), "gradient_norm_before_clip": grad_norm,
                "elapsed_seconds": time.perf_counter() - started})
        if step % 250 == 0:
            print(condition, seed, step, "seconds", round(time.perf_counter() - started, 1), flush=True)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    assert {int(optimizer.state[v]["step"]) for v in params} == {UPDATES}
    assert all(torch.isfinite(v).all() for v in state.values())
    write_csv(OUT / "curves" / f"{condition}_{seed}.csv", curves)
    save_tensor(final_path, {"model": state, "optimizer": optimizer.state_dict(), "condition": condition,
        "seed": seed, "step": UPDATES, "protocol_sha256": protocol_sha,
        "initial_sha256": initial["initial_sha256"], "state_sha256": state_sha(state),
        "gradient_nonzero_seen": gradient_seen, "finished_utc": utc(),
        "elapsed_seconds": time.perf_counter() - started})
    print("COMPLETE", condition, seed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "verify", "train"))
    parser.add_argument("--condition", choices=tuple(CONDITIONS))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    configure_runtime()
    if args.action == "prepare":
        prepare()
    elif args.action == "verify":
        verify()
    else:
        train(args.condition, args.seed)
