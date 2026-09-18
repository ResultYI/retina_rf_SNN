from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch

from evaluation.mechanistic_retina.clean_sampled_data import _sample_spikes
from evaluation.mechanistic_retina.mechanism_observation import Intervention, InterventionSpec, observe_mechanism
from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.retipath import RetiPath
from models.mechanistic_retina.retipath_canonical_gain import CanonicalGainRetiPath, canonicalize_gain_state
from training.mechanistic_retina.losses import expected_bernoulli_nll

OUT = ROOT / "output/experiments/retipath_multiobs_synthetic_s0_20260917"
MANIFEST = ROOT / "output/evaluations/retipath_gain_canonicalization_20260915/migration_manifest.json"
SEEDS = (2026091701, 2026091702, 2026091703)
CONDITIONS = {"RGC_ONLY": ("R",), "RGC_H1": ("H", "R"),
              "RGC_BC": ("B", "R"), "RGC_H1_BC": ("H", "B", "R"),
              "EXTRA_RGC_CONTROL": ("R_EXTRA",)}
INTERVENTIONS = ("BLOCK_H1_FEEDBACK", "BLOCK_DIRECT_BC_DRIVE",
                 "BLOCK_AC_POSTSYNAPTIC_DRIVE", "REMOVE_ADAPTATION_TERM")
BRANCHES = ("direct_sustained", "direct_transient", "broad_sustained", "broad_transient")
UPDATES, BATCH, EPS = 3000, 4, 1e-8


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_sha(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_tensor(path, value):
    with Path(path).open("xb") as stream:
        torch.save(value, stream)


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_runtime():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def model_args(metadata):
    return (MechanisticRetinaConfig(**metadata["model_config"]),
            torch.as_tensor(metadata["cone_positions_degs"]),
            torch.as_tensor(metadata["cell_positions_degs"]),
            tuple(metadata["cell_types"]), tuple(metadata["polarities"]))


def build(metadata, state=None):
    model = CanonicalGainRetiPath(*model_args(metadata),
        rms_e=torch.tensor(metadata["rms"]["e"]), rms_i=torch.tensor(metadata["rms"]["i"]))
    if state is not None:
        model.load_state_dict(state, strict=True)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 33
    return model


def fresh(metadata, seed):
    torch.manual_seed(seed)
    legacy = RetiPath(*model_args(metadata), rms_e=torch.tensor(metadata["rms"]["e"]),
                     rms_i=torch.tensor(metadata["rms"]["i"]))
    return build(metadata, canonicalize_gain_state(legacy.state_dict()))


def h_observation(model, x, node):
    return model.h1(x, amplitude=model.gates.h1).state[..., node:node + 1]


def b_observation(model, x):
    parts = model.spatial_components(x)
    return torch.cat((parts.direct.sum(-2), parts.broad.sum(-2)), -1).squeeze(2)


def score_mask(n):
    mask = torch.ones(n, 150, 1, dtype=torch.bool)
    mask[:, :30] = False
    return mask


def stimulus(family, count, seed, positions):
    rng = torch.Generator().manual_seed(seed)
    axis_x = torch.unique(positions[:, 0], sorted=True)
    axis_y = torch.unique(positions[:, 1], sorted=True)
    assert len(axis_x) == len(axis_y) == 17
    coords = (positions - positions.mean(0)) / torch.stack((axis_x.diff().median(), axis_y.diff().median()))
    if family in ("H", "R"):
        sigma, rho = (2.0, 0.85) if family == "H" else (1.25, 0.92)
        distances = torch.cdist(coords, coords)
        kernel = torch.exp(-0.5 * (distances / sigma).square())
        kernel = kernel / kernel.square().sum(1, keepdim=True).sqrt()
        innovations = torch.randn(count, 150, 289, generator=rng) @ kernel.T
        if family == "R":
            innovations = (innovations + 0.35 * torch.randn(count, 150, 1, generator=rng)) / (1.0 + 0.35**2)**0.5
        state = torch.zeros(count, 289)
        values = []
        for t in range(150):
            state = rho * state + (1 - rho**2)**0.5 * innovations[:, t]
            values.append(state)
        values = torch.stack(values, 1)
        if family == "H":
            levels = torch.tensor([-0.6, -0.3, 0.3, 0.6])
            steps = levels[torch.randint(4, (count, 5, 1), generator=rng)].repeat_interleave(30, 1)
            x = 0.8 * torch.tanh(0.6 * values + 0.8 * steps)
        else:
            x = 0.8 * torch.tanh(0.7 * values)
    elif family == "B":
        x = torch.empty(count, 150, 289)
        for i in range(count):
            if i % 2 == 0:
                noise = torch.randn(150, generator=rng)
                state = torch.tensor(0.0)
                values = []
                for t in range(150):
                    state = 0.3 * state + (1 - 0.3**2)**0.5 * noise[t]
                    values.append(state)
                x[i] = (0.8 * torch.tanh(torch.stack(values)))[:, None]
            else:
                angle = torch.tensor((i // 2 % 4) * torch.pi / 4)
                phase = 2 * torch.pi * torch.rand((), generator=rng)
                direction = torch.stack((torch.cos(angle), torch.sin(angle)))
                pattern = torch.cos(2 * torch.pi * (coords @ direction) / 8 + phase)
                sign = 2 * torch.randint(2, (), generator=rng).item() - 1
                reversal = sign * (1 - 2 * ((torch.arange(150) // 15) % 2))
                amplitude = (0.2 + 0.4 * torch.rand(10, generator=rng)).repeat_interleave(15)
                x[i] = amplitude[:, None] * reversal[:, None] * pattern[None]
    else:
        raise ValueError(family)
    assert x.shape == (count, 150, 289) and torch.isfinite(x).all()
    assert float(x.abs().max()) <= 0.8
    return x.contiguous()


def schedule(seed, dataset, count):
    offset = {"H": 100003, "B": 200003, "R": 300003, "R_EXTRA": 400003}[dataset]
    rng = torch.Generator().manual_seed(seed + offset)
    return torch.randint(count, (UPDATES, BATCH), generator=rng)


def source_paths():
    paths = list((ROOT / "models/mechanistic_retina").glob("*.py"))
    paths += [ROOT / p for p in ("evaluation/mechanistic_retina/mechanism_observation.py",
        "evaluation/mechanistic_retina/clean_sampled_data.py", "training/mechanistic_retina/losses.py",
        "work/retipath_phase2_common.py", "work/retipath_phase2_train.py")]
    paths += [Path(__file__), Path(__file__).with_name("test_contract.py")]
    return sorted(paths)


def checked_protocol():
    p = read_json(OUT / "protocol.json")
    for name, expected in p["source_sha256"].items():
        assert sha(ROOT / name) == expected, f"Changed frozen source: {name}"
    assert sha(ROOT / p["teacher"]["path"]) == p["teacher"]["sha256"]
    return p


def prepare():
    if OUT.exists():
        raise FileExistsError("Existing S0 run must not be overwritten")
    manifest = read_json(MANIFEST)
    entry = next(row for row in manifest["checkpoints"] if row["cell"] == "67#7" and row["seed"] == 2026091301)
    assert sha(ROOT / entry["path"]) == entry["sha256"]
    cp = torch.load(ROOT / entry["path"], weights_only=True, map_location="cpu")
    assert cp["cell_id"] == "67#7" and cp["seed"] == 2026091301
    assert cp["gain_schema"] == "retipath_canonical_gain_v1"
    metadata = {key: cp[key] for key in ("model_config", "cone_positions_degs", "cell_positions_degs", "cell_types", "polarities", "rms")}
    for key in ("cone_positions_degs", "cell_positions_degs"):
        metadata[key] = metadata[key].tolist()
    positions = cp["cone_positions_degs"]
    distances = (positions - cp["cell_positions_degs"][0]).square().sum(-1)
    node = int(distances.argmin())
    specifications = {
        "D_H_train": ["H", 64, 2026091711], "D_H_heldout": ["H", 32, 2026091712],
        "D_B_train": ["B", 64, 2026091721], "D_B_heldout": ["B", 32, 2026091722],
        "D_R_train": ["R", 64, 2026091731], "D_R_heldout": ["R", 32, 2026091732],
        "D_R_EXTRA_train": ["R", 192, 2026091741],
        "mechanism_H": ["H", 32, 2026091751], "mechanism_B": ["B", 32, 2026091752],
        "mechanism_R": ["R", 32, 2026091753],
    }
    protocol = {
        "task": "RetiPath Synthetic MultiObs Stage S0", "frozen_utc": utc(),
        "teacher": {key: entry[key] for key in ("cell", "seed", "path", "sha256")},
        "teacher_manifest_sha256": sha(MANIFEST), "student_seeds": list(SEEDS),
        "conditions": CONDITIONS, "datasets": specifications,
        "sequence": {"bins": 150, "Hz": 150, "warmup": 30, "scored_bins": 120, "independent_reset": True},
        "stimulus": {"range": [-0.8, 0.8], "implementation": "work/retipath_multiobs_s0/run.py::stimulus",
            "H": "spatial Gaussian sigma=2 grid pixels, L2 row-normalized innovation kernel; AR(1) rho=.85, innovation sqrt(1-rho^2), reset0; full-field levels [-.6,-.3,.3,.6] every30 bins; .8*tanh(.6*AR+.8*step)",
            "B": "alternating sequence types: full-field rho=.3 flicker .8*tanh(AR); fixed spatial cos gratings period8 grid pixels, orientations[0,45,90,135]deg, independent phase Uniform[0,2pi) fixed within sequence, reversal every15bins, independent amplitude Uniform[.2,.6] per15bin block, random initial sign",
            "R": "Gaussian sigma1.25grid pixels L2 normalized; local + .35global Gaussian innovations, divide sqrt(1+.35^2); ARrho=.92 reset0; .8*tanh(.7*AR)",
            "mechanism_bank": "96 independent sequences: 32H+32B+32R; equal sequence weighting; held-out only"},
        "spike_seeds": {name: spec[2] + 10000 for name, spec in specifications.items() if name.startswith("D_R")},
        "causal_sampling": "reuse clean_sampled_data._sample_spikes, trials=1; update history from previous sampled spike before current p; save conditional teacher p separately, never read by training",
        "observation": {"head": "identity", "noise": None, "learned_scale_offset_filter": False,
            "h1_node_index": node, "h1_node_deg": positions[node].tolist(),
            "functional_center_deg": cp["cell_positions_degs"][0].tolist(),
            "center_rule": "nearest graph/input node to checkpoint fixed functional cell center; first index breaks ties; no response-based center selection",
            "BC": list(BRANCHES), "BC_reduction": "sum K=2 of canonical parts.direct/parts.broad before composition/gain; concatenate direct[s,t],broad[s,t]; identical to official forward bc_direct/bc_broad",
            "normalization": "teacher respective TRAIN scored bins only; population std(ddof=0) per channel; fail if std<=1e-8; freeze same constants for all students"},
        "training": {"optimizer": "Adam", "lr": .003, "betas": [.9, .999], "eps": 1e-8,
            "weight_decay": 0, "gradient_norm_clip": 5, "batch_size_per_active_dataset": BATCH,
            "macro_steps": UPDATES, "total_fits": 15, "total_optimizer_updates": 45000,
            "schedule": "independent dataset minibatch with replacement; same seed/dataset schedule across conditions; H then B then R gradients accumulated, one optimizer.step",
            "loss_coefficients": {"H": 1, "B": 1, "R": 1}, "R": "existing mean Bernoulli NLL on sampled spikes",
            "H": "mean normalized MSE", "B": "mean normalized MSE over time and4channels",
            "constraints": "existing project_mechanism_parameters and native bounded coordinates, including alpha[.05,2] and history gate[0,1]",
            "early_stopping": False, "checkpoint_selection": False, "primary_checkpoint": "final step3000",
            "contract_source": "current RetiPath Phase2 Adam .003/batch4/clip5/max3000; S0 fixes all3000 with no selection",
            "alternative_not_used": "older clean_sampled contract400/.03/batch8 uses pre-spatial model and incompatible phase1 parameter list; it supplies causal sampler, not optimizer policy",
            "initialization": "standard fresh RetiPath(config, geometry, metadata, seed) followed by exact canonicalize_gain_state; NO teacher learnable weights or optimizer state",
            "fixed_buffers": "teacher architecture config, geometry and frozen RMS retained to define identical realizable coordinate system; legacy alias gains excluded from prediction; no real data read"},
        "evaluation": {"performed_after_all_final_checkpoints": True, "RGC": "D_R_heldout sampled NLL and expected CE against teacher conditional p given the SAME sampled past",
            "latent": "D_H_heldout H1 nRMSE/correlation; D_B_heldout4 per-branch and arithmetic mean, normalized by respective teacher TRAIN std",
            "parameter": "secondary: physical tau/delay/amplitude, alpha, G_E/G_I and anchored canonical composition; exclude raw/gauge coordinates, frozen values, history/readout degeneracies",
            "interventions": list(INTERVENTIONS), "history": "FIX_HISTORY_ZERO via all-zero observed_counts for every intervention; NORMAL baseline also explicitly FIX_HISTORY_ZERO",
            "intervention_error": "RMS(student Delta - teacher Delta)/(RMS(teacher Delta)+1e-8), scored bins of independent96-sequence mechanism bank",
            "latent_ambiguity": "on SAME96-sequence mechanism bank; each of3 unordered seed pairs: H1 RMS(diff)/train_Hstd; BC mean_q RMS(diff_q)/train_Bstd_q; then average pairs",
            "intervention_ambiguity": "per intervention RMS(seed_i Delta - seed_j Delta)/(RMS(teacher Delta)+1e-8); average3 pairs",
            "epsilon": EPS, "output_preservation_descriptive_margin": "expected CE no more than1% higher than comparator mean; not a statistical or biological noninferiority claim",
            "inference": "3 seeds/one teacher: descriptive means, ranges, paired signs; no population inference, no claim of unique identifiability"},
        "runtime": {"python": sys.version, "torch": torch.__version__, "device": "cpu", "dtype": "float32", "threads_per_fit": 1, "deterministic_algorithms": True},
        "source_sha256": {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in source_paths()},
        "boundaries": ["synthetic only", "no real response or movie loading", "teacher frozen", "no architecture edits", "no extra conditions", "no S1 execution"],
        "invalid_predecessor": {"path": str(OUT.relative_to(ROOT)).replace('\\', '/') + "_invalid_stimulus_overlap",
            "reason": "Discrete grating phase/sign generated exact BC train/heldout/mechanism overlap. Five partial fits terminated before any held-out metric inspection. All excluded; artifacts preserved. Continuous independent phase and block amplitude fix, unchanged valid fit budgets and all other conditions.",
            "scientific_results_inspected": False, "valid_benchmark_updates": 45000,
            "invalid_attempt_compute": "additional discarded partial updates; not represented as valid benchmark results or an outcome-driven extension"},
    }
    OUT.mkdir(parents=True)
    for folder in ("datasets", "datasets/teacher_probabilities", "initial_states", "checkpoints", "curves", "verification"):
        (OUT / folder).mkdir(exist_ok=True)
    write_json(OUT / "protocol.json", protocol)
    teacher = build(metadata, cp["model"]).eval().requires_grad_(False)
    before = state_sha(teacher.state_dict())
    write_json(OUT / "teacher_metadata.json", {**metadata, "cell": "67#7", "seed": 2026091301,
        "checkpoint_sha256": entry["sha256"], "state_sha256": before, "all_parameters_frozen": True,
        "meaning": "known synthetic dynamical system only; not physiological ground truth"})
    norm = {}
    inventory = {}
    seen_sequences, seen_scored = set(), set()
    with torch.no_grad():
        for name, (family, count, seed) in specifications.items():
            x = stimulus(family, count, seed, positions)
            sequences = [tensor_sha(v) for v in x]
            scored = [tensor_sha(v[30:]) for v in x]
            assert len(set(sequences)) == count and len(set(scored)) == count
            assert not (set(sequences) & seen_sequences) and not (set(scored) & seen_scored)
            seen_sequences.update(sequences)
            seen_scored.update(scored)
            payload = {"x": x, "mask": score_mask(count)}
            if name.startswith("D_H"):
                payload["target"] = torch.cat([h_observation(teacher, z, node) for z in x.split(8)])
            elif name.startswith("D_B"):
                payload["target"] = torch.cat([b_observation(teacher, z) for z in x.split(8)])
            elif name.startswith("D_R"):
                events = _sample_spikes(teacher, x, trials=1, seed=protocol["spike_seeds"][name])[:, 0]
                payload["target"] = events
                probabilities = torch.cat([teacher(z, observed_counts=y).spike_probability
                    for z, y in zip(x.split(8), events.split(8), strict=True)])
                save_tensor(OUT / "datasets/teacher_probabilities" / f"{name}.pt", probabilities)
            if name in ("D_H_train", "D_B_train"):
                target = payload["target"][:, 30:].double().reshape(-1, payload["target"].shape[-1])
                mean, std = target.mean(0), target.std(0, correction=0)
                assert bool((std > EPS).all() & torch.isfinite(std).all())
                norm[family] = {"mean": mean.tolist(), "std": std.tolist(), "bins": len(target), "source": name}
            path = OUT / "datasets" / f"{name}.pt"
            save_tensor(path, payload)
            inventory[name] = {"file_sha256": sha(path), "x_sha256": tensor_sha(x), "shape": list(x.shape),
                               "minimum": float(x.min()), "maximum": float(x.max()), "keys": list(payload),
                               "sequence_sha256": sequences, "scored_stimulus_sha256": scored}
            print("generated", name, flush=True)
    write_json(OUT / "normalization.json", norm)
    write_json(OUT / "dataset_manifest.json", inventory)
    initial_hashes = {}
    for seed in SEEDS:
        student = fresh(metadata, seed)
        initial = {k: v.detach().clone() for k, v in student.state_dict().items()}
        initial_hashes[str(seed)] = state_sha(initial)
        trainable = dict(student.named_parameters())
        assert any(not torch.equal(initial[k], cp["model"][k]) for k, p in trainable.items() if p.requires_grad)
        save_tensor(OUT / "initial_states" / f"{seed}.pt", {"model": initial, "seed": seed,
            "protocol_sha256": sha(OUT / "protocol.json"), "initial_sha256": initial_hashes[str(seed)]})
    assert len(set(initial_hashes.values())) == 3
    assert state_sha(teacher.state_dict()) == before
    write_json(OUT / "preparation_complete.json", {"completed_utc": utc(), "protocol_sha256": sha(OUT / "protocol.json"),
        "normalization_sha256": sha(OUT / "normalization.json"), "initial_sha256": initial_hashes,
        "teacher_unchanged": True, "student_optimizer_updates": 0,
        "schedule_sha256": {str(s): {d: tensor_sha(schedule(s, d, 192 if d == "R_EXTRA" else 64))
            for d in ("H", "B", "R", "R_EXTRA")} for s in SEEDS}})


def train(condition, seed):
    protocol = checked_protocol()
    verification = read_json(OUT / "verification/pretraining.json")
    assert verification["status"] == "PASS" and verification["protocol_sha256"] == sha(OUT / "protocol.json")
    assert condition in CONDITIONS and seed in SEEDS
    target_path = OUT / "checkpoints" / f"{condition}_{seed}_final.pt"
    start_path = OUT / "checkpoints" / f"{condition}_{seed}_started.json"
    if target_path.exists() or start_path.exists():
        raise FileExistsError("Do not restart, extend, or overwrite a frozen fit")
    prep = read_json(OUT / "preparation_complete.json")
    assert prep["normalization_sha256"] == sha(OUT / "normalization.json")
    meta = read_json(OUT / "teacher_metadata.json")
    initial = torch.load(OUT / "initial_states" / f"{seed}.pt", weights_only=True)
    assert initial["protocol_sha256"] == sha(OUT / "protocol.json")
    assert state_sha(initial["model"]) == prep["initial_sha256"][str(seed)]
    model = build(meta, initial["model"]).train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=.003, betas=(.9, .999), eps=1e-8, weight_decay=0)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in params}
    datasets, batches = {}, {}
    manifest = read_json(OUT / "dataset_manifest.json")
    for dataset in CONDITIONS[condition]:
        name = f"D_{dataset}_train"
        path = OUT / "datasets" / f"{name}.pt"
        assert sha(path) == manifest[name]["file_sha256"]
        datasets[dataset] = torch.load(path, weights_only=True)
        batches[dataset] = schedule(seed, dataset, len(datasets[dataset]["x"]))
        assert tensor_sha(batches[dataset]) == prep["schedule_sha256"][str(seed)][dataset]
    norm = {key: {name: torch.tensor(value[name], dtype=torch.float32) for name in ("mean", "std")}
            for key, value in read_json(OUT / "normalization.json").items()}
    write_json(start_path, {"started_utc": utc(), "condition": condition, "seed": seed,
        "protocol_sha256": sha(OUT / "protocol.json"), "initial_sha256": initial["initial_sha256"],
        "training_files": [f"datasets/D_{key}_train.pt" for key in CONDITIONS[condition]],
        "heldout_or_teacher_probability_files_read": [], "target_updates": UPDATES})
    started = time.perf_counter()
    curves = []
    gradient_seen = {key: False for key, p in model.named_parameters() if p.requires_grad}
    for step in range(1, UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        for dataset in CONDITIONS[condition]:
            data, idx = datasets[dataset], batches[dataset][step - 1]
            x, target = data["x"][idx], data["target"][idx]
            if dataset in ("R", "R_EXTRA"):
                logits = model(x, observed_counts=target).logits
                loss = expected_bernoulli_nll(logits, target, data["mask"][idx])
            else:
                prediction = (h_observation(model, x, protocol["observation"]["h1_node_index"])
                              if dataset == "H" else b_observation(model, x))
                mean, std = norm[dataset]["mean"], norm[dataset]["std"]
                loss = (((prediction[:, 30:] - mean) / std - (target[:, 30:] - mean) / std)**2).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite {condition}/{seed}/{step}/{dataset}")
            loss.backward()
            losses[dataset] = float(loss.detach())
        for key, parameter in model.named_parameters():
            if parameter.requires_grad:
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError(f"Missing or invalid gradient {key}")
                gradient_seen[key] |= bool(torch.count_nonzero(parameter.grad))
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 5, error_if_nonfinite=True))
        optimizer.step()
        model.project_mechanism_parameters()
        if step == 1 or step % 25 == 0:
            curves.append({"condition": condition, "seed": seed, "step": step,
                "L_H": losses.get("H"), "L_B": losses.get("B"),
                "L_R": losses.get("R", losses.get("R_EXTRA")), "total_loss": sum(losses.values()),
                "gradient_norm_before_clip": grad_norm, "elapsed_seconds": time.perf_counter() - started})
        if step % 250 == 0:
            print(condition, seed, step, "seconds", round(time.perf_counter() - started, 1), flush=True)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    assert all(int(optimizer.state[p]["step"]) == UPDATES for p in params)
    assert all(torch.isfinite(p).all() for p in state.values())
    write_csv(OUT / "curves" / f"{condition}_{seed}.csv", curves)
    save_tensor(target_path, {"model": state, "optimizer": optimizer.state_dict(), "condition": condition,
        "seed": seed, "step": UPDATES, "protocol_sha256": sha(OUT / "protocol.json"),
        "initial_sha256": initial["initial_sha256"], "state_sha256": state_sha(state),
        "gradient_nonzero_seen": gradient_seen,
        "parameter_changed": {key: not torch.equal(initial["model"][key], value.detach())
            for key, value in model.named_parameters() if value.requires_grad},
        "finished_utc": utc(), "elapsed_seconds": time.perf_counter() - started})
    print("COMPLETE", condition, seed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train"))
    parser.add_argument("--condition", choices=tuple(CONDITIONS))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    configure_runtime()
    if args.action == "prepare":
        prepare()
    else:
        train(args.condition, args.seed)
