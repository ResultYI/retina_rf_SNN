from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

from experiments.retipath_multiscale_v0 import synthetic_study as io
from .circuit import BCOutput, Intervention, PopulationRetipath, Stimulus


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "output/synthetic/retipath_population_stage_b_20260918"
CONDITIONS = {"A": "RGC_ONLY", "C": "MULTILEVEL_PARTIAL", "E": "EXTRA_RGC_CONTROL"}
SEEDS = (4101, 4102, 4103)
H_NODES = (12, 10, 14, 2, 22)
BC_NODES = H_NODES + tuple(i + 25 for i in H_NODES)
TRAIN_SCALES = (.15, .3, .6)
TEST_SCALES = (.225, .45)
STEPS = 400
WARMUP = 60
SIGMA = .03
WEIGHTS = {"base_R": 1 / 3, "base_R_1": 1 / 3, "base_R_2": 1 / 3,
           "H": .4, "BC": .4, "extra_R_1": 1 / 3, "extra_R_2": 1 / 3}
PRIMARY = "direct_delta_logit_rmse"
PORTS = ("h_H", "s_B", "delta_r_B", "H1_feedback", "u_A", "a_A", "delta_o_A",
         "d_E", "d_I", "gE", "gI", "V", "logit", "probability")
COUPLINGS = ("a_H", "gamma_BR", "gamma_BA", "gamma_AR")


def model() -> PopulationRetipath:
    return PopulationRetipath(bc_output=BCOutput.LEGACY_PRELU, dtype=torch.float32)


def stimulus(data: dict, indices: list[int]) -> Stimulus:
    values = data["temporal"][indices, :, None] * data["spatial"][indices, None, :]
    return Stimulus(values, io.BOUNDS, io.AREA, io.TIME, torch.ones_like(values, dtype=torch.bool))


def check_sources(run: Path) -> dict:
    protocol = io.read_json(run / "protocol.json")
    lock = io.read_json(run / "SOURCE_LOCK.json")
    assert io.digest(run / "protocol.json") == lock["protocol_sha256"]
    for name, expected in protocol["source_hashes"].items():
        assert io.digest(ROOT / name) == expected, name
    return protocol


def parameter_values(net: PopulationRetipath) -> dict:
    return {name: value.detach().clone() for name, value in net.physical_parameters().items()}


def initialized(key: str, *, teacher: bool) -> PopulationRetipath:
    net = model()
    with torch.no_grad():
        for name, field in net.fields.items():
            sd = (field.center_sd or .25) if teacher else .1
            field.center.add_(sd * torch.randn(field.center.shape, generator=io.generator(key + name + ":center")))
            if teacher and field.contrast is not None:
                field.contrast.copy_(field.sd * torch.randn(field.contrast.shape, generator=io.generator(key + name + ":contrast")))
    return net


def freeze(run: Path) -> None:
    paths = list((ROOT / "experiments/retipath_multiscale_v0").glob("*.py"))
    paths += list((ROOT / "experiments/retipath_population_v0_1").glob("*.py"))
    paths += [ROOT / name for name in (
        "experiments/retipath_multiscale_v0/protocol.json", "models/mechanistic_retina/state.py",
        "models/mechanistic_retina/retipath_spatial_ei.py", "AGENTS.md", "docs/RETIPATH_POPULATION_ARCHITECTURE_V0_1.md",
        "docs/RETIPATH_POPULATION_STAGE_A.md", "docs/RETIPATH_POPULATION_STAGE_A5_MIGRATION.md")]
    sources = {p.relative_to(ROOT).as_posix(): io.digest(p) for p in sorted(set(paths))}
    protocol = {
        "schema": "population_stage_b_v1", "frozen_utc": io.utc(), "conditions": CONDITIONS,
        "worlds": [f"W{i:02}" for i in range(1, 6)], "student_seeds": SEEDS, "source_hashes": sources,
        "architecture": "Unmodified Population v0.1, LegacyPReLU, 359 trainable scalars; all joint from step1; fixed Q/routing/conductance/interventions. No softplus comparison.",
        "teacher": {
            "namespace": "PopulationStageB:Wxx", "family": "realizable PopulationRetipath",
            "center_draw": "independent Gaussian raw perturbations around constructor centers: sd=field.center_sd when present, otherwise0.25",
            "individual_draw": "orthonormal zero-sum raw contrast coefficients iid N(0,field.sd^2): H tau/delay0.3, a_H0.1, BC0.6. Existing bounded sigmoid and priors unchanged",
            "AC": "family-shared tau_A/delay_A/b_A; no unit deviations", "gamma_BA": "fixed1, not estimated",
            "selection": "exactly5 initial draws, no response screening, rejection or replacement"},
        "initialization": "new independent namespace per world/seed; constructor centers plus iid N(0,0.1^2), contrasts zero; same exact state file for A/C/E; no teacher access",
        "stimulus": {"grid": [32, 32], "field_deg": [-1, 1, -1, 1], "T": 300, "dt_ms": io.DT_MS,
                     "warmup_bins": WARMUP, "scored_bins": 240, "train_scales": TRAIN_SCALES, "test_scales": TEST_SCALES,
                     "definition": "Reuse G2 physical_arrays and make_records: pixel-area Gaussian average, centers in{-0.1,0,0.1}deg, contrast0.4 times mean four sine waves at0.5/1.5/3/6Hz. New namespace and independent phases; no old samples",
                     "reset": "independent baseline reset for every sequence; full known field, no crops or missing pixels"},
        "data": {"base_sequences": 72, "base_per_scale": 24, "extra_sequences": 144, "extra_per_scale": 48,
                 "test_sequences": 16, "test_per_scale": 8, "development_sequences": 0,
                 "H_nodes": H_NODES, "BC_nodes": BC_NODES, "H_total": 25, "BC_total": 50, "RGC_total": 2,
                 "subset_rule": "user-fixed physical coordinates (0,0),(+0.30,0),(-0.30,0),(0,+0.30),(0,-0.30) deg in H and each BC polarity;20% each, same identities across worlds; no response selection",
                 "C_targets": "H:h_H; BC:stack(s_B,delta_r_B) on10 nodes; independent iid Gaussian noise sigma0.03 on each scalar once. All3 training scales visible at these nodes",
                 "R_targets": "both RGC causal Bernoulli events, no probability targets; same base observations A/C/E; E two disjoint extra pools of72 families each (24 per scale), independent from base and each other",
                 "visibility": "training layer files contain only sliced noisy targets; full test traces and teacher state sealed under evaluator_only; no teacher/full latent loaded by workers"},
        "optimization": {"steps": STEPS, "batch_size": 4, "optimizer": "Adam", "lr": .01, "betas": [.9, .999], "eps": 1e-8,
                         "weight_decay": 0, "clip_grad_norm": 1., "weights": WEIGHTS,
                         "data_loss": "mean over sequences/time60:300/visible nodes; Bernoulli BCE logits for R;0.5*((prediction-target)/0.03)^2 for H and BC (BC mean includes both ports)",
                         "hierarchy": "add unmodified hierarchy_penalty once per step, coefficient1; no rescaling by dataset count; center anchors and all deviation priors retained",
                         "schedule": "base_R every step all conditions; A adds two base-R repeat microbatches; C adds H and BC; E adds two independent extra-R microbatches. Separate frozen streams rotate identical scale order and shuffle without replacement per scale cycle; streams and step indices shared across paired seeds",
                         "selection": "final step400 only; no validation, early stopping, tuning, extension, rescue retry or best checkpoint",
                         "nonfinite": "stop affected fit, retain failure; no redraw, changed weight or extra steps"},
        "exposure": {"A": {"batches": 1200, "sequence": 4800, "scalars": 2304000},
                     "C": {"batches": 1200, "sequence": 4800, "scalars": 10368000},
                     "E": {"batches": 1200, "sequence": 4800, "scalars": 2304000},
                     "common_base_R": {"batches": 400, "sequence": 1600, "scalars": 768000},
                     "limit": "A/C/E exposure-matched:400 updates,1200 sequence microbatches,4800 sequence presentations,1152000 scored sequence-time bins each; not information-matched, scalar-count-matched or compute-matched. A has3x RGC exposure from base pool; C and E share400 base-R batches; E adds800 extra-R batches. Total-gradient clipping acts on the sum"},
        "gradient_diagnostics": "every actual dataset batch, same parameter snapshot before clipping: raw/weighted L2 per state/output/coupling/all; pair cosine on shared autograd-supported parameter coordinates, record counts; N/A if disjoint, UNDEFINED_ZERO_NORM if zero. Hierarchy separately. No gradient surgery/reweighting/extra optimizer updates",
        "evaluation": {"primary": PRIMARY, "delta": "blocked_logit-normal_logit; BLOCK_DIRECT_BC_DRIVE for whole sequence, same teacher-normal conditioning events, baseline reset; conditional intervention, not autonomous events or drug block",
                       "secondary": ["rgc_excess_ce", "observed/unobserved h_H", "observed/unobserved s_B/delta_r_B", "d_E", "d_I", "coupling parameters separately", "H1_feedback/u_A and H1/AC block effects separately", "cross_seed_ambiguity"],
                       "reduction": "per-sequence RMSE over bins60:300 and indicated nodes/modes, equal sequence mean within scale then equal two heldout scale means; excess CE from teacher conditional p minus its Bernoulli entropy. Parameter errors per named physical coupling field, never mixed with states",
                       "lock": "freeze all45 final checkpoints plus completions before opening evaluator_only in scoring; global TEST_CONSUMED marker precedes first test/teacher load",
                       "comparison": "C-A and C-E per paired seed, per teacher mean, equal descriptive mean over5teachers; report signs without p-values/success threshold/biological population inference",
                       "ambiguity": "three pairwise seed RMSE distances for each trajectory/parameter quantity, same test, separate from truth accuracy; no composite score",
                       "RF": "not run", "stop": "StageB delivery only; no StageC"},
        "runtime": {"dtype": "float32", "metric_dtype": "float64", "device": "cpu", "threads_per_process": 1,
                    "max_processes": 3, "deterministic_algorithms": True, "torch": torch.__version__, "python": sys.version},
        "total_fits": 45, "total_optimizer_steps": 18000,
    }
    run.mkdir(parents=True, exist_ok=False)
    io.write_json(run / "protocol.json", protocol)
    io.write_json(run / "SOURCE_LOCK.json", {"utc": io.utc(), "protocol_sha256": io.digest(run / "protocol.json"), "sources": sources})
    for rel in sources:
        destination = run / "source" / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write((ROOT / rel).read_bytes())
    families = set()
    for world in protocol["worlds"]:
        directory = run / "worlds" / world
        key = "PopulationStageB:" + world
        manifest = {"base": io.make_records(key + ":base", list(TRAIN_SCALES), 24, "train"),
                    "extra_R_1": io.make_records(key + ":extra1", list(TRAIN_SCALES), 24, "train"),
                    "extra_R_2": io.make_records(key + ":extra2", list(TRAIN_SCALES), 24, "train"),
                    "test": io.make_records(key + ":test", list(TEST_SCALES), 8, "test")}
        local_families = [r["family"] for records in manifest.values() for r in records]
        assert len(set(local_families)) == 232 and not families.intersection(local_families)
        families.update(local_families)
        streams = {name: io.make_stream(manifest[pool], STEPS, key + ":schedule:" + name)
                   for name, pool in (("base_R", "base"), ("base_R_1", "base"), ("base_R_2", "base"),
                                      ("H", "base"), ("BC", "base"), ("extra_R_1", "extra_R_1"), ("extra_R_2", "extra_R_2"))}
        io.write_json(directory / "STIMULUS_MANIFEST.json", manifest)
        io.write_json(directory / "SCHEDULES.json", streams)
    print("FROZEN", run, flush=True)


def trace_arrays(net: PopulationRetipath, data: dict, indices: list[int]) -> dict:
    x, events = stimulus(data, indices), data["events"][indices]
    normal = net(x, observed_events=events)
    arrays = {name: (normal.inputs[name] if name in normal.inputs else normal.observation(name)).detach() for name in PORTS}
    for name, intervention in (("direct", Intervention.BLOCK_DIRECT_BC_DRIVE),
                               ("H1", Intervention.BLOCK_H1_FEEDBACK), ("AC", Intervention.BLOCK_AC_POSTSYNAPTIC_DRIVE)):
        blocked = net(x, observed_events=events, intervention=intervention)
        arrays[name + "_delta_logit"] = blocked.outputs["logit"] - normal.outputs["logit"]
        if name == "direct":
            assert torch.equal(blocked.outputs["d_I"], normal.outputs["d_I"])
            assert torch.equal(blocked.outputs["gE"], torch.ones_like(blocked.outputs["gE"]))
    return arrays


def generate(run: Path) -> None:
    protocol = check_sources(run)
    io.write_json(run / "GENERATION_STARTED.json", {"utc": io.utc()})
    all_files, sampler_errors, heterogeneity = {}, {}, []
    for world in protocol["worlds"]:
        directory = run / "worlds" / world
        key = "PopulationStageB:" + world
        teacher = initialized(key + ":teacher:", teacher=True).requires_grad_(False).eval()
        io.save_tensor(directory / "evaluator_only/teacher.pt", {"model_state": teacher.state_dict(), "physical_parameters": parameter_values(teacher)})
        for name, field in teacher.fields.items():
            value = field().detach()
            for family in field.family.unique().tolist():
                group = value[field.family == family]
                heterogeneity.append({"world": world, "field": name, "family": family, "units": group.numel(),
                                      "mean": float(group.mean()), "sd": float(group.std(unbiased=False)), "min": float(group.min()), "max": float(group.max())})
        for seed in SEEDS:
            initial = initialized(key + f":student:{seed}:", teacher=False)
            io.save_tensor(directory / f"initial_students/{seed}.pt", {"model_state": initial.state_dict(), "seed": seed})
        manifest = io.read_json(directory / "STIMULUS_MANIFEST.json")
        errors = {}
        for pool_name, records in manifest.items():
            spatial, temporal = io.physical_arrays(records)
            inputs = {"records": records, "spatial": spatial, "temporal": temporal}
            collected = defaultdict(list)
            largest = 0.
            for start in range(0, len(records), 4):
                ids = list(range(start, min(start + 4, len(records))))
                x = stimulus(inputs, ids)
                with torch.no_grad():
                    base = teacher(x, observed_events=torch.zeros(len(ids), 300, 2))
                    uniforms = torch.stack([torch.rand(300, 2, generator=io.generator(records[i]["family"] + ":events")) for i in ids])
                    events, probability = io.causal_sample(base.outputs["logit"], uniforms)
                    normal = teacher(x, observed_events=events)
                    error = float((probability - normal.outputs["probability"]).abs().max())
                    largest = max(largest, error)
                    assert error < 2e-7 and torch.equal(events, (uniforms < normal.outputs["probability"]).float())
                    collected["events"].append(events)
                    if pool_name == "base":
                        for layer, clean in (("H", normal.states["h_H"][..., list(H_NODES)]),
                                             ("BC", torch.stack([normal.states["s_B"][..., list(BC_NODES)], normal.outputs["delta_r_B"][..., list(BC_NODES)]], -1))):
                            noise = torch.stack([torch.randn(clean.shape[1:], generator=io.generator(records[i]["family"] + ":noise:" + layer)) for i in ids])
                            collected[layer].append(clean + SIGMA * noise)
                    if pool_name == "test":
                        batch = {"records": [records[i] for i in ids], "spatial": spatial[ids], "temporal": temporal[ids], "events": events}
                        for name, value in trace_arrays(teacher, batch, list(range(len(ids)))).items():
                            collected[name].append(value)
            errors[pool_name] = largest
            joined = {name: torch.cat(values) for name, values in collected.items()}
            if pool_name == "test":
                io.save_tensor(directory / "evaluator_only/test.pt", {**inputs, **joined})
            else:
                dataset_name = "base_R" if pool_name == "base" else pool_name
                io.save_tensor(directory / f"train_data/{dataset_name}.pt", {**inputs, "events": joined["events"], "targets": joined["events"]})
                if pool_name == "base":
                    for layer in ("H", "BC"):
                        io.save_tensor(directory / f"train_data/{layer}.pt", {**inputs, "targets": joined[layer]})
        files = {p.relative_to(directory).as_posix(): io.digest(p) for p in directory.rglob("*") if p.is_file()}
        io.write_json(directory / "DATA_LOCK.json", {"utc": io.utc(), "files": files, "sampler_max_abs_error": errors})
        all_files[world] = io.digest(directory / "DATA_LOCK.json")
        sampler_errors[world] = errors
        print("DATA_FROZEN", world, flush=True)
    io.write_csv(run / "teacher_heterogeneity.csv", heterogeneity)
    io.write_json(run / "GENERATION_COMPLETED.json", {"utc": io.utc(), "world_data_locks": all_files, "sampler_errors": sampler_errors, "all_data_before_any_training": True})
    check_sources(run)


def dataset_loss(net: PopulationRetipath, data: dict, name: str, indices: list[int]) -> torch.Tensor:
    events = data["events"][indices] if "events" in data else torch.zeros(len(indices), 300, 2)
    trace = net(stimulus(data, indices), observed_events=events)
    target = data["targets"][indices, WARMUP:]
    if name == "H":
        pred = trace.states["h_H"][:, WARMUP:, list(H_NODES)]
    elif name == "BC":
        pred = torch.stack([trace.states["s_B"][:, WARMUP:, list(BC_NODES)], trace.outputs["delta_r_B"][:, WARMUP:, list(BC_NODES)]], -1)
    else:
        return F.binary_cross_entropy_with_logits(trace.outputs["logit"][:, WARMUP:], target)
    return .5 * ((pred - target) / SIGMA).square().mean()


def gradient_vector(loss: torch.Tensor, params: tuple, *, retain_graph=False) -> tuple[torch.Tensor, torch.Tensor]:
    grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=retain_graph)
    values = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p, g in zip(params, grads)])
    support = torch.cat([torch.full((p.numel(),), g is not None, dtype=torch.bool) for p, g in zip(params, grads)])
    assert bool(torch.isfinite(values).all())
    return values, support


def cosine(left, right, mask) -> tuple[float | None, str, int]:
    shared = mask & left[1] & right[1]
    n = int(shared.sum())
    if not n:
        return None, "N/A", n
    x, y = left[0][shared].double(), right[0][shared].double()
    denominator = float(x.norm() * y.norm())
    if denominator == 0:
        return None, "UNDEFINED_ZERO_NORM", n
    return float(torch.dot(x, y)) / denominator, "DEFINED", n


def group_masks(net: PopulationRetipath) -> dict:
    groups = {id(p): field.group for field in net.fields.values() for p in field.parameters()}
    labels = [groups[id(p)] for p in net.parameters() for _ in range(p.numel())]
    return {group: torch.tensor([group == "all" or value == group for value in labels]) for group in ("all", "state", "output", "coupling")}


def train_fit(run: Path, world: str, condition: str, seed: int) -> None:
    check_sources(run)
    assert (run / "GENERATION_COMPLETED.json").exists() and not (run / "CHECKPOINT_LOCK.json").exists()
    directory = run / "worlds" / world
    creation = io.read_json(directory / "DATA_LOCK.json")
    schedule = io.read_json(directory / "SCHEDULES.json")
    assert io.digest(directory / "SCHEDULES.json") == creation["files"]["SCHEDULES.json"]
    names = {"A": ["base_R", "base_R_1", "base_R_2"], "C": ["base_R", "H", "BC"], "E": ["base_R", "extra_R_1", "extra_R_2"]}[condition]
    dataset_files = {n: "base_R" if n.startswith("base_R") else n for n in names}
    relative_files = {f"train_data/{n}.pt" for n in dataset_files.values()} | {f"initial_students/{seed}.pt"}
    allowed = {str((directory / name).resolve()).lower() for name in relative_files}
    own_checkpoint = str((directory / f"fits/{condition}_{seed}/final.pt").resolve()).lower()
    accessed = set()

    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        filename = str(Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()).lower()
        if "evaluator_only" in filename:
            raise PermissionError("Training must not open evaluator_only")
        if filename == own_checkpoint:
            return
        if filename.endswith(".pt") and not any(char in str(args[1]) for char in ("w", "x", "a")):
            if filename not in allowed:
                raise PermissionError("Unapproved training tensor read: " + filename)
            accessed.add(filename)

    sys.addaudithook(audit)
    datasets = {}
    for name in names:
        rel = f"train_data/{dataset_files[name]}.pt"
        assert io.digest(directory / rel) == creation["files"][rel]
        data = io.load_tensor(directory / rel)
        expected_keys = {"records", "spatial", "temporal", "targets"} | ({"events"} if "R" in name else set())
        assert set(data) == expected_keys
        assert data["targets"].shape[1:] == ((300, 5) if name == "H" else (300, 10, 2) if name == "BC" else (300, 2))
        datasets[name] = data
    initial_path = directory / f"initial_students/{seed}.pt"
    assert io.digest(initial_path) == creation["files"][f"initial_students/{seed}.pt"]
    net = model()
    net.load_state_dict(io.load_tensor(initial_path)["model_state"], strict=True)
    params = tuple(net.parameters())
    initial_params = {n: p.detach().clone() for n, p in net.named_parameters()}
    masks = group_masks(net)
    optimizer = torch.optim.Adam(params, lr=.01, betas=(.9, .999), eps=1e-8, weight_decay=0)
    dest = directory / f"fits/{condition}_{seed}"
    dest.mkdir(parents=True, exist_ok=False)
    trajectories, norms, cosines = [], [], []
    counts, nonzero = Counter(), torch.zeros(sum(p.numel() for p in params), dtype=torch.bool)
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        vectors, losses = {}, {}
        for name in names:
            loss = dataset_loss(net, datasets[name], name, schedule[name][step - 1])
            assert bool(torch.isfinite(loss))
            raw, support = gradient_vector(loss, params)
            vectors[name] = (raw * WEIGHTS[name], support)
            losses[name] = float(loss.detach())
            counts[name] += 1
            for group, mask in masks.items():
                norms.append({"step": step, "dataset": name, "group": group, "raw_norm": float(raw[mask].double().norm()),
                              "weighted_norm": float(vectors[name][0][mask].double().norm()), "supported_coordinates": int((mask & support).sum())})
        for left, right in itertools.combinations(names, 2):
            for group, mask in masks.items():
                value, status, coordinates = cosine(vectors[left], vectors[right], mask)
                cosines.append({"step": step, "left": left, "right": right, "group": group,
                                "cosine": value, "status": status, "shared_coordinates": coordinates})
        prior = net.hierarchy_penalty()
        prior_gradient, prior_support = gradient_vector(prior, params)
        total = prior_gradient.clone()
        for gradient, _ in vectors.values():
            total += gradient
        nonzero |= total != 0
        for group, mask in masks.items():
            value = float(prior_gradient[mask].double().norm())
            norms.append({"step": step, "dataset": "hierarchy", "group": group, "raw_norm": value, "weighted_norm": value,
                          "supported_coordinates": int((mask & prior_support).sum())})
        offset = 0
        for p in params:
            p.grad = total[offset:offset + p.numel()].reshape_as(p).clone()
            offset += p.numel()
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        assert all(bool(torch.isfinite(p).all()) for p in params)
        trajectories.append({"step": step, "loss_base_R": losses["base_R"], "loss_base_R_1": losses.get("base_R_1"), "loss_base_R_2": losses.get("base_R_2"), "loss_H": losses.get("H"), "loss_BC": losses.get("BC"),
                             "loss_extra_R_1": losses.get("extra_R_1"), "loss_extra_R_2": losses.get("extra_R_2"), "hierarchy": float(prior.detach()),
                             "weighted_data_loss": sum(WEIGHTS[n] * v for n, v in losses.items()), "preclip_norm": float(norm), "elapsed_seconds": time.perf_counter() - started})
        if step % 50 == 0:
            print("TRAIN", world, condition, seed, step, "elapsed", round(time.perf_counter() - started, 1), flush=True)
    assert all(int(state["step"]) == STEPS for state in optimizer.state.values())
    changes = {n: float((p.detach() - initial_params[n]).abs().max()) for n, p in net.named_parameters()}
    io.save_tensor(dest / "final.pt", {"model_state": net.state_dict(), "optimizer_state": optimizer.state_dict(), "condition": condition,
                                     "world": world, "seed": seed, "steps": STEPS, "initial_sha256": io.digest(initial_path), "protocol_sha256": io.digest(run / "protocol.json")})
    for name, rows in (("trajectory", trajectories), ("gradient_norms", norms), ("gradient_cosines", cosines)):
        if rows:
            io.write_csv(dest / f"{name}.csv", rows)
    io.write_json(dest / "completed.json", {"status": "VERIFIED", "utc": io.utc(), "condition": condition, "world": world, "seed": seed,
                  "steps": STEPS, "dataset_batches": dict(counts), "sequence_exposures": sum(counts.values()) * 4,
                  "checkpoint_sha256": io.digest(dest / "final.pt"), "initial_sha256": io.digest(initial_path),
                  "raw_max_abs_parameter_changes": changes, "ever_nonzero_total_gradient_scalars": int(nonzero.sum()),
                  "trainable_scalars": total.numel(), "optimizer_listed_scalars": sum(p.numel() for p in params),
                  "loaded_tensor_files": sorted(accessed), "evaluator_reads": 0, "elapsed_seconds": time.perf_counter() - started})
    check_sources(run)


def train(run: Path) -> None:
    protocol = check_sources(run)
    generation = io.read_json(run / "GENERATION_COMPLETED.json")
    for world, expected in generation["world_data_locks"].items():
        assert io.digest(run / "worlds" / world / "DATA_LOCK.json") == expected
    tasks = [(w, c, s) for w in protocol["worlds"] for s in SEEDS for c in CONDITIONS]
    io.write_json(run / "TRAINING_STARTED.json", {"utc": io.utc(), "fits": tasks, "attempts_per_fit": 1})
    (run / "logs").mkdir(exist_ok=False)

    def launch(task):
        w, c, s = task
        with (run / "logs" / f"{w}_{c}_{s}.log").open("x", encoding="utf-8") as out:
            result = subprocess.run([sys.executable, "-B", "-m", "experiments.retipath_population_v0_1.stage_b", "worker", "--run", str(run),
                                     "--world", w, "--condition", c, "--seed", str(s)], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, check=False)
        return {"world": w, "condition": c, "seed": s, "returncode": result.returncode}

    completions = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(launch, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            completions.append(result)
            print("FIT_FINISHED", len(completions), "/45", result, flush=True)
    io.write_json(run / "PROCESS_COMPLETIONS.json", completions)
    assert len(completions) == 45 and all(x["returncode"] == 0 for x in completions), "Training failure retained; no rescue retry"
    locked = {}
    for w, c, s in tasks:
        directory = run / f"worlds/{w}/fits/{c}_{s}"
        complete = io.read_json(directory / "completed.json")
        assert complete["steps"] == 400 and sum(complete["dataset_batches"].values()) == 1200
        assert complete["checkpoint_sha256"] == io.digest(directory / "final.pt")
        for filename in ("final.pt", "completed.json"):
            path = directory / filename
            locked[path.relative_to(run).as_posix()] = io.digest(path)
    check_sources(run)
    io.write_json(run / "CHECKPOINT_LOCK.json", {"utc": io.utc(), "fits": 45, "optimizer_steps": 18000, "files": locked,
                  "all_final_only": True, "before_student_test_access": True})
    print("ALL_45_CHECKPOINTS_LOCKED", flush=True)


def rms(value: torch.Tensor) -> float:
    return float(value.double().square().mean().sqrt())


def selections() -> dict:
    h_hidden = [i for i in range(25) if i not in H_NODES]
    b_hidden = [i for i in range(50) if i not in BC_NODES]
    result = {f"{prefix}_{port}_rmse": (port, list(nodes)) for prefix, ports, nodes in (
        ("observed", ("h_H",), H_NODES), ("unobserved", ("h_H",), h_hidden),
        ("observed", ("s_B", "delta_r_B"), BC_NODES), ("unobserved", ("s_B", "delta_r_B"), b_hidden)) for port in ports}
    result.update({port + "_rmse": (port, None) for port in ("d_E", "d_I", "H1_feedback", "u_A", "a_A", "delta_o_A", "direct_delta_logit", "H1_delta_logit", "AC_delta_logit", "probability")})
    return result


def sequence_errors(pred: dict, truth: dict, index: int, *, ambiguity=False) -> dict:
    result = {}
    for name, (port, nodes) in selections().items():
        difference = pred[port][index, WARMUP:].double() - truth[port][index, WARMUP:].double()
        if nodes is not None:
            difference = difference[..., nodes]
        result[name] = rms(difference)
    if not ambiguity:
        p = truth["probability"][index, WARMUP:].double()
        ell = pred["logit"][index, WARMUP:].double()
        target_ell = truth["logit"][index, WARMUP:].double()
        result["rgc_excess_ce"] = float((F.softplus(ell) - p * ell - (F.softplus(target_ell) - p * target_ell)).mean())
    return result


def aggregate(sequence_rows: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for row in sequence_rows:
        key = (row["world"], row["condition"], row["seed"], row["metric"])
        buckets[key].append(row)
    result = []
    for (world, condition, seed, metric), rows in sorted(buckets.items()):
        scale_means = {scale: statistics.mean(r["value"] for r in rows if r["scale"] == scale) for scale in TEST_SCALES}
        result.append({"world": world, "condition": condition, "seed": seed, "metric": metric,
                       "value": statistics.mean(scale_means.values())})
    return result


def comparisons(fits: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    lookup = {(r["world"], r["condition"], r["seed"], r["metric"]): r["value"] for r in fits}
    paired, by_world, overall = [], [], []
    for world, seed, metric in sorted({(r["world"], r["seed"], r["metric"]) for r in fits}):
        for other in ("A", "E"):
            c, baseline = lookup[(world, "C", seed, metric)], lookup[(world, other, seed, metric)]
            paired.append({"world": world, "seed": seed, "metric": metric, "comparison": "C-" + other,
                           "C": c, "reference": baseline, "difference": c - baseline})
    for world, metric, comparison in sorted({(r["world"], r["metric"], r["comparison"]) for r in paired}):
        rows = [r for r in paired if (r["world"], r["metric"], r["comparison"]) == (world, metric, comparison)]
        by_world.append({"world": world, "metric": metric, "comparison": comparison, "C": statistics.mean(r["C"] for r in rows),
                         "reference": statistics.mean(r["reference"] for r in rows), "difference": statistics.mean(r["difference"] for r in rows),
                         "negative_seeds": sum(r["difference"] < 0 for r in rows), "zero_seeds": sum(r["difference"] == 0 for r in rows), "n_seeds": len(rows)})
    for metric, comparison in sorted({(r["metric"], r["comparison"]) for r in by_world}):
        rows = [r for r in by_world if (r["metric"], r["comparison"]) == (metric, comparison)]
        overall.append({"metric": metric, "comparison": comparison, "C": statistics.mean(r["C"] for r in rows),
                        "reference": statistics.mean(r["reference"] for r in rows), "difference": statistics.mean(r["difference"] for r in rows),
                        "negative_worlds": sum(r["difference"] < 0 for r in rows), "zero_worlds": sum(r["difference"] == 0 for r in rows),
                        "negative_paired_seeds": sum(r["negative_seeds"] for r in rows), "zero_paired_seeds": sum(r["zero_seeds"] for r in rows),
                        "n_worlds": len(rows), "n_paired_seeds": sum(r["n_seeds"] for r in rows),
                        "world_delta_min": min(r["difference"] for r in rows), "world_delta_max": max(r["difference"] for r in rows)})
    return paired, by_world, overall


def evaluate(run: Path) -> None:
    protocol = check_sources(run)
    lock = io.read_json(run / "CHECKPOINT_LOCK.json")
    assert lock["fits"] == 45
    for name, expected in lock["files"].items():
        assert io.digest(run / name) == expected
    io.write_json(run / "TEST_CONSUMED.json", {"utc": io.utc(), "checkpoint_lock_sha256": io.digest(run / "CHECKPOINT_LOCK.json"),
                  "scope": "all5 worlds, all45 final students,16 heldout sequences/world; no future selection"})
    seq_rows, parameter_rows, ambiguity_seq, ambiguity_parameters, raw_parameter_rows = [], [], [], [], []
    truth_replay_errors = {}
    for world in protocol["worlds"]:
        directory = run / "worlds" / world
        data_lock = io.read_json(directory / "DATA_LOCK.json")
        for name in ("evaluator_only/test.pt", "evaluator_only/teacher.pt"):
            assert io.digest(directory / name) == data_lock["files"][name]
        data = io.load_tensor(directory / "evaluator_only/test.pt")
        teacher_state = io.load_tensor(directory / "evaluator_only/teacher.pt")
        teacher = model().requires_grad_(False)
        teacher.load_state_dict(teacher_state["model_state"], strict=True)
        with torch.no_grad():
            replay = trace_arrays(teacher, data, list(range(4)))
        truth_replay_errors[world] = {name: float((value - data[name][:4]).abs().max()) for name, value in replay.items()}
        assert max(truth_replay_errors[world].values()) == 0
        predictions = {}
        for condition in CONDITIONS:
            predictions[condition] = {}
            for seed in SEEDS:
                net = model().requires_grad_(False)
                checkpoint = io.load_tensor(directory / f"fits/{condition}_{seed}/final.pt")
                assert checkpoint["steps"] == STEPS
                net.load_state_dict(checkpoint["model_state"], strict=True)
                parts = defaultdict(list)
                with torch.no_grad():
                    for start in range(0, 16, 4):
                        for name, value in trace_arrays(net, data, list(range(start, start + 4))).items():
                            parts[name].append(value)
                pred = {name: torch.cat(values) for name, values in parts.items()}
                pred["parameters"] = parameter_values(net)
                assert all(bool(torch.isfinite(v).all()) for n, v in pred.items() if n != "parameters")
                io.save_tensor(directory / f"evaluation/{condition}_{seed}_raw.pt", pred)
                predictions[condition][seed] = pred
                for index, record in enumerate(data["records"]):
                    for metric, value in sequence_errors(pred, data, index).items():
                        seq_rows.append({"world": world, "condition": condition, "seed": seed, "sequence": record["id"], "scale": record["scale"], "metric": metric, "value": value})
                for name in COUPLINGS:
                    estimate, truth = pred["parameters"][name], teacher_state["physical_parameters"][name]
                    parameter_rows.append({"world": world, "condition": condition, "seed": seed, "metric": name + "_parameter_rmse", "value": rms(estimate - truth)})
                    for i, (t, p) in enumerate(zip(truth.flatten().tolist(), estimate.flatten().tolist())):
                        raw_parameter_rows.append({"world": world, "condition": condition, "seed": seed, "parameter": name, "coordinate": i, "teacher": t, "student": p, "difference": p - t, "fixed": name == "gamma_BA"})
            for left, right in itertools.combinations(SEEDS, 2):
                x, y = predictions[condition][left], predictions[condition][right]
                for i, record in enumerate(data["records"]):
                    for metric, value in sequence_errors(x, y, i, ambiguity=True).items():
                        ambiguity_seq.append({"world": world, "condition": condition, "seed": f"{left}-{right}", "sequence": record["id"], "scale": record["scale"], "metric": metric, "value": value})
                for name in COUPLINGS:
                    ambiguity_parameters.append({"world": world, "condition": condition, "seed": f"{left}-{right}", "metric": name + "_parameter_rmse", "value": rms(x["parameters"][name] - y["parameters"][name])})
        print("EVALUATED", world, flush=True)
    fits = aggregate(seq_rows) + parameter_rows
    paired, by_world, overall = comparisons(fits)
    ambiguity = aggregate(ambiguity_seq) + ambiguity_parameters
    amb_paired, amb_world, amb_overall = comparisons(ambiguity)
    tables = {"per_sequence": seq_rows, "per_fit": fits, "paired_differences": paired, "teacher_summary": by_world,
              "descriptive_summary": overall, "coupling_coordinates": raw_parameter_rows, "ambiguity_per_sequence": ambiguity_seq,
              "ambiguity_per_seed_pair": ambiguity, "ambiguity_paired_differences": amb_paired, "ambiguity_teacher_summary": amb_world,
              "ambiguity_descriptive_summary": amb_overall}
    for name, rows in tables.items():
        io.write_csv(run / f"{name}.csv", rows)
    io.write_json(run / "results.json", {"protocol_sha256": io.digest(run / "protocol.json"), "primary": PRIMARY, **tables})
    io.write_json(run / "EVALUATION_COMPLETED.json", {"utc": io.utc(), "fits": 45, "truth_replay_max_abs_errors": truth_replay_errors,
                  "test_already_consumed": True, "no_RF": True, "no_model_selection": True})
    check_sources(run)


def selfcheck() -> None:
    records = io.make_records("StageBCorrectnessOnly", [.3], 4, "fixture")
    spatial, temporal = io.physical_arrays(records)
    data = {"records": records, "spatial": spatial, "temporal": temporal, "events": torch.zeros(4, 300, 2), "targets": torch.zeros(4, 300, 2)}
    net = model()
    params = tuple(net.parameters())
    loss = dataset_loss(net, data, "base_R", list(range(4)))
    direct = net.regularized_objective(loss * WEIGHTS["base_R"])
    expected, _ = gradient_vector(direct, params, retain_graph=True)
    data_grad, _ = gradient_vector(loss, params)
    prior_grad, _ = gradient_vector(net.hierarchy_penalty(), params)
    torch.testing.assert_close(data_grad * WEIGHTS["base_R"] + prior_grad, expected, atol=1e-7, rtol=1e-5)
    for field in net.fields.values():
        assert field.contrast is None or torch.allclose(field.deviations().sum(), torch.tensor(0.))
    assert len(H_NODES) == 5 and len(BC_NODES) == 10 and sum(p.numel() for p in params) == 359
    expected_xy = {(0., 0.), (.3, 0.), (-.3, 0.), (0., .3), (0., -.3)}
    for positions in (net.h_xy[list(H_NODES)], net.bc_xy[list(BC_NODES[:5])], net.bc_xy[list(BC_NODES[5:])]):
        actual = {tuple(round(float(x), 2) for x in row) for row in positions}
        assert actual == expected_xy
    left = (torch.tensor([1., 2., 0.]), torch.tensor([True, False, False]))
    right = (torch.tensor([-1., 5., 0.]), torch.tensor([True, True, False]))
    assert cosine(left, right, torch.ones(3, dtype=torch.bool)) == (-1., "DEFINED", 1)
    assert cosine(left, right, torch.tensor([False, False, True])) == (None, "N/A", 0)
    zero = (torch.zeros(3), left[1])
    assert cosine(left, zero, torch.ones(3, dtype=torch.bool))[1] == "UNDEFINED_ZERO_NORM"
    print("STAGE_B_CORRECTNESS_PREFLIGHT_PASS: no optimizer steps, no benchmark targets", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("selfcheck", "freeze", "generate", "train", "worker", "evaluate"))
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--world")
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.action == "selfcheck":
        selfcheck()
    elif args.action == "worker":
        assert args.world and args.condition and args.seed
        train_fit(args.run, args.world, args.condition, args.seed)
    else:
        {"freeze": freeze, "generate": generate, "train": train, "evaluate": evaluate}[args.action](args.run)


if __name__ == "__main__":
    main()
