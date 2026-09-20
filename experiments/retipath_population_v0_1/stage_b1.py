from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

from . import stage_b as b


io = b.io
ROOT = b.ROOT
PARENT = b.DEFAULT_RUN
DEFAULT_RUN = ROOT / "output/synthetic/retipath_population_stage_b1_20260920"
NEW = {"CW": "C_WEIGHT_MATCHED", "CX": "C_RGC_EXPOSURE_MATCHED"}
ALL = {**b.CONDITIONS, **NEW}
STREAMS = {"CW": ("base_R", "H", "BC"), "CX": ("base_R", "base_R_1", "base_R_2", "H", "BC")}
WEIGHTS = {c: {n: (1. if c == "CW" else 1 / 3) if n.startswith("base_R") else .4
               for n in names} for c, names in STREAMS.items()}
WORLDS = tuple(f"W{i:02}" for i in range(1, 6))
PAIRS = (("CW", "C"), ("CX", "A"), ("CX", "CW"), ("C", "A"))
PORTS = ("h_H", "s_B", "delta_r_B", "d_E", "d_I", "logit", "probability")
METRICS = {k: v for k, v in b.selections().items()
           if k.startswith(("observed_", "unobserved_")) or k in
           {"d_E_rmse", "d_I_rmse", "direct_delta_logit_rmse", "AC_delta_logit_rmse", "probability_rmse"}}


def check(run: Path) -> dict:
    protocol = io.read_json(run / "protocol.json")
    lock = io.read_json(run / "SOURCE_LOCK.json")
    assert io.digest(run / "protocol.json") == lock["protocol_sha256"]
    for name, expected in protocol["source_hashes"].items():
        assert io.digest(ROOT / name) == expected, name
    assert io.digest(PARENT / "protocol.json") == protocol["parent_protocol_sha256"]
    assert io.digest(run / "PARENT_INPUT_LOCK.json") == protocol["parent_input_lock_sha256"]
    return protocol


def parent_inputs(run: Path, *, verify: bool) -> dict:
    files = io.read_json(run / "PARENT_INPUT_LOCK.json")["files"]
    if verify:
        for name, expected in files.items():
            assert io.digest(PARENT / name) == expected, name
    return files


def freeze(run: Path) -> None:
    parent = b.check_sources(PARENT)
    manifest = io.read_json(PARENT / "FILE_MANIFEST.json")
    relative = ["protocol.json", "SOURCE_LOCK.json", "CHECKPOINT_LOCK.json", "GENERATION_COMPLETED.json"]
    for w in WORLDS:
        prefix = f"worlds/{w}/"
        relative += [prefix + name for name in ("SCHEDULES.json", "STIMULUS_MANIFEST.json", "DATA_LOCK.json",
                      "evaluator_only/teacher.pt", "train_data/base_R.pt", "train_data/H.pt", "train_data/BC.pt")]
        relative += [prefix + f"initial_students/{seed}.pt" for seed in b.SEEDS]
        relative += [prefix + f"fits/{c}_{seed}/{name}" for c in b.CONDITIONS for seed in b.SEEDS
                     for name in ("final.pt", "completed.json", "trajectory.csv", "gradient_norms.csv", "gradient_cosines.csv")]
    files = {}
    for name in relative:
        files[name] = io.digest(PARENT / name)
        assert files[name] == manifest["files"][name]["sha256"], name
    run.mkdir(parents=True, exist_ok=False)
    io.write_json(run / "PARENT_INPUT_LOCK.json", {"utc": io.utc(), "parent": str(PARENT), "files": files,
                  "parent_manifest_sha256": io.digest(PARENT / "FILE_MANIFEST.json")})
    sources = dict(parent["source_hashes"])
    sources[Path(__file__).relative_to(ROOT).as_posix()] = io.digest(Path(__file__))
    protocol = {
        "schema": "population_stage_b1_v1", "utc": io.utc(), "designation": "post-hoc targeted follow-up, not confirmatory",
        "question": "Whether C RGC prediction penalty is compatible with insufficient RGC supervision: weighting versus exposure",
        "parent": str(PARENT), "parent_protocol_sha256": io.digest(PARENT / "protocol.json"),
        "parent_input_lock_sha256": io.digest(run / "PARENT_INPUT_LOCK.json"), "source_hashes": sources,
        "worlds": WORLDS, "seeds": b.SEEDS, "new_conditions": NEW, "evaluation_conditions": ALL,
        "new_fits": 30, "evaluation_fits": 75, "steps": 400, "selection": "step400 final only; no early stop, extension, selection, retry or tuning",
        "architecture": parent["architecture"], "frozen": "all priors/hierarchy, LegacyPReLU, Q, routing, conductance, interventions, original teachers/initial states",
        "streams": STREAMS, "weights": WEIGHTS, "hierarchy_coefficient": 1,
        "optimizer": {"name": "Adam", "lr": .01, "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0, "clip_norm": 1., "batch_size": 4},
        "training_data": "byte-identical original Stage B base_R/H/BC and original schedules; no new training data; CX base_R_1/base_R_2 reuse A schedules",
        "observation_sites": {"H_nodes": b.H_NODES, "BC_nodes": b.BC_NODES, "sigma": b.SIGMA, "loss": parent["optimization"]["data_loss"]},
        "exposure": {"CW": {"batches": 1200, "RGC_batches": 400, "sequence_exposures": 4800, "RGC_sequence_exposures": 1600},
                     "CX": {"batches": 2000, "RGC_batches": 1200, "sequence_exposures": 8000, "RGC_sequence_exposures": 4800},
                     "limit": "CX has extra H/BC microbatches and compute; same RGC exposure/source/total coefficient as A; not total-exposure/information/compute matched"},
        "fresh_test": {"namespace": "PopulationStageB1Fresh20260920:Wxx:v1", "scales": b.TEST_SCALES, "per_scale": 8, "sequences_per_world": 16,
                       "rule": "reuse physical stimulus distribution, independently keyed stimulus phases/centers and causal event uniforms; no redraw or screening",
                       "seeds": "io.seeded_key(record.family + ':waveform' / ':events'); explicit numeric seeds in FRESH_TEST_MANIFEST.json",
                       "geometry": parent["stimulus"], "order": "generate all5 teacher truths and lock before any new optimizer update; trainer denies evaluator_only reads"},
        "evaluation": {"primary": "rgc_excess_ce", "primary_comparisons": ["CW-C", "CX-A"],
                       "context_comparisons": ["CX-CW", "C-A"], "comparisons": PAIRS,
                       "secondary": list(METRICS), "ambiguity": "three pairwise seed trajectory RMSEs, separate from truth error; same selections and equal-scale reduction",
                       "delta": parent["evaluation"]["delta"], "reduction": parent["evaluation"]["reduction"],
                       "gate": "all30 new finals and original45 final hashes locked before student fresh-test access; TEST_CONSUMED before evaluator opens fresh truth",
                       "RF": "none", "thresholds": "no equivalence margin, significance tests or new success thresholds; near must be expressed as continuous differences"},
        "interpretation": ["CW improvement is compatible with loss weighting; not exposure-only",
                           "CX retains direct advantage with CE at/below A supports non-necessary prediction sacrifice under matched RGC supervision",
                           "CX stably above A means residual tradeoff, not automatically gradient conflict",
                           "No causal percentage of penalty attribution; no Stage C, E/I observations or prior changes"],
        "runtime": {"threads_per_process": 1, "max_processes": 3, "dtype": "float32", "metric_dtype": "float64", "device": "cpu",
                    "torch": torch.__version__, "python": sys.version, "deterministic_algorithms": True}}
    io.write_json(run / "protocol.json", protocol)
    io.write_json(run / "SOURCE_LOCK.json", {"utc": io.utc(), "protocol_sha256": io.digest(run / "protocol.json"), "sources": sources})
    for rel in sources:
        target = run / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as f:
            f.write((ROOT / rel).read_bytes())
    for w in WORLDS:
        records = io.make_records(f"PopulationStageB1Fresh20260920:{w}:v1", list(b.TEST_SCALES), 8, "fresh_test")
        old = io.read_json(PARENT / f"worlds/{w}/STIMULUS_MANIFEST.json")
        old_records = [r for group in old.values() for r in group]
        assert not {r["family"] for r in records}.intersection(r["family"] for r in old_records)
        assert not {tuple(r["phases"]) for r in records}.intersection(tuple(r["phases"]) for r in old_records)
        io.write_json(run / f"worlds/{w}/FRESH_TEST_MANIFEST.json", {"records": records, "numeric_seeds": [
            {"family": r["family"], "waveform": io.seeded_key(r["family"] + ":waveform"), "events": io.seeded_key(r["family"] + ":events")} for r in records]})
    with (run / "PROTOCOL.md").open("x", encoding="utf-8") as f:
        f.write("# Population Stage B.1 frozen protocol\n\nPost-hoc targeted follow-up; not confirmatory.\n\n"
                "CW = L_R + 0.4 L_H + 0.4 L_BC + P; CX = (L_R+L_R1+L_R2)/3 + 0.4 L_H + 0.4 L_BC + P.\n\n"
                "30 new fits;400 updates;step400 only. Reuse original5 teachers,3 paired initializations,all training data/schedules. "
                "Original45 A/C/E checkpoints read-only. Fresh16 sequences/world at0.225/0.45deg generated and locked before training. "
                "Only RGC weights/stream count differ as requested. CX has5 microbatches versus3; no information/compute-match claim.\n\n"
                "Primary: RGC excess CE; CW-C and CX-A. Context: CX-CW and C-A on the same fresh test. "
                "Secondary and ambiguity retain Stage B definitions. No new equivalence threshold, significance test, RF, E/I observation, prior change or Stage C.\n\n"
                "Complete machine-readable contract:\n\n```json\n" + json.dumps(protocol, ensure_ascii=False, indent=2) + "\n```\n")
    print("PROTOCOL_FROZEN", run, flush=True)


def traces(net, data: dict, ids: list[int]) -> dict:
    x, events = b.stimulus(data, ids), data["events"][ids]
    normal = net(x, observed_events=events)
    out = {name: normal.observation(name).detach() for name in PORTS}
    for name, block in (("direct", b.Intervention.BLOCK_DIRECT_BC_DRIVE), ("AC", b.Intervention.BLOCK_AC_POSTSYNAPTIC_DRIVE)):
        blocked = net(x, observed_events=events, intervention=block)
        out[name + "_delta_logit"] = (blocked.outputs["logit"] - normal.outputs["logit"]).detach()
        retained, tonic = ("d_I", "gE") if name == "direct" else ("d_E", "gI")
        assert torch.equal(blocked.outputs[retained], normal.outputs[retained])
        assert torch.equal(blocked.outputs[tonic], torch.ones_like(blocked.outputs[tonic]))
    return out


def generate(run: Path) -> None:
    check(run)
    assert not (run / "TRAINING_STARTED.json").exists()
    parent_inputs(run, verify=True)
    io.write_json(run / "FRESH_GENERATION_STARTED.json", {"utc": io.utc()})
    files, errors = {}, {}
    for w in WORLDS:
        records = io.read_json(run / f"worlds/{w}/FRESH_TEST_MANIFEST.json")["records"]
        teacher = b.model().requires_grad_(False).eval()
        teacher.load_state_dict(io.load_tensor(PARENT / f"worlds/{w}/evaluator_only/teacher.pt")["model_state"], strict=True)
        spatial, temporal = io.physical_arrays(records)
        inputs = {"records": records, "spatial": spatial, "temporal": temporal}
        collected, largest = defaultdict(list), 0.
        with torch.no_grad():
            for start in range(0, 16, 4):
                ids = list(range(start, start + 4))
                base = teacher(b.stimulus(inputs, ids), observed_events=torch.zeros(4, 300, 2))
                uniforms = torch.stack([torch.rand(300, 2, generator=io.generator(records[i]["family"] + ":events")) for i in ids])
                events, probability = io.causal_sample(base.outputs["logit"], uniforms)
                batch = {"spatial": spatial[ids], "temporal": temporal[ids], "events": events}
                truth = traces(teacher, batch, list(range(4)))
                largest = max(largest, float((probability - truth["probability"]).abs().max()))
                assert torch.equal(events, (uniforms < truth["probability"]).float())
                collected["events"].append(events)
                for name, value in truth.items():
                    collected[name].append(value)
        assert largest < 2e-7
        path = run / f"worlds/{w}/evaluator_only/fresh_test.pt"
        io.save_tensor(path, {**inputs, **{n: torch.cat(v) for n, v in collected.items()}})
        for item in (path, run / f"worlds/{w}/FRESH_TEST_MANIFEST.json"):
            files[item.relative_to(run).as_posix()] = io.digest(item)
        errors[w] = largest
        print("FRESH_TEST_FROZEN", w, flush=True)
    io.write_json(run / "FRESH_TEST_LOCK.json", {"utc": io.utc(), "files": files, "sequences": 80,
                  "before_any_new_training": True, "sampler_max_abs_error": errors, "protocol_sha256": io.digest(run / "protocol.json")})


def worker(run: Path, world: str, condition: str, seed: int) -> None:
    check(run)
    assert world in WORLDS and condition in NEW and seed in b.SEEDS
    assert (run / "FRESH_TEST_LOCK.json").exists() and not (run / "CHECKPOINT_LOCK.json").exists()
    names, weights = STREAMS[condition], WEIGHTS[condition]
    directory = PARENT / "worlds" / world
    locked = parent_inputs(run, verify=False)
    schedule_path = directory / "SCHEDULES.json"
    assert io.digest(schedule_path) == locked[f"worlds/{world}/SCHEDULES.json"]
    schedules = io.read_json(schedule_path)
    files = {n: directory / "train_data" / (("base_R" if n.startswith("base_R") else n) + ".pt") for n in names}
    initial_path = directory / f"initial_students/{seed}.pt"
    allowed = {str(p.resolve()).lower() for p in [*files.values(), initial_path]}
    dest = run / f"worlds/{world}/fits/{condition}_{seed}"
    dest.mkdir(parents=True, exist_ok=False)
    own = str((dest / "final.pt").resolve()).lower()
    accessed = set()

    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        path = str(Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()).lower()
        if "evaluator_only" in path:
            raise PermissionError("Trainer cannot access evaluator truth")
        if path.endswith(".pt") and path != own:
            if path not in allowed:
                raise PermissionError("Unapproved trainer tensor access: " + path)
            accessed.add(path)

    sys.addaudithook(audit)
    datasets = {}
    for name, path in files.items():
        assert io.digest(path) == locked[path.relative_to(PARENT).as_posix()]
        data = io.load_tensor(path)
        assert set(data) == {"records", "spatial", "temporal", "targets"} | ({"events"} if name.startswith("base_R") else set())
        assert data["targets"].shape[1:] == ((300, 5) if name == "H" else (300, 10, 2) if name == "BC" else (300, 2))
        assert len(schedules[name]) == 400 and all(len(x) == 4 for x in schedules[name])
        datasets[name] = data
    assert io.digest(initial_path) == locked[initial_path.relative_to(PARENT).as_posix()]
    net = b.model()
    net.load_state_dict(io.load_tensor(initial_path)["model_state"], strict=True)
    params, masks = tuple(net.parameters()), b.group_masks(net)
    initial_params = {n: p.detach().clone() for n, p in net.named_parameters()}
    optimizer = torch.optim.Adam(params, lr=.01, betas=(.9, .999), eps=1e-8, weight_decay=0)
    trajectories, norms, cosines = [], [], []
    counts, started = Counter(), time.perf_counter()
    for step in range(1, 401):
        optimizer.zero_grad(set_to_none=True)
        vectors, losses = {}, {}
        for name in names:
            loss = b.dataset_loss(net, datasets[name], name, schedules[name][step - 1])
            assert bool(torch.isfinite(loss))
            raw, support = b.gradient_vector(loss, params)
            vectors[name] = (raw * weights[name], support)
            losses[name] = float(loss.detach())
            counts[name] += 1
            for group, mask in masks.items():
                norms.append({"step": step, "dataset": name, "group": group, "raw_norm": float(raw[mask].double().norm()),
                              "weighted_norm": float(vectors[name][0][mask].double().norm()), "supported_coordinates": int((mask & support).sum())})
        for left, right in itertools.combinations(names, 2):
            for group, mask in masks.items():
                value, status, n = b.cosine(vectors[left], vectors[right], mask)
                cosines.append({"step": step, "left": left, "right": right, "group": group, "cosine": value, "status": status, "shared_coordinates": n})
        prior = net.hierarchy_penalty()
        pg, support = b.gradient_vector(prior, params)
        total = pg.clone()
        for gradient, _ in vectors.values():
            total += gradient
        for group, mask in masks.items():
            value = float(pg[mask].double().norm())
            norms.append({"step": step, "dataset": "hierarchy", "group": group, "raw_norm": value, "weighted_norm": value,
                          "supported_coordinates": int((mask & support).sum())})
        offset = 0
        for p in params:
            p.grad = total[offset:offset + p.numel()].reshape_as(p).clone()
            offset += p.numel()
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        assert all(bool(torch.isfinite(p).all()) for p in params)
        trajectories.append({"step": step, **{"loss_" + n: losses[n] for n in names}, "hierarchy": float(prior.detach()),
                             "weighted_data_loss": sum(weights[n] * losses[n] for n in names), "preclip_norm": float(norm),
                             "clip_triggered": bool(norm > 1), "elapsed_seconds": time.perf_counter() - started})
        if step % 50 == 0:
            print("TRAIN", world, condition, seed, step, "elapsed", round(time.perf_counter() - started, 1), flush=True)
    assert all(int(state["step"]) == 400 for state in optimizer.state.values())
    io.save_tensor(dest / "final.pt", {"model_state": net.state_dict(), "optimizer_state": optimizer.state_dict(), "world": world,
                   "condition": condition, "seed": seed, "steps": 400, "initial_sha256": io.digest(initial_path),
                   "protocol_sha256": io.digest(run / "protocol.json")})
    for name, rows in (("trajectory", trajectories), ("gradient_norms", norms), ("gradient_cosines", cosines)):
        io.write_csv(dest / (name + ".csv"), rows)
    io.write_json(dest / "completed.json", {"utc": io.utc(), "world": world, "condition": condition, "seed": seed, "steps": 400,
                  "dataset_batches": dict(counts), "sequence_exposures": sum(counts.values()) * 4, "RGC_sequence_exposures": sum(v for k, v in counts.items() if k.startswith("base_R")) * 4,
                  "checkpoint_sha256": io.digest(dest / "final.pt"), "initial_sha256": io.digest(initial_path), "schedule_sha256": io.digest(schedule_path),
                  "fresh_lock_sha256": io.digest(run / "FRESH_TEST_LOCK.json"), "loaded_tensor_files": sorted(accessed), "evaluator_reads": 0,
                  "raw_max_abs_parameter_changes": {n: float((p.detach() - initial_params[n]).abs().max()) for n, p in net.named_parameters()},
                  "elapsed_seconds": time.perf_counter() - started})
    check(run)


def train(run: Path) -> None:
    protocol = check(run)
    parent_inputs(run, verify=True)
    assert (run / "PREFLIGHT.json").exists()
    fresh = io.read_json(run / "FRESH_TEST_LOCK.json")
    assert fresh["protocol_sha256"] == io.digest(run / "protocol.json")
    assert fresh["before_any_new_training"] is True and fresh["sequences"] == 80
    tasks = [(w, c, seed) for w in WORLDS for seed in b.SEEDS for c in NEW]
    io.write_json(run / "TRAINING_STARTED.json", {"utc": io.utc(), "tasks": tasks, "attempts_per_fit": 1,
                  "fresh_lock_sha256": io.digest(run / "FRESH_TEST_LOCK.json"), "protocol_sha256": io.digest(run / "protocol.json")})
    logs = run / "logs"
    logs.mkdir(exist_ok=False)

    def launch(task):
        w, c, seed = task
        with (logs / f"{w}_{c}_{seed}.log").open("x", encoding="utf-8") as out:
            result = subprocess.run([sys.executable, "-B", "-m", "experiments.retipath_population_v0_1.stage_b1", "worker",
                                     "--run", str(run), "--world", w, "--condition", c, "--seed", str(seed)],
                                    cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, check=False)
        row = {"utc": io.utc(), "world": w, "condition": c, "seed": seed, "returncode": result.returncode}
        io.write_json(logs / f"{w}_{c}_{seed}_process.json", row)
        return row

    results = []
    with ThreadPoolExecutor(max_workers=protocol["runtime"]["max_processes"]) as pool:
        futures = [pool.submit(launch, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
            print("FIT_FINISHED", len(results), "/30", results[-1], flush=True)
    io.write_json(run / "PROCESS_COMPLETIONS.json", results)
    assert len(results) == 30 and all(x["returncode"] == 0 for x in results), "Failure retained; no automatic retry"
    files = {}
    for w in WORLDS:
        for c in ALL:
            for seed in b.SEEDS:
                directory = (PARENT if c in b.CONDITIONS else run) / f"worlds/{w}/fits/{c}_{seed}"
                done = io.read_json(directory / "completed.json")
                assert done["steps"] == 400 and io.digest(directory / "final.pt") == done["checkpoint_sha256"]
                for name in ("final.pt", "completed.json"):
                    files[str((directory / name).resolve())] = io.digest(directory / name)
    parent_inputs(run, verify=True)
    check(run)
    io.write_json(run / "CHECKPOINT_LOCK.json", {"utc": io.utc(), "new_fits": 30, "original_fits": 45, "files": files,
                  "new_optimizer_steps": 12000, "fresh_lock_sha256": io.digest(run / "FRESH_TEST_LOCK.json"), "before_student_test_access": True})
    print("ALL_75_CHECKPOINTS_LOCKED", flush=True)


def errors(pred: dict, truth: dict, i: int, *, ambiguity=False) -> dict:
    result = {}
    for name, (port, nodes) in METRICS.items():
        difference = pred[port][i, b.WARMUP:].double() - truth[port][i, b.WARMUP:].double()
        result[name] = b.rms(difference if nodes is None else difference[..., nodes])
    if not ambiguity:
        p, ell, target = (truth["probability"][i, b.WARMUP:].double(), pred["logit"][i, b.WARMUP:].double(), truth["logit"][i, b.WARMUP:].double())
        result["rgc_excess_ce"] = float((F.softplus(ell) - p * ell - (F.softplus(target) - p * target)).mean())
    return result


def comparisons(fits: list[dict]) -> tuple[list, list, list]:
    lookup = {(x["world"], x["condition"], x["seed"], x["metric"]): x["value"] for x in fits}
    paired, world_rows, overall = [], [], []
    for w, seed, metric in sorted({(x["world"], x["seed"], x["metric"]) for x in fits}):
        for left, right in PAIRS:
            a, z = lookup[w, left, seed, metric], lookup[w, right, seed, metric]
            paired.append({"world": w, "seed": seed, "metric": metric, "comparison": left + "-" + right,
                           "left": a, "reference": z, "difference": a - z})
    for w, metric, comp in sorted({(x["world"], x["metric"], x["comparison"]) for x in paired}):
        rows = [x for x in paired if (x["world"], x["metric"], x["comparison"]) == (w, metric, comp)]
        world_rows.append({"world": w, "metric": metric, "comparison": comp, "left": statistics.mean(x["left"] for x in rows),
                           "reference": statistics.mean(x["reference"] for x in rows), "difference": statistics.mean(x["difference"] for x in rows),
                           "negative_seeds": sum(x["difference"] < 0 for x in rows), "zero_seeds": sum(x["difference"] == 0 for x in rows), "n_seeds": len(rows)})
    for metric, comp in sorted({(x["metric"], x["comparison"]) for x in world_rows}):
        rows = [x for x in world_rows if (x["metric"], x["comparison"]) == (metric, comp)]
        overall.append({"metric": metric, "comparison": comp, "left": statistics.mean(x["left"] for x in rows),
                        "reference": statistics.mean(x["reference"] for x in rows), "difference": statistics.mean(x["difference"] for x in rows),
                        "negative_worlds": sum(x["difference"] < 0 for x in rows), "negative_paired_seeds": sum(x["negative_seeds"] for x in rows),
                        "zero_worlds": sum(x["difference"] == 0 for x in rows), "n_worlds": len(rows), "n_paired_seeds": sum(x["n_seeds"] for x in rows)})
    return paired, world_rows, overall


def evaluate(run: Path) -> None:
    check(run)
    lock = io.read_json(run / "CHECKPOINT_LOCK.json")
    assert lock["new_fits"] == 30 and lock["original_fits"] == 45
    for name, expected in lock["files"].items():
        assert io.digest(Path(name)) == expected
    io.write_json(run / "TEST_CONSUMED.json", {"utc": io.utc(), "checkpoint_lock_sha256": io.digest(run / "CHECKPOINT_LOCK.json"),
                  "scope": "all75 frozen checkpoints on all80 fresh sequences; no checkpoint/condition selection"})
    fresh = io.read_json(run / "FRESH_TEST_LOCK.json")
    seq, ambiguity = [], []
    for w in WORLDS:
        path = run / f"worlds/{w}/evaluator_only/fresh_test.pt"
        assert io.digest(path) == fresh["files"][path.relative_to(run).as_posix()]
        truth = io.load_tensor(path)
        for c in ALL:
            predictions = {}
            for seed in b.SEEDS:
                cp = (PARENT if c in b.CONDITIONS else run) / f"worlds/{w}/fits/{c}_{seed}/final.pt"
                checkpoint = io.load_tensor(cp)
                assert checkpoint["steps"] == 400
                net = b.model().requires_grad_(False).eval()
                net.load_state_dict(checkpoint["model_state"], strict=True)
                collected = defaultdict(list)
                with torch.no_grad():
                    for start in range(0, 16, 4):
                        for n, values in traces(net, truth, list(range(start, start + 4))).items():
                            collected[n].append(values)
                pred = {n: torch.cat(v) for n, v in collected.items()}
                assert all(bool(torch.isfinite(x).all()) for x in pred.values())
                io.save_tensor(run / f"worlds/{w}/evaluation/{c}_{seed}_raw.pt", pred)
                predictions[seed] = pred
                for i, record in enumerate(truth["records"]):
                    for metric, value in errors(pred, truth, i).items():
                        seq.append({"world": w, "condition": c, "seed": seed, "sequence": record["id"], "scale": record["scale"], "metric": metric, "value": value})
            for left, right in itertools.combinations(b.SEEDS, 2):
                for i, record in enumerate(truth["records"]):
                    for metric, value in errors(predictions[left], predictions[right], i, ambiguity=True).items():
                        ambiguity.append({"world": w, "condition": c, "seed": f"{left}-{right}", "sequence": record["id"], "scale": record["scale"], "metric": metric, "value": value})
        print("EVALUATED_FRESH", w, flush=True)
    fit, amb = b.aggregate(seq), b.aggregate(ambiguity)
    pair, world_rows, overall = comparisons(fit)
    ap, aw, ao = comparisons(amb)
    tables = {"per_sequence": seq, "per_fit": fit, "paired_differences": pair, "teacher_summary": world_rows,
              "descriptive_summary": overall, "ambiguity_per_sequence": ambiguity, "ambiguity_per_seed_pair": amb,
              "ambiguity_paired_differences": ap, "ambiguity_teacher_summary": aw, "ambiguity_descriptive_summary": ao}
    for name, rows in tables.items():
        io.write_csv(run / (name + ".csv"), rows)
    io.write_json(run / "results.json", {"protocol_sha256": io.digest(run / "protocol.json"), "primary": "rgc_excess_ce", **tables})
    parent_inputs(run, verify=True)
    check(run)
    io.write_json(run / "EVALUATION_COMPLETED.json", {"utc": io.utc(), "checkpoints": 75, "fresh_sequences": 80,
                  "new_training_fits": 30, "old_checkpoints_unchanged": True, "no_RF": True})


def preflight(run: Path) -> None:
    check(run)
    assert (run / "FRESH_TEST_LOCK.json").exists() and not (run / "TRAINING_STARTED.json").exists()
    assert WEIGHTS["CW"] == {"base_R": 1., "H": .4, "BC": .4}
    assert WEIGHTS["CX"] == {"base_R": 1 / 3, "base_R_1": 1 / 3, "base_R_2": 1 / 3, "H": .4, "BC": .4}
    directory = PARENT / "worlds/W01"
    schedules = io.read_json(directory / "SCHEDULES.json")
    initial = io.load_tensor(directory / "initial_students/4101.pt")
    datasets = {n: io.load_tensor(directory / f"train_data/{n}.pt") for n in ("base_R", "H", "BC")}
    maximum = {}
    for c, names in STREAMS.items():
        net = b.model()
        net.load_state_dict(initial["model_state"], strict=True)
        params = tuple(net.parameters())
        losses = {n: b.dataset_loss(net, datasets["base_R" if n.startswith("base_R") else n], n, schedules[n][0]) for n in names}
        prior = net.hierarchy_penalty()
        objective = prior + sum(WEIGHTS[c][n] * losses[n] for n in names)
        expected, _ = b.gradient_vector(objective, params, retain_graph=True)
        actual, _ = b.gradient_vector(prior, params)
        for n in names:
            g, _ = b.gradient_vector(losses[n], params)
            actual += WEIGHTS[c][n] * g
        maximum[c] = float((actual - expected).abs().max())
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
        assert sum(p.numel() for p in params) == 359
    io.write_json(run / "PREFLIGHT.json", {"utc": io.utc(), "status": "VERIFIED", "gradient_sum_max_abs_errors": maximum,
                  "optimizer_updates": 0, "test_reads": 0, "fixture": "first original W01 train batch and paired initial state", "weights_and_original_streams_checked": True})
    print("B1_PREFLIGHT_PASS", maximum, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "generate", "preflight", "train", "worker", "evaluate"))
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--world", choices=WORLDS)
    parser.add_argument("--condition", choices=NEW)
    parser.add_argument("--seed", type=int, choices=b.SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.action == "worker":
        worker(args.run, args.world, args.condition, args.seed)
    else:
        {"freeze": freeze, "generate": generate, "preflight": preflight, "train": train, "evaluate": evaluate}[args.action](args.run)


if __name__ == "__main__":
    main()
