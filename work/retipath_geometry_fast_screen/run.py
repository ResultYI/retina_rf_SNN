from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "work")]

import numpy as np
import torch

from model import SIGMAS, build
from retipath_final_common import identity
from retipath_phase2_common import evaluate, load_development, load_train, minibatches
from retipath_spatial_ei_pilot import sha, tensor_sha
from training.mechanistic_retina.losses import expected_bernoulli_nll

CELLS = ("69#4", "67#6", "68#4", "67#4")
SEED = 2026091301
BASE = ROOT / "output/experiments/retipath_bounded_geometry_fast_screen"
REGISTRY = ROOT / "output/evaluations/retipath_final_model_evidence_20260913/model_registry.json"
PHASE2 = ROOT / "output/experiments/retipath_spatial_ei_phase2_population/checkpoints/protocol.json"
FIELDS = (
    "row_type", "cell_id", "cell_type", "condition", "seed", "step", "train_nll", "dev_nll",
    "delta_dev_vs_A", "delta_train_vs_A", "delta_dev_C_minus_B", "wins", "losses", "ties",
    "mean_delta_dev", "median_delta_dev", "mean_without_best_cell", "verdict",
    "start_train_nll", "final_train_nll", "best_train_nll", "best_train_step",
    "start_dev_nll", "final_dev_nll", "best_dev_nll", "best_dev_step", "train_at_best_dev",
    "trainable_parameters", "center_x0_deg", "center_y0_deg", "grid_spacing_deg",
    "delta_x_deg", "delta_y_deg", "delta_x_grid", "delta_y_grid", "sigma1_deg", "sigma2_deg",
    "scale1_ratio", "scale2_ratio", "center_bound_fraction", "scale_bound_fraction",
    "near_bound_components", "new_parameter_elements", "new_nonzero_gradient_elements",
    "new_changed_elements", "new_grad_abs_max_json", "max_gradient_norm", "clipped_updates",
    "finite_parameters", "frozen_values_unchanged", "support_topology_unchanged",
    "state_sha256", "parameter_state_json", "checkpoint_sha256", "schedule_sha256",
    "train_identity_json", "dev_identity_json", "elapsed_seconds", "details_json",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor_sha(value).encode())
    return digest.hexdigest()


def append_rows(path, rows):
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False)
                             if isinstance(v, (dict, list)) else v for k, v in row.items()})
        stream.flush()
        os.fsync(stream.fileno())


def frozen_values(model):
    result = {k: v.detach().clone() for k, v in model.named_buffers()}
    result.update({k: v.detach().clone() for k, v in model.named_parameters() if not v.requires_grad})
    return result


def geometric_state(model, cp, condition):
    center = cp["cell_positions_degs"][0].tolist()
    sigma = SIGMAS[cp["cell_types"][0]]
    result = {"center_x0_deg": center[0], "center_y0_deg": center[1],
              "delta_x_deg": 0., "delta_y_deg": 0., "delta_x_grid": 0., "delta_y_grid": 0.,
              "sigma1_deg": sigma[0], "sigma2_deg": sigma[1], "scale1_ratio": 1., "scale2_ratio": 1.,
              "center_bound_fraction": 0., "scale_bound_fraction": 0., "near_bound_components": 0,
              "support_topology_unchanged": True}
    if condition == "A":
        grid = cp["cone_positions_degs"][:, 0]
        result["grid_spacing_deg"] = float((grid.max() - grid.min()) / 16)
        return result
    bank = model.feature_bank
    with torch.no_grad():
        delta = bank.center_delta_deg
        factors = bank.scale_factors
        sigmas = factors * bank.original_sigmas_deg
        position_fraction = bank.raw_center.tanh()
        scale_fraction = bank.raw_scale.tanh()
        assert bool((position_fraction.abs() <= 1).all() and (scale_fraction.abs() <= 1).all())
        assert sigmas[0] < sigmas[1]
        basis = bank.geometry_basis()
        assert basis.shape == (1, 4, 2, 289) and bool(torch.isfinite(basis).all())
        assert torch.equal(basis > 0, bank.path_spatial_basis > 0)
        torch.testing.assert_close(basis.sum(-1), torch.ones_like(basis.sum(-1)), atol=2e-6, rtol=0)
        result.update(delta_x_deg=float(delta[0]), delta_y_deg=float(delta[1]),
                      delta_x_grid=float(position_fraction[0]), delta_y_grid=float(position_fraction[1]),
                      sigma1_deg=float(sigmas[0]), sigma2_deg=float(sigmas[1]),
                      scale1_ratio=float(factors[0]), scale2_ratio=float(factors[1]),
                      grid_spacing_deg=float(bank.grid_spacing_deg[0]),
                      center_bound_fraction=float(position_fraction.abs().max()),
                      scale_bound_fraction=float(scale_fraction.abs().max()),
                      near_bound_components=int((position_fraction.abs() >= .95).sum()
                                                + (scale_fraction.abs() >= .95).sum()))
    return result


def check_cell(cp, train):
    baseline, _ = build(cp, "A")
    x, y = train.cone_drive[:1], train.spike_events[:1]
    with torch.no_grad():
        original = baseline(x, observed_counts=y).logits
    evidence = {}
    for condition in ("B", "C"):
        model, params = build(cp, condition)
        logits = model(x, observed_counts=y).logits
        initial_error = float((logits.detach() - original).abs().max())
        torch.testing.assert_close(logits, original, atol=3e-6, rtol=3e-6)
        for name, value in cp["model"].items():
            assert torch.equal(model.state_dict()[name], value), name
        loss = expected_bernoulli_nll(logits, y, train.valid_mask[:1])
        loss.backward()
        new = {k: p for k, p in model.named_parameters() if k.endswith(("raw_center", "raw_scale"))}
        gradients = {k: p.grad.tolist() for k, p in new.items()}
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool((p.grad != 0).all())
                   for p in new.values())
        with torch.no_grad():
            altered_x = x.clone(); altered_x[:, 65:] += .7
            future = model(altered_x, observed_counts=y).logits
            altered_y = y.clone(); altered_y[:, 64:] = 1 - altered_y[:, 64:]
            history = model(x, observed_counts=altered_y).logits
            future_error = float((future[:, :65] - original[:, :65]).abs().max())
            history_error = float((history[:, :65] - original[:, :65]).abs().max())
            torch.testing.assert_close(future[:, :65], logits[:, :65], atol=0, rtol=0)
            torch.testing.assert_close(history[:, :65], logits[:, :65], atol=0, rtol=0)
            model.feature_bank.raw_center.copy_(torch.tensor([20., -20.]))
            if condition == "C":
                model.feature_bank.raw_scale.copy_(torch.tensor([20., -20.]))
            geometric_state(model, cp, condition)
            assert bool(torch.isfinite(model(x, observed_counts=y).logits).all())
        bank = model.feature_bank.double()
        with torch.no_grad():
            bank.raw_center.copy_(torch.tensor([.13, -.17]))
            if condition == "C":
                bank.raw_scale.copy_(torch.tensor([-.11, .19]))
        probe = torch.linspace(-1, 1, 1 * 4 * 2 * 289, dtype=torch.float64).reshape(1, 4, 2, 289)
        bank.zero_grad()
        (bank.geometry_basis() * probe.square()).sum().backward()
        fd_error = 0.
        for parameter in (bank.raw_center, bank.raw_scale) if condition == "C" else (bank.raw_center,):
            analytic = parameter.grad.clone()
            for i in range(2):
                with torch.no_grad():
                    initial = float(parameter[i]); h = 1e-5
                    parameter[i] = initial + h
                    plus = float((bank.geometry_basis() * probe.square()).sum())
                    parameter[i] = initial - h
                    minus = float((bank.geometry_basis() * probe.square()).sum())
                    parameter[i] = initial
                numeric = (plus - minus) / (2 * h)
                fd_error = max(fd_error, abs(numeric - float(analytic[i])))
                assert abs(numeric - float(analytic[i])) < 1e-8
        evidence[condition] = {"initial_logit_max_abs_error": initial_error,
                               "original_state_exact": True, "new_gradients": gradients,
                               "gradient_finite_difference_max_error": fd_error,
                               "future_stimulus_error": future_error,
                               "current_or_future_spike_error": history_error,
                               "extreme_bound_support_and_finiteness": True,
                               "optimizer_scalar_count": sum(p.numel() for p in params)}
    return evidence


def prepare():
    registry, phase2 = read_json(REGISTRY), read_json(PHASE2)
    assert registry["architecture_id"] == "retipath_spatial_conductance_v1"
    assert tuple(registry["seeds"]) == (2026091301, 2026091302, 2026091303)
    refs, checks = {}, {}
    for cell in CELLS:
        source = registry["cells"][cell]["RetiPath"][str(SEED)]
        ref = phase2["cells"][cell]
        assert sha(Path(source["path"])) == source["sha256"]
        assert sha(Path(ref["input_path"])) == ref["input_sha256"]
        cp = torch.load(source["path"], map_location="cpu", weights_only=True)
        assert cp["cell_id"] == cell and cp["seed"] == SEED
        train = load_train(ref)
        checks[cell] = check_cell(cp, train)
        refs[cell] = {"checkpoint": source, "input_path": ref["input_path"],
                      "input_sha256": ref["input_sha256"], "group": ref["group"],
                      "train_identity": identity(train), "train_sequences": len(train.cone_drive),
                      "schedule_sha256": tensor_sha(minibatches(len(train.cone_drive), SEED, steps=200)),
                      "center_deg": cp["cell_positions_degs"].tolist(),
                      "sigma_deg": SIGMAS[cp["cell_types"][0]],
                      "support_counts": {k: int(cp["model"][f"feature_bank.{k}_support"].sum())
                                         for k in ("bc", "ac", "h1")}}
        print(f"preflight {cell}: PASS", flush=True)
    source_paths = [Path(__file__), Path(__file__).with_name("model.py"), REGISTRY, PHASE2,
                    ROOT / "configs/retipath_final.json",
                    ROOT / "work/retipath_phase2_common.py", ROOT / "work/retipath_spatial_ei_pilot.py",
                    ROOT / "work/retipath_final_common.py", ROOT / "training/mechanistic_retina/losses.py",
                    ROOT / "training/mechanistic_retina/optimizer.py"]
    source_paths += sorted((ROOT / "models/mechanistic_retina").glob("*.py"))
    lock = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "architecture": "official models.mechanistic_retina.retipath.RetiPath; spatial_conductance K=2",
        "conditions": {"A": "formal checkpoint with frozen geometry, matched continuation",
                       "B": "same checkpoint + center = LN center + grid_spacing*tanh(raw_center)",
                       "C": "B + sigma_k = original_sigma_k*(1+0.10*tanh(raw_scale_k))"},
        "support_semantics": "Original BC/AC/H1 masks and graph edges remain exact. Only positive Gaussian weights within existing disks change; functional Gaussian center is not a translated anatomical support disk. Direct/broad share the same two adjusted bases with original separate masked normalizations.",
        "grid_spacing_deg": 0.05390624701976776,
        "sigma_ranges_deg": {"midget": [[.045, .055], [.126, .154]],
                             "parasol": [[.081, .099], [.18, .22]]},
        "range_rationale": "Design choice before optimization: +/-10%, disjoint ordered ranges; no scoring-based range selection.",
        "initialization": "raw center/scale zero; identical original weights per cell; fresh Adam moments for all A/B/C; no re-estimation of conductance RMS",
        "trainable_scalar_counts": {"A": 37, "B": 39, "C": 41},
        "seed": SEED, "updates": 200, "evaluate_every": 20, "batch_size": 4,
        "optimizer": "Adam", "lr": .003, "betas": [.9, .999], "eps": 1e-8,
        "weight_decay": 0, "gradient_clip_norm": 5, "scheduler": None,
        "selection": "Fixed step200 primary; independent best train/dev among 0,20,...,200 descriptive only. No inner selection or fresh refit in this FAST SCREEN, superseding the earlier population plan.",
        "contract": "train [0,16); development [16,20); existing L+M Weber 17x17,150Hz Bernoulli occupancy; independent150 bins;30warmup;120scored; same IDs, observed strictly-past history, masks and minibatches",
        "prohibited": "No RF/Jacobian RF, illusion, pathway experiment, [20,60) or later block, population training, extra seed, restart, extra update, H1-BC capacity bank, default model edit",
        "primary": "B-A development Bernoulli NLL at200; C-A candidate screen; C-B secondary scale increment",
        "verdict_rules": {"numerical_tie_abs_nats_per_bin": 1e-6,
                          "tiny_mean_abs_nats_per_bin": 1e-4,
                          "PROMISING": "mean < -1e-4, >=3/4 wins, and mean excluding best-improving cell < -1e-6; no numerical/representation failure",
                          "STOP": "mean > +1e-4 with <=1/4 wins, or numerical/representation failure",
                          "UNRESOLVED": "otherwise; includes mixed directions, tiny effect, single-cell domination",
                          "near_bound_diagnostic": "abs(tanh(raw))>=0.95; diagnostic only, not automatic degeneration",
                          "degeneration": "nonfinite loss/state/gradient, lost required gradient, support or ordering violation; no improvement threshold on RF"},
        "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                        "device": "CPU", "threads": 1, "deterministic_algorithms": True},
        "cells": refs, "correctness": checks,
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in source_paths},
    }
    out = BASE if not BASE.exists() else BASE / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    with (out / "REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write("# RetiPath bounded geometry FAST SCREEN\n\n")
        stream.write("本轮仅作4-cell、单seed、200步架构筛选。以下合同与正确性结果在训练前冻结。\n\n")
        stream.write("<details><summary>训练前冻结合同、来源与正确性</summary>\n\n```json\n")
        stream.write(json.dumps(lock, ensure_ascii=False, indent=2, allow_nan=False))
        stream.write("\n```\n\n</details>\n\n")
        stream.flush(); os.fsync(stream.fileno())
    with (out / "fast_screen.csv").open("x", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=FIELDS).writeheader()
    print(f"READY {out}", flush=True)


def load_lock(out):
    report = (out / "REPORT.md").read_text(encoding="utf-8")
    return json.loads(report.split("```json\n", 1)[1].split("\n```", 1)[0])


def fit_cell(out, lock, cell):
    ref = lock["cells"][cell]
    cp = torch.load(ref["checkpoint"]["path"], weights_only=True, map_location="cpu")
    train, dev = load_train(ref), load_development(ref)
    train_id, dev_id = identity(train), identity(dev)
    assert train_id == ref["train_identity"]
    batches = minibatches(len(train.cone_drive), SEED, steps=200)
    assert tensor_sha(batches) == ref["schedule_sha256"]
    cell_curves = {}
    for condition in ("A", "B", "C"):
        model, params = build(cp, condition)
        fixed = frozen_values(model)
        initial = {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}
        optimizer = torch.optim.Adam(params, lr=.003, weight_decay=0)
        grad_max = {k: torch.zeros_like(p) for k, p in model.named_parameters() if p.requires_grad}
        curve, best_train, best_dev = [], float("inf"), float("inf")
        max_norm, clipped = 0., 0
        started = time.perf_counter()
        for step in range(201):
            if step:
                model.train()
                ids = batches[step - 1]
                optimizer.zero_grad(set_to_none=True)
                logits = model(train.cone_drive[ids], observed_counts=train.spike_events[ids]).logits
                loss = expected_bernoulli_nll(logits, train.spike_events[ids], train.valid_mask[ids])
                assert bool(torch.isfinite(loss))
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), name
                        grad_max[name] = torch.maximum(grad_max[name], parameter.grad.detach().abs())
                norm = float(torch.nn.utils.clip_grad_norm_(params, 5, error_if_nonfinite=True))
                max_norm = max(max_norm, norm); clipped += norm > 5
                optimizer.step()
                model.project_mechanism_parameters()
                assert all(bool(torch.isfinite(p).all()) for p in params)
            if step % 20:
                continue
            train_nll, _ = evaluate(model, train)
            dev_nll, _ = evaluate(model, dev)
            current = model.state_dict()
            assert all(torch.equal(v, current[k]) for k, v in fixed.items())
            parameters = {k: p.detach().tolist() for k, p in model.named_parameters() if p.requires_grad}
            new = {k: g for k, g in grad_max.items() if k.endswith(("raw_center", "raw_scale"))}
            row = {"row_type": "trajectory", "cell_id": cell, "cell_type": ref["group"],
                   "condition": condition, "seed": SEED, "step": step,
                   "train_nll": train_nll, "dev_nll": dev_nll,
                   "trainable_parameters": sum(p.numel() for p in params),
                   "new_parameter_elements": sum(g.numel() for g in new.values()),
                   "new_nonzero_gradient_elements": sum(int((g > 0).sum()) for g in new.values()),
                   "new_changed_elements": sum(int((dict(model.named_parameters())[k].detach() != initial[k]).sum()) for k in new),
                   "new_grad_abs_max_json": {k: g.tolist() for k, g in new.items()},
                   "max_gradient_norm": max_norm, "clipped_updates": clipped,
                   "finite_parameters": True, "frozen_values_unchanged": True,
                   "state_sha256": state_hash(current), "parameter_state_json": parameters,
                   "checkpoint_sha256": ref["checkpoint"]["sha256"],
                   "schedule_sha256": ref["schedule_sha256"],
                   "train_identity_json": train_id, "dev_identity_json": dev_id,
                   "elapsed_seconds": time.perf_counter() - started,
                   **geometric_state(model, cp, condition)}
            if condition != "A":
                control = cell_curves["A"][step // 20]
                row["delta_dev_vs_A"] = dev_nll - control["dev_nll"]
                row["delta_train_vs_A"] = train_nll - control["train_nll"]
            if condition == "C":
                row["delta_dev_C_minus_B"] = dev_nll - cell_curves["B"][step // 20]["dev_nll"]
            if train_nll < best_train:
                best_train = train_nll; best_train_step = step
            if dev_nll < best_dev:
                best_dev = dev_nll; best_dev_step = step; train_at_best_dev = train_nll
            curve.append(row)
            append_rows(out / "fast_screen.csv", [row])
        assert all(int(optimizer.state[p]["step"]) == 200 for p in params)
        assert all(bool((g > 0).all()) for g in new.values())
        replay, _ = build(cp, condition)
        replay.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(replay(dev.cone_drive[:1], observed_counts=dev.spike_events[:1]).logits,
                                       model(dev.cone_drive[:1], observed_counts=dev.spike_events[:1]).logits,
                                       atol=0, rtol=0)
        summary = dict(curve[-1], row_type="fit_summary", start_train_nll=curve[0]["train_nll"],
                       final_train_nll=curve[-1]["train_nll"], best_train_nll=best_train,
                       best_train_step=best_train_step, start_dev_nll=curve[0]["dev_nll"],
                       final_dev_nll=curve[-1]["dev_nll"], best_dev_nll=best_dev,
                       best_dev_step=best_dev_step, train_at_best_dev=train_at_best_dev,
                       details_json={"optimizer_steps": 200, "state_replay_exact": True,
                                     "all_original_gradient_nonzero_seen": {k: bool((g != 0).any()) for k, g in grad_max.items()},
                                     "best_definition": "independent minima on scheduled evaluation points, no checkpoint selected for further work"})
        append_rows(out / "fast_screen.csv", [summary])
        cell_curves[condition] = curve
        print(f"DONE {cell} {condition}: 200/200; dev {curve[0]['dev_nll']:.6f} -> {curve[-1]['dev_nll']:.6f}", flush=True)


def summarize(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader((out / "fast_screen.csv").open(encoding="utf-8")))
    fits = [r for r in rows if r["row_type"] == "fit_summary"]
    curves = [r for r in rows if r["row_type"] == "trajectory"]
    assert len(fits) == 12 and len(curves) == 132
    assert not any(r["row_type"] == "screen_summary" for r in rows)
    indexed = {(r["cell_id"], r["condition"]): r for r in fits}
    statistics = {}
    for candidate, control in (("B", "A"), ("C", "A"), ("C", "B")):
        delta = np.array([float(indexed[c, candidate]["final_dev_nll"]) - float(indexed[c, control]["final_dev_nll"]) for c in CELLS])
        mean, median = float(delta.mean()), float(np.median(delta))
        wins, losses = int((delta < -1e-6).sum()), int((delta > 1e-6).sum())
        leave_best_out = float(np.delete(delta, delta.argmin()).mean())
        verdict = ("PROMISING" if mean < -1e-4 and wins >= 3 and leave_best_out < -1e-6 else
                   "STOP" if mean > 1e-4 and wins <= 1 else "UNRESOLVED")
        stats = {"row_type": "screen_summary", "condition": candidate + "-" + control,
                 "seed": SEED, "step": 200, "wins": wins, "losses": losses, "ties": 4-wins-losses,
                 "mean_delta_dev": mean, "median_delta_dev": median,
                 "mean_without_best_cell": leave_best_out, "verdict": verdict,
                 "details_json": {c: float(d) for c, d in zip(CELLS, delta)}}
        statistics[candidate + "-" + control] = stats
    append_rows(out / "fast_screen.csv", list(statistics.values()))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
    colors = {"A": "#555555", "B": "#0072B2", "C": "#D55E00"}
    labels = {"A": "Frozen geometry control", "B": "Center residual", "C": "Center + bounded scales"}
    for col, cell in enumerate(CELLS):
        for condition in ("A", "B", "C"):
            curve = sorted([r for r in curves if r["cell_id"] == cell and r["condition"] == condition], key=lambda r: int(r["step"]))
            steps = [int(r["step"]) for r in curve]
            for level, metric in enumerate(("train_nll", "dev_nll")):
                axes[level, col].plot(steps, [float(r[metric]) for r in curve], color=colors[condition], label=labels[condition], lw=1.7)
            if condition != "A":
                axes[2, col].plot(steps, [float(r["delta_dev_vs_A"]) for r in curve], color=colors[condition], marker="o", ms=3)
        axes[0, col].set_title(cell + " | " + indexed[cell, "A"]["cell_type"])
        axes[2, col].axhline(0, color="#888888", lw=.8, ls="--")
        axes[2, col].set_xlabel("Optimizer updates")
        for level in range(3):
            axes[level, col].grid(alpha=.2)
            axes[level, col].set_xticks([0, 40, 80, 120, 160, 200])
    axes[0, 0].set_ylabel("Train NLL (nats / scored bin)")
    axes[1, 0].set_ylabel("Development NLL")
    axes[2, 0].set_ylabel("Development candidate - control")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(.5, .947))
    fig.suptitle("RetiPath bounded geometry FAST SCREEN | 4 cells, 1 seed, fixed 200 updates", y=.985, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(out / "training_curves.png", dpi=180)
    plt.close(fig)
    text = ["1. **200步内是否显示预测收益？**\n",
            "主判读使用固定step200。负差值有利于候选；各条件独立best值仅作描述，已保存在CSV。\n",
            "|比较|mean ΔNLL|median ΔNLL|wins/losses/ties|verdict|",
            "|---|---:|---:|---|---|"]
    for comparison, stat in statistics.items():
        text.append(f"|{comparison}|{stat['mean_delta_dev']:+.7f}|{stat['median_delta_dev']:+.7f}|{stat['wins']}/{stat['losses']}/{stat['ties']}|{stat['verdict']}|")
    text += ["\n2. **几个cell受益？**\n", "|cell / type|A dev NLL|B dev NLL|C dev NLL|B−A|C−A|C−B|", "|---|---:|---:|---:|---:|---:|---:|"]
    for cell in CELLS:
        a, b, c = [float(indexed[cell, k]["final_dev_nll"]) for k in ("A", "B", "C")]
        text.append(f"|{cell} / {indexed[cell, 'A']['cell_type']}|{a:.7f}|{b:.7f}|{c:.7f}|{b-a:+.7f}|{c-a:+.7f}|{c-b:+.7f}|")
    text += ["\n每种ON/OFF×midget/parasol仅1个代表cell，不能把这些差异解释为cell-type效应，也不推断与原LN位置的总体关系。\n",
             "3. **收益是否在训练过程中持续？**\n"]
    for condition in ("B", "C"):
        means = []
        for step in range(20, 201, 20):
            vals = [float(r["delta_dev_vs_A"]) for r in curves if r["condition"] == condition and int(r["step"]) == step]
            means.append(float(np.mean(vals)))
        tail = means[-5:]
        text.append(f"{condition}：10个非零评估点中，平均development差值为负的有{sum(m < -1e-6 for m in means)}/10；后5点为{sum(m < -1e-6 for m in tail)}/5。逐cell及train/dev轨迹见training_curves.png；完整数值每20步保存在CSV。")
    end_best = sum(int(r["best_dev_step"]) == 200 for r in fits)
    text += [f"\n{end_best}/12个fit的最低development NLL在step200。预算上限不是充分收敛证据；本轮未追加步数或重新训练。\n",
             "4. **有没有明显数值或参数退化？**\n",
             "所有12个fit完成200步，损失、参数和梯度有限；原固定参数/buffer逐评估点完全保持，support非零拓扑和两个scale的顺序保持。新增24个标量全部观察到非零梯度。端点state重建后预测逐位一致。\n",
             "|cell|条件|Δx / 格点|Δy / 格点|σ1/原值|σ2/原值|近bound分量|",
             "|---|---|---:|---:|---:|---:|---:|"]
    for cell in CELLS:
        for condition in ("B", "C"):
            r = indexed[cell, condition]
            text.append(f"|{cell}|{condition}|{float(r['delta_x_grid']):+.4f}|{float(r['delta_y_grid']):+.4f}|{float(r['scale1_ratio']):.4f}|{float(r['scale2_ratio']):.4f}|{r['near_bound_components']}|")
    near = sum(int(r["near_bound_components"]) for r in fits)
    changed = sum(int(r["new_changed_elements"]) for r in fits)
    seconds = sum(float(r["elapsed_seconds"]) for r in fits)
    text += [f"\n端点新增参数{changed}/24发生变化，{near}/24接近预设bound（|tanh(raw)|≥0.95）。A/B/C分别37/39/41个可学习标量；总2400次更新，fit计时合计{seconds/60:.2f}分钟。固定格距约0.05390625°，完整角度偏移、sigma、梯度和参数状态见CSV。",
             "\n这里调整的是原support内部的Gaussian功能中心，未移动或扩张硬support；两个midget cell的原direct support仅3/4个格点，因此筛选结论限于这一受限geometry。未运行RF、错视或pathway实验。\n",
             "5. **verdict是什么？**\n"]
    for condition in ("B", "C"):
        stat = statistics[condition + "-A"]
        text.append(f"{condition}相对当前正式RetiPath的同预算control：**{stat['verdict']}**。去掉收益最大的cell后，mean ΔNLL={stat['mean_without_best_cell']:+.7f}。")
    if any(statistics[k]["verdict"] == "PROMISING" for k in ("B-A", "C-A")):
        text.append("\n建议进入下一阶段训练。仅提出建议；未启动22-cell训练。")
    else:
        text.append("\n本轮没有候选满足PROMISING条件，不进入后续训练。")
    text.append("\n此结果是已消费development上的短训练架构筛选，不是population result、独立confirmation或最终性能比较；不能据此量化22-cell OpenRetina gap，也不能决定是否需要自由capacity layer。正式RetiPath和历史H1→BC结果均未修改。本轮结束。")
    with (out / "REPORT.md").open("a", encoding="utf-8") as stream:
        stream.write("\n".join(text) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({k: {f: v for f, v in stat.items() if f in ("mean_delta_dev", "wins", "losses", "verdict")}
                      for k, stat in statistics.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, default=BASE)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.action == "prepare":
        prepare()
        return
    lock = load_lock(args.out)
    for path, digest in lock["source_sha256"].items():
        assert sha(ROOT / path) == digest, path
    existing = list(csv.DictReader((args.out / "fast_screen.csv").open(encoding="utf-8")))
    assert not existing, "No automatic restart is allowed; inspect preserved partial output."
    for ref in lock["cells"].values():
        assert sha(Path(ref["checkpoint"]["path"])) == ref["checkpoint"]["sha256"]
        assert sha(Path(ref["input_path"])) == ref["input_sha256"]
    for cell in CELLS:
        fit_cell(args.out, lock, cell)
    for path, digest in lock["source_sha256"].items():
        assert sha(ROOT / path) == digest, path
    for ref in lock["cells"].values():
        assert sha(Path(ref["checkpoint"]["path"])) == ref["checkpoint"]["sha256"]
        assert sha(Path(ref["input_path"])) == ref["input_sha256"]
    summarize(args.out)


if __name__ == "__main__":
    main()
