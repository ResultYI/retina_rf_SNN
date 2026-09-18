from __future__ import annotations

import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from adapter import OpenRetinaAdapted
from data_adapter import DEST, MovieBank, digest, exclusive_json, read_json

SEEDS = (2026091301, 2026091302, 2026091303)
MAX_UPDATES = 345
EVAL_EVERY = 23
PATIENCE_CHECKS = 10
MIN_DELTA = 0.001
BATCH = 4
LR = 0.005


def optimizer_and_scheduler(model: OpenRetinaAdapted) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.OneCycleLR]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=0.01, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=MAX_UPDATES,
        pct_start=0.3, anneal_strategy="cos", cycle_momentum=True, base_momentum=0.85,
        max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, three_phase=False)
    return optimizer, scheduler


def loss_for_sample(model: OpenRetinaAdapted, bank: MovieBank, windows: torch.Tensor,
                    rows: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    features = model.core_features(bank.inputs[windows])
    data_terms = []
    bins = 0
    for cell, selected in rows.items():
        obs = bank.cells[cell]
        drive = model.stimulus_drive(features, cell)
        logits = model.logits_from_drive(drive, obs.history[selected], cell)
        mask = obs.mask[selected]
        data_terms.append(F.binary_cross_entropy_with_logits(logits[mask], obs.targets[selected][mask], reduction="sum"))
        bins += int(mask.sum())
    nll_sum = torch.stack(data_terms).sum()
    core_reg, readout_reg = model.regularization(tuple(rows))
    loss = nll_sum + core_reg + readout_reg
    return loss, {"batch_nll_sum": float(nll_sum.detach()), "batch_scored_bins": bins,
                  "batch_nll_mean": float(nll_sum.detach()) / bins,
                  "core_regularizer": float(core_reg.detach()), "readout_regularizer": float(readout_reg.detach()),
                  "total_objective_sum": float(loss.detach())}


def evaluate(model: OpenRetinaAdapted, bank: MovieBank) -> tuple[float, dict[str, float], dict[str, torch.Tensor]]:
    model.eval()
    all_drive: dict[str, list[torch.Tensor]] = {cell: [] for cell in bank.cells}
    with torch.no_grad():
        for first in range(0, len(bank.inputs), BATCH):
            features = model.core_features(bank.inputs[first:first + BATCH])
            for cell in bank.cells:
                all_drive[cell].append(model.stimulus_drive(features, cell))
        losses, logits_saved = {}, {}
        for cell, obs in bank.cells.items():
            drive = torch.cat(all_drive[cell])[obs.input_indices]
            logits = model.logits_from_drive(drive, obs.history, cell)
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError(f"Nonfinite frozen output in {bank.phase}: {cell}")
            losses[cell] = float(F.binary_cross_entropy_with_logits(
                logits.double()[obs.mask], obs.targets.double()[obs.mask], reduction="mean"))
            logits_saved[cell] = logits
    return sum(losses.values()) / len(losses), losses, logits_saved


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".pending")
    torch.save(payload, temporary)
    temporary.replace(path)


def fit(seed: int, phase: str, bank: MovieBank, validation: MovieBank | None,
        selected_steps: int | None = None) -> dict:
    folder = DEST / "checkpoints" / str(seed)
    folder.mkdir(exist_ok=True)
    finished = folder / f"{phase}_complete.json"
    if finished.exists():
        return read_json(finished)
    model = OpenRetinaAdapted(tuple(bank.cells), bank.means(), seed).place()
    optimizer, scheduler = optimizer_and_scheduler(model)
    generator = torch.Generator().manual_seed(seed + 1_000_003)
    target_step = MAX_UPDATES if phase == "selection" else selected_steps
    if target_step is None:
        raise ValueError("A fresh refit needs the already selected number of updates")
    protocol_sha = digest(DEST / "PROTOCOL.md")
    records = []
    best_nll, best_step, plateau_nll, last_improved = math.inf, 0, math.inf, 0
    start_step, previous_seconds = 0, 0.0
    latest = folder / f"{phase}_latest.pt"
    if latest.exists():
        saved = torch.load(latest, weights_only=True, map_location="cpu")
        if saved["protocol_sha256"] != protocol_sha or saved["target_steps"] != target_step:
            raise ValueError("A resumed run differs from the frozen protocol")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        generator.set_state(saved["generator_state"])
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        records, start_step, previous_seconds = saved["records"], saved["step"], saved["elapsed_seconds"]
        best_nll, best_step = saved["best_nll"], saved["best_step"]
        plateau_nll, last_improved = saved["plateau_nll"], saved["last_improved"]
    started = time.perf_counter()
    initial_state = model.cpu_state() if start_step == 0 else None
    log_path = folder / f"{phase}.log"

    def payload(step: int) -> dict:
        return {"model": model.cpu_state(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "seed": seed, "phase": phase, "step": step, "target_steps": target_step,
            "cells": list(bank.cells), "training_occupancy_initialization": bank.means(),
            "protocol_sha256": protocol_sha, "best_nll": best_nll, "best_step": best_step,
            "plateau_nll": plateau_nll, "last_improved": last_improved, "records": records,
            "generator_state": generator.get_state(), "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "elapsed_seconds": previous_seconds + time.perf_counter() - started}

    def check(step: int, batch_stats: dict[str, float], gradient_norm: float) -> None:
        nonlocal best_nll, best_step, plateau_nll, last_improved
        train_nll, _, _ = evaluate(model, bank)
        val_nll = evaluate(model, validation)[0] if validation is not None else None
        improved = val_nll is not None and val_nll < best_nll
        if improved:
            best_nll, best_step = val_nll, step
        if val_nll is not None and val_nll < plateau_nll - MIN_DELTA:
            plateau_nll, last_improved = val_nll, step
        records.append({"seed": seed, "phase": phase, "step": step, "train_nll": train_nll,
            "inner_validation_nll": val_nll, "best_step": best_step if validation is not None else selected_steps,
            "lr_next_update": optimizer.param_groups[0]["lr"], "gradient_norm_before_clip": gradient_norm,
            "elapsed_seconds": previous_seconds + time.perf_counter() - started, **batch_stats})
        if improved:
            save_checkpoint(folder / "selected.pt", payload(step))
        save_checkpoint(latest, payload(step))
        with log_path.open("a", encoding="utf-8") as stream:
            import json
            stream.write(json.dumps(records[-1], allow_nan=False) + "\n")

    if start_step == 0:
        check(0, {}, 0.0)
    stopped = "budget_limit" if phase == "selection" else "fixed_selected_length"
    completed_step = start_step
    for step in range(start_step + 1, target_step + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        windows, rows = bank.sample(BATCH, generator)
        loss, batch_stats = loss_for_sample(model, bank, windows, rows)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Nonfinite training loss at {seed}/{phase}/{step}")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
        optimizer.step()
        scheduler.step()
        model.project_readout()
        completed_step = step
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"update={step} nll_sum={batch_stats['batch_nll_sum']:.9g} reg_core={batch_stats['core_regularizer']:.9g} reg_readout={batch_stats['readout_regularizer']:.9g} grad={gradient_norm:.9g}\n")
        if step % EVAL_EVERY == 0 or step == target_step:
            check(step, batch_stats, gradient_norm)
            if validation is not None and step - last_improved >= PATIENCE_CHECKS * EVAL_EVERY:
                stopped = "inner_validation_patience"
                break
    if phase == "refit":
        best_nll, best_step = 0.0, selected_steps
        save_checkpoint(folder / "refit.pt", payload(completed_step))
    elapsed = previous_seconds + time.perf_counter() - started
    result = {"seed": seed, "phase": phase, "completed_steps": completed_step,
        "selected_steps": best_step if phase == "selection" else selected_steps,
        "best_inner_validation_nll": best_nll if phase == "selection" else None,
        "stop_reason": stopped, "elapsed_seconds": elapsed, "completed_utc": utc(),
        "protocol_sha256": protocol_sha, "parameters": sum(p.numel() for p in model.parameters()),
        "all_finite": all(bool(torch.isfinite(p).all()) for p in model.parameters()),
        "trainable_parameters_changed": None if initial_state is None else sum(
            not torch.equal(p.detach().cpu(), initial_state[name]) for name, p in model.named_parameters()),
        "checkpoint_sha256": digest(folder / ("selected.pt" if phase == "selection" else "refit.pt")),
        "curve_records": records}
    exclusive_json(finished, result)
    print(f"{phase.upper()} seed={seed} updates={completed_step} selected={result['selected_steps']} stop={stopped} minutes={elapsed / 60:.1f}", flush=True)
    return result


def write_training_summary(results: list[dict]) -> None:
    rows = []
    for result in results:
        for row in result["curve_records"]:
            rows.append({**row, "completed_steps": result["completed_steps"],
                "selected_steps": result["selected_steps"], "stop_reason": result["stop_reason"],
                "fit_elapsed_seconds": result["elapsed_seconds"], "all_finite": result["all_finite"]})
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (DEST / "training_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
