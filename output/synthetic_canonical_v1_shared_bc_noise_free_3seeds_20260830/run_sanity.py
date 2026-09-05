#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0"]
# ///
# How to run, from the repository root with its frozen runtime:
# D:/anaconda/python.exe -B output/synthetic_canonical_v1_shared_bc_noise_free_3seeds_20260830/run_sanity.py
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Final, TypedDict

import torch

ROOT: Final = Path(__file__).resolve().parents[2]
OUT: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from causal_replay import replay
from evaluation.mechanistic_retina.clean_sampled_artifacts import save_clean_checkpoint
from evaluation.mechanistic_retina.clean_sampled_data import (
    CleanBenchmarkConfig, CleanBenchmarkState, _build_model, _cone_drives, _geometry,
)
from evaluation.mechanistic_retina.clean_sampled_reporting import rf_bundle
from evaluation.mechanistic_retina.clean_sampled_teacher import configure_clean_teacher
from evaluation.mechanistic_retina.noise_free_recovery import _STUDENT_SEEDS, _build_student, _fit
from evaluation.mechanistic_retina.noise_free_recovery_metrics import counterfactuals, expected_metrics, probability
from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina, build_mechanistic_retina

type TensorBundle = dict[str, torch.Tensor]


class Similarity(TypedDict):
    cosine: float
    relative_l2: float


def similarity(reference: torch.Tensor, value: torch.Tensor) -> Similarity:
    """Flatten the existing tensor endpoint; relative L2 uses teacher norm."""
    reference, value = reference.detach().flatten().double(), value.detach().flatten().double()
    norm = reference.norm()
    denominator = norm * value.norm()
    assert norm > 0 and denominator > 0
    return {"cosine": float(reference @ value / denominator),
            "relative_l2": float((value - reference).norm() / norm)}


def selected_rf(model: MechanisticGraphTemporalRetina, cones: torch.Tensor) -> TensorBundle:
    """Preserve existing ordered RF decomposition; name its direct branch explicitly."""
    history = cones.new_zeros(*cones.shape[:2], model.rgc.response_bias.numel())
    bundle = rf_bundle(model, cones, history)
    return {"global": bundle["global"], "H1": bundle["H1"],
            "direct_BC": bundle["BC"], "AC": bundle["AC"]}


def main() -> None:
    assert not list(OUT.glob("*.pt")), "fresh output only; no resume"
    assert not (OUT / "results.json").exists(), "results already exist"
    torch.set_num_threads(1)
    sources = sorted(set(
        ROOT.glob("models/mechanistic_retina/*.py")
    ) | {ROOT / f"evaluation/mechanistic_retina/{name}.py" for name in (
        "clean_sampled_artifacts", "clean_sampled_data", "clean_sampled_teacher",
        "mechanism_teacher_support", "clean_sampled_reporting", "noise_free_recovery",
        "noise_free_recovery_metrics", "rf_effective",
    )} | {ROOT / f"training/mechanistic_retina/{name}.py" for name in ("optimizer", "losses")})
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    config = CleanBenchmarkConfig(student_seed=_STUDENT_SEEDS[0])
    geometry = _geometry()
    model_config = MechanisticRetinaConfig(architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE, cell_specific_gains=True)
    teacher = _build_model(model_config, *geometry, config.teacher_seed)
    configure_clean_teacher(teacher)
    student = _build_model(model_config, *geometry, config.student_seed)
    train_cones, validation_cones = _cone_drives(config, geometry[0].shape[0])
    state = CleanBenchmarkState(config, teacher, student, *geometry, train_cones, validation_cones,
                                torch.empty(0), torch.empty(0), {})
    train_target, validation_target = probability(teacher, train_cones), probability(teacher, validation_cones)
    teacher_before = {name: value.clone() for name, value in teacher.state_dict().items()}
    save_clean_checkpoint(OUT / "teacher.pt", teacher, state, "fresh_teacher")
    torch.save({"train_cones": train_cones, "validation_cones": validation_cones,
                "train_teacher_probability": train_target, "validation_teacher_probability": validation_target},
               OUT / "stimulus-and-probabilities.pt")
    teacher_rf = selected_rf(teacher, validation_cones[:2])
    teacher_cf, _ = counterfactuals(teacher, validation_cones, teacher_rf["global"])
    rf_tensors = {"teacher": teacher_rf}
    cf_tensors = {"teacher": teacher_cf}
    solutions = {}
    for seed in _STUDENT_SEEDS:
        model = _build_student(state, seed)
        seed_state = replace(state, config=replace(config, student_seed=seed), student=model)
        save_clean_checkpoint(OUT / f"student-seed-{seed}-raw.pt", model, seed_state, f"student-seed-{seed}-raw")
        raw_kl = expected_metrics(model, validation_cones, validation_target)["kl"]
        raw_rf = selected_rf(model, validation_cones[:2])
        print(f"seed={seed} raw_validation_kl={raw_kl:.12g}; starting frozen 400-step fit", flush=True)
        _fit(model, state, train_target)
        trained_kl = expected_metrics(model, validation_cones, validation_target)["kl"]
        trained_rf = selected_rf(model, validation_cones[:2])
        trained_cf, _ = counterfactuals(model, validation_cones, trained_rf["global"])
        solutions[str(seed)] = {
            "validation_kl": {"raw": raw_kl, "trained": trained_kl},
            "rf": {name: {"raw": similarity(teacher_rf[name], raw_rf[name]),
                           "trained": similarity(teacher_rf[name], trained_rf[name])} for name in teacher_rf},
            "counterfactual": {name: {endpoint: similarity(teacher_cf[name][endpoint], trained_cf[name][endpoint])
                                        for endpoint in teacher_cf[name]} for name in teacher_cf},
        }
        save_clean_checkpoint(OUT / f"student-seed-{seed}-trained.pt", model, seed_state, f"student-seed-{seed}-trained")
        rf_tensors[f"{seed}_raw"], rf_tensors[f"{seed}_trained"] = raw_rf, trained_rf
        cf_tensors[str(seed)] = trained_cf
        (OUT / f"seed-{seed}-results.json").write_text(json.dumps(solutions[str(seed)], indent=2), encoding="utf-8")
        print(f"seed={seed} trained_validation_kl={trained_kl:.12g}; saved", flush=True)
    assert all(torch.equal(teacher_before[n], v) for n, v in teacher.state_dict().items())
    torch.save(rf_tensors, OUT / "rf-tensors.pt")
    torch.save(cf_tensors, OUT / "counterfactual-tensors.pt")
    checks, replay_tensors = {}, {}
    roles = {"teacher": "teacher.pt"} | {str(seed): f"student-seed-{seed}-trained.pt" for seed in _STUDENT_SEEDS}
    for role, filename in roles.items():
        checkpoint = torch.load(OUT / filename, map_location="cpu", weights_only=True)
        restored = build_mechanistic_retina(MechanisticRetinaConfig(**checkpoint["model_config"]),
            checkpoint["cone_positions"], checkpoint["cell_positions"], checkpoint["cell_types"], checkpoint["polarities"])
        restored.load_state_dict(checkpoint["model_state"], strict=True)
        checks[role], replay_tensors[role] = replay(restored, validation_cones)
        replay_kl = expected_metrics(restored, validation_cones, validation_target)["kl"]
        checks[role]["checkpoint_replay_kl"] = replay_kl
        if role != "teacher":
            checks[role]["checkpoint_validation_kl_equal"] = replay_kl == solutions[role]["validation_kl"]["trained"]
        print(f"causal_replay={role} all_passed={checks[role]['all_passed']}", flush=True)
    torch.save(replay_tensors, OUT / "causal-replay-tensors.pt")
    payload = {
        "model_name": "Canonical V1", "causal_contract": teacher.config.causal_contract,
        "protocol": asdict(config), "student_seeds": _STUDENT_SEEDS,
        "fresh_teacher": True, "fresh_students": True, "fresh_optimizer_per_student": True,
        "old_checkpoints_loaded_or_converted": False, "teacher_trained": False,
        "sampled_spike_banks_generated": False, "parameter_recovery_audit_run": False,
        "training_target": "teacher_probability_expected_Bernoulli_CE",
        "history": "zero; identical teacher/student conditional history",
        "rf_contract": {"contexts": "first two validation stimuli", "lag_bins": 16,
            "lag_order": "oldest to current", "global": "forward-autograd logit Jacobian",
            "H1": "global RF minus H1-off RF",
            "direct_BC": "RF with H1-off and AC-off (existing ordered rf_bundle BC endpoint)",
            "AC": "H1-off RF minus direct_BC RF (existing ordered rf_bundle endpoint)"},
        "counterfactual_contract": "clamped minus normal: logits/probability over all validation stimuli; RF over first two",
        "relative_l2": "norm(student-teacher)/norm(teacher), flattened tensor, float64 metric accumulation",
        "validation_kl": "unchanged float32 expected CE minus teacher entropy helper",
        "solutions": solutions, "causal_contract_checks": checks,
        "source_hashes_before": hashes,
        "source_hashes_unchanged": all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items()),
        "runtime": {"torch": torch.__version__, "threads": torch.get_num_threads(), "dtype": str(train_cones.dtype)},
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    assert payload["source_hashes_unchanged"]
    assert all(values["all_passed"] for values in checks.values())
    print(f"completed={OUT}", flush=True)


if __name__ == "__main__":
    main()
