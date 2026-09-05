from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from collections.abc import Mapping

from evaluation.mechanistic_retina.artifacts import FINAL_TEST_BOUNDARY
from evaluation.mechanistic_retina.rf_base import load_candidate0
from evaluation.mechanistic_retina.spike_banks import (
    generate_nested_spike_bank,
    slice_spike_bank,
    tensor_sha256,
)
from evaluation.model_comparison.baseline_runs import (
    run_bias,
    run_glm,
    run_graph_tcn,
    run_lnln,
)
from evaluation.model_comparison.config import ComparisonConfig, load_comparison_config
from evaluation.model_comparison.mechanistic_run import run_mechanistic
from evaluation.model_comparison.parameters import parameter_inventory
from evaluation.model_comparison.reporting import ArtifactBundle, write_experiment_artifacts
from evaluation.model_comparison.run_data import ProgressCallback, BankRunData
from evaluation.model_comparison.stability import stability_payload
from evaluation.model_comparison.summary import aggregate_models, scientific_decision
from evaluation.model_comparison.types import RunResult
from evaluation.mechanistic_retina.metrics import JsonValue
from evaluation.mechanistic_retina.mechanism_runtime import build_student
from training.mechanistic_retina.optimizer import phase1_parameters
from training.mechanistic_retina.stages import build_seed_data


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    runs: tuple[RunResult, ...]
    rows: tuple[Mapping[str, JsonValue], ...]
    decision: Mapping[str, JsonValue]


def run_experiment(
    root: Path, config_path: Path, progress: ProgressCallback
) -> ExperimentResult:
    config = load_comparison_config(config_path)
    candidate = load_candidate0(
        root / config.candidate0_path,
        usage=config.candidate_teacher_usage,
        reference_candidate_index=config.candidate_teacher_reference_index,
    )
    if candidate.rf_sha256 != config.candidate0_rf_sha256:
        raise RuntimeError("BENCHMARK_IDENTITY_MISMATCH: Candidate0 RF")
    data = build_seed_data(config.data_seed, candidate)
    reference = build_student(data, config.model_seeds[0])
    inventory = parameter_inventory(reference, phase1_parameters(reference))
    match_target = inventory.requires_grad
    runs = []
    bank_manifest = []
    for bank_seed in config.bank_seeds:
        bank = slice_spike_bank(
            generate_nested_spike_bank(
                data.train_probability[:, 0],
                data.validation_probability[:, 0],
                seed=bank_seed,
                max_trials=64,
            ),
            config.trials,
        )
        _verify_bank(config, bank_seed, bank.train_sha256, bank.validation_sha256)
        train_mask = data.train_mask[:, 0, None].expand_as(bank.train_spikes).clone()
        validation_mask = data.validation_mask[:, 0, None].expand_as(
            bank.validation_spikes
        ).clone()
        request = BankRunData(
            root,
            config,
            data,
            candidate,
            bank_seed,
            match_target,
            bank.train_spikes,
            bank.validation_spikes,
            train_mask,
            validation_mask,
            progress,
        )
        bank_manifest.append(
            {
                "seed": bank_seed,
                "trials": config.trials,
                "train_sha256": bank.train_sha256,
                "validation_sha256": bank.validation_sha256,
            }
        )
        runs.extend((run_bias(request), run_glm(request)))
        for model_seed in config.model_seeds:
            runs.extend(
                (
                    run_lnln(request, model_seed),
                    run_graph_tcn(request, model_seed),
                    run_mechanistic(request, model_seed),
                )
            )
    _validate_runs(runs)
    stability = stability_payload(runs)
    rows = aggregate_models(runs, stability)
    decision = scientific_decision(rows)
    parameters = _parameter_payload(inventory, runs)
    identity = _identity_payload(
        root, config_path, config, data, candidate, bank_manifest, runs
    )
    write_experiment_artifacts(
        ArtifactBundle(
            root,
            config_path,
            config,
            tuple(runs),
            rows,
            stability,
            decision,
            parameters,
            identity,
        )
    )
    return ExperimentResult(tuple(runs), tuple(rows), decision)


def _verify_bank(config, seed, train_hash, validation_hash):
    expected = config.bank_hashes[seed]
    if (train_hash, validation_hash) != expected:
        raise RuntimeError(f"BENCHMARK_IDENTITY_MISMATCH: spike bank {seed}")


def _validate_runs(runs):
    expected = {"Bias": 3, "GLM-SH": 3, "LN-LN": 9, "Graph-TCN": 9, "Mechanistic Retina": 9}
    actual = {model: sum(run.model == model for run in runs) for model in expected}
    if actual != expected:
        raise RuntimeError(f"contract violation: run counts {actual}")
    for run in runs:
        values = (
            run.prediction.teacher_expected_ce,
            run.prediction.sampled_nll,
            run.prediction.bits_per_spike,
            run.prediction.logit_rmse,
            run.prediction.brier_score,
        )
        if not run.gradients_finite or not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"NON_FINITE_METRIC: {run.model} bank={run.bank_seed}")


def _parameter_payload(inventory, runs):
    counts = {
        model: next(run.parameter_count for run in runs if run.model == model)
        for model in ("Bias", "GLM-SH", "LN-LN", "Graph-TCN", "Mechanistic Retina")
    }
    target = inventory.requires_grad
    return {
        "P_main_matching_basis": target,
        "P_main_canonical_optimizer_listed": inventory.optimizer_listed,
        "P_main_canonical_nonzero_gradient": inventory.nonzero_gradient,
        "P_main_canonical_actually_updated": inventory.actually_updated,
        "mechanistic": asdict(inventory),
        "models": {
            model: {
                "total": count,
                "trainable": count,
                "match_error_fraction": None
                if model in {"Bias", "GLM-SH", "Mechanistic Retina"}
                else abs(count - target) / target,
            }
            for model, count in counts.items()
        },
        "matching_rule": "deterministic nearest integer configuration before validation",
    }


def _identity_payload(root, config_path, config, data, candidate, banks, runs):
    source_paths = (
        root / "models/mechanistic_retina/model.py",
        root / "training/mechanistic_retina/sampled.py",
        root / "training/mechanistic_retina/optimizer.py",
        root / "baselines/point_process_glm.py",
        root / "baselines/lnln_subunit.py",
        root / "baselines/graph_tcn.py",
        *sorted((root / "evaluation/model_comparison").glob("*.py")),
        root / "scripts/run_model_comparison.py",
        config_path,
    )
    checkpoints = [
        {
            "bank_seed": run.bank_seed,
            "model_seed": run.model_seed,
            "path": run.extras["checkpoint_path"],
            "sha256": run.extras["checkpoint_sha256"],
        }
        for run in runs
        if run.model == "Mechanistic Retina"
    ]
    mechanism_evidence = (
        root / ".omo/evidence/mechanism-heldout-final/heldout-h1-results.json",
        root / ".omo/evidence/mechanism-heldout-final/heldout-ac-results.json",
        root / ".omo/evidence/mechanism-heldout-final/decision-report-zh.md",
    )
    return {
        "lineage": "candidate0-t2-mechanism-identifiable-canonical-v1",
        "bank_decision": "REUSE_EXISTING_BANKS",
        "bank_generation_dependency": "Candidate0 probability + Bernoulli RNG seed only",
        "architecture": "mechanism_identifiable",
        "teacher": "Candidate0",
        "candidate0_rf_sha256": candidate.rf_sha256,
        "candidate_teacher_usage": candidate.teacher_usage.value,
        "candidate_artifact_case": candidate.artifact_case,
        "candidate_selected_index": candidate.selected_candidate,
        "candidate_loaded_index": candidate.loaded_candidate_index,
        "candidate_preflight_passed": candidate.preflight_passed,
        "candidate0_source_sha256": hashlib.sha256((root / config.candidate0_path).read_bytes()).hexdigest(),
        "train_cone_sha256": tensor_sha256(data.train_cones),
        "validation_cone_sha256": tensor_sha256(data.validation_cones),
        "cell_order": [value.cell_id for value in candidate.metadata],
        "cone_order": list(range(data.train_cones.shape[-1])),
        "lag_order": list(range(16)),
        "dt_ms": 5.0,
        "trial_budget": 2,
        "banks": banks,
        "model_seeds": list(config.model_seeds),
        "initialization": "teacher-independent raw initialization",
        "rf_estimand": "conditional total-dynamic logit RF, 16 lags",
        "primary_identity": "exact-cell",
        "secondary_identity": "nearest-cell-derived type×polarity",
        "prototype_metric": "type_prototype_centroid_consistency",
        "final_test_boundary": list(FINAL_TEST_BOUNDARY),
        "mechanistic_checkpoints": checkpoints,
        "reused_mechanism_support_evidence": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in mechanism_evidence
            if path.exists()
        },
        "source_hashes": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
    }


__all__ = ["ExperimentResult", "run_experiment"]
