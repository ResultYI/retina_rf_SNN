from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from baselines.center_surround_ln import CONTEXT_BINS, LNError
from evaluation.mechanistic_retina.schottdorf_ln_source import LNSourcePaths, load_ln_cell
from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import sha256_file
from training.mechanistic_retina.center_surround_ln import (
    BATCH_SIZE, LEARNING_RATE, REGULARIZATIONS, SEED, MAX_STEPS, PATIENCE, MIN_DELTA,
    evaluate_center_surround_ln, fresh_ln, select_and_refit_ln,
)


@dataclass(frozen=True, slots=True)
class LNCellResult:
    artifact_dir: Path
    cell_id: str
    parameter_count: int
    inner_dev_nll: tuple[float, ...]
    selected_lambda: float
    best_step: int
    stop_step: int
    validation_nll: float


def run_ln_cell(paths: LNSourcePaths, cell_id: str, output: Path) -> LNCellResult:
    resolved = output.resolve()
    source_root = paths.retinal_artifact.resolve()
    if resolved == source_root or source_root in resolved.parents or resolved in source_root.parents:
        raise LNError("LN output must be outside frozen source artifacts")
    if output.exists() and any(output.iterdir()):
        raise LNError("LN output directory must be empty")
    loaded = load_ln_cell(paths, cell_id)
    data = loaded.data
    selection = select_and_refit_ln(data.train, loaded.history)
    raw_model = fresh_ln(data.train, loaded.history)
    raw_metrics, raw_logits = evaluate_center_surround_ln(raw_model, data.validation)
    metrics, logits = evaluate_center_surround_ln(selection.refit.model, data.validation)
    for name, digest in loaded.source_hashes.items():
        if sha256_file(Path(name)) != digest:
            raise LNError("frozen source changed during LN smoke")
    output.mkdir(parents=True, exist_ok=True)
    for candidate in selection.candidates:
        torch.save({
            "regularization": candidate.regularization, "model": candidate.model.state_dict(),
            "initial_state": candidate.initial_state,
            "best_step": candidate.best_step, "stop_step": candidate.stop_step,
            "best_dev_nll": candidate.best_dev_nll, "development_curve": candidate.development_curve,
            "history": asdict(loaded.history), "seed": SEED,
        }, output / f"inner-lambda-{candidate.regularization:g}.pt")
    checkpoint_contract = {
        "schema": "schottdorf_center_surround_separable_ln_v1", "cell_id": cell_id,
        "recording_ids": data.recording_ids, "history": asdict(loaded.history),
        "seed": SEED, "context_bins": CONTEXT_BINS, "selected_lambda": selection.selected_lambda,
        "source_hashes": loaded.source_hashes,
        "best_step": selection.selected_best_step, "stop_step": selection.selected_stop_step,
        "refit_steps": selection.refit.stop_step,
    }
    torch.save({**checkpoint_contract, "model": selection.refit.initial_state}, output / "ln-raw.pt")
    torch.save({**checkpoint_contract, "model": selection.refit.model.state_dict()}, output / "ln-trained.pt")
    torch.save({
        "logits_raw": raw_logits, "logits_trained": logits,
        "probabilities_raw": raw_logits.sigmoid(), "probabilities_trained": logits.sigmoid(),
        "target": data.validation.spike_events, "valid_mask": data.validation.valid_mask,
        "source_image_ids": data.validation.source_image_ids,
        "trial_indices": data.validation.trial_indices,
    }, output / "validation-predictions.pt")
    payload = {
        **checkpoint_contract, "adapter_config": asdict(loaded.adapter),
        "input_representation": data.input_representation, "dt_ms": data.dt_ms,
        "recorded_cell_classes": data.recorded_cell_classes,
        "canonical_cell_type": data.cell_types[0], "polarity": data.polarities[0],
        "context_duration_ms": CONTEXT_BINS * data.dt_ms,
        "sampled_lag_ms": [i * data.dt_ms for i in range(CONTEXT_BINS)],
        "history_feature": "Canonical fixed_one_bin_history_state; same fixed tau and zero initial state; free scalar LN coefficient",
        "sequence_boundary": "zero padded/reset independently per original 150-bin sequence; valid bins unchanged",
        "model_formula": "z[t]=b+w_h*h[t]+sum_l(A_c*G_c(x)*k_c[l]-A_s*G_s(x)*k_s[l])*stimulus[t-l,x]; p=sigmoid(z)",
        "spatial_components": "unit-sum isotropic 2D Gaussians, common learnable center_xy in pooled pixels; sigma_c=softplus(u0)+1e-6; sigma_s=sigma_c+softplus(u1)+1e-6; A=softplus(a)+1e-6",
        "temporal_kernel": "two independent rank-1 components; each temporal = raw_temporal / its L2 norm",
        "regularizer": "mean((A_j*G_j)^2) + mean(concat(dx(A_j*G_j),dy(A_j*G_j))^2) + mean(diff2(normalized_temporal_j)^2), means include both components",
        "initialization": "center=(0,0) pixels; sigmas=(1.5,3) pixels; amplitudes=(1,1); seeded independent N(0,1) raw temporal; zero history coefficient; train-only constant logit bias",
        "training_contract": {
            "optimizer": "Adam", "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
            "maximum_inner_steps": MAX_STEPS, "seed": SEED, "early_stopping": True,
            "patience": PATIENCE, "min_delta": MIN_DELTA, "development_evaluation_interval": 1,
            "raw_step_zero_is_eligible": True,
            "best_checkpoint": "exact lowest development NLL; patience reset only after min_delta improvement",
            "batches": "sequence sampling with replacement; RNG reset to seed for every fit",
            "lambda_candidates": REGULARIZATIONS, "selection": "best unpenalized inner-dev Bernoulli NLL",
            "tie_break": "first minimum in ascending lambda order", "full_train_fresh_refit": True,
            "original_validation_used_for_selection": False,
            "full_train_refit_steps": selection.selected_best_step,
        },
        "inner_boundaries": [asdict(value) for value in selection.inner_split.boundaries],
        "guard_definition": "[80%-60,80%) context-only; dev starts at 80%; dev history reset at guard start",
        "train_sequences": data.train.cone_drive.shape[0],
        "validation_sequences": data.validation.cone_drive.shape[0],
        "train_valid_bins": int(data.train.valid_mask.sum()),
        "validation_valid_bins": int(data.validation.valid_mask.sum()),
        "inner_train_valid_bins": int(selection.inner_split.train.valid_mask.sum()),
        "inner_dev_valid_bins": int(selection.inner_split.development.valid_mask.sum()),
        "inner_candidates": [
            {"lambda": candidate.regularization, "inner_dev_nll": score,
             "train_nll_raw": candidate.train_nll_raw, "train_nll_trained": candidate.train_nll_trained,
             "best_step": candidate.best_step, "stop_step": candidate.stop_step,
             "development_curve": candidate.development_curve,
             "parameter_counts": asdict(candidate.parameter_counts)}
            for candidate, score in zip(selection.candidates, selection.inner_dev_nll, strict=True)
        ],
        "parameter_counts": asdict(selection.refit.parameter_counts),
        "effective_continuous_parameter_dimensions": selection.refit.parameter_counts.total - 2,
        "best_inner_dev_nll": min(selection.inner_dev_nll),
        "torch_version": str(torch.__version__), "torch_threads": torch.get_num_threads(),
        "refit_train_nll_raw": selection.refit.train_nll_raw,
        "refit_train_nll_trained": selection.refit.train_nll_trained,
        "validation_nll_raw": raw_metrics.population_nll,
        "validation_nll_trained": metrics.population_nll,
        "gradients_finite": all(fit.gradients_finite for fit in (*selection.candidates, selection.refit)),
        "source_hashes_unchanged": True,
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return LNCellResult(output, cell_id, selection.refit.parameter_counts.total,
                        selection.inner_dev_nll, selection.selected_lambda,
                        selection.selected_best_step, selection.selected_stop_step, metrics.population_nll)
