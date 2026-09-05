from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
READOUT = ROOT / ".omo/evidence/rf-math-contract-and-readout-audit"
TRIAL = ROOT / ".omo/evidence/trial-scaling-rf-identifiability"
PRIOR = ROOT / ".omo/evidence/rf-identifiability-reachability-audit"
for import_root in (READOUT, TRIAL, PRIOR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from intervention_eval import fit_bias_delta
from path_metrics import teacher_recovery
from trial_contract import histories_for_rows, mean_field_contract, values_for_rows
from trial_fitting import data_volume
from trial_oracle import fit_batched_oracle, predict, rf_metrics
from trial_nulls import null_distributions, sample_complexity_gate


def test_history_trace_excludes_current_bin_event() -> None:
    spikes = torch.tensor([[[[[1.0], [0.0], [0.0]]]]])
    rows = ((0, 0), (0, 1), (0, 2))
    history = histories_for_rows(spikes, rows)
    decay = float(torch.exp(torch.tensor(-0.1)))
    assert history.shape == (1, 3, 1, 1)
    assert torch.allclose(
        history.flatten(),
        torch.tensor([0.0, 1.0, decay]),
    )


def test_values_for_rows_preserves_bank_and_trial_axes() -> None:
    values = torch.arange(2 * 3 * 4 * 5 * 1).reshape(2, 3, 4, 5, 1)
    rows = ((0, 1), (2, 4))
    selected = values_for_rows(values, rows)
    assert selected.shape == (2, 2, 4, 1)
    assert torch.equal(selected[:, 0], values[:, 0, :, 1])
    assert torch.equal(selected[:, 1], values[:, 2, :, 4])


def test_mean_field_history_is_causal_and_deterministic() -> None:
    base = torch.zeros(1, 3, 1)
    history, probability = mean_field_contract(base)
    assert torch.equal(history[:, 0], torch.zeros_like(history[:, 0]))
    assert torch.allclose(probability[:, 0], torch.full_like(probability[:, 0], 0.5))
    repeated_history, repeated_probability = mean_field_contract(base)
    assert torch.equal(history, repeated_history)
    assert torch.equal(probability, repeated_probability)


def test_batched_oracle_supports_shared_and_banked_histories() -> None:
    generator = torch.Generator().manual_seed(17)
    design = torch.randn(10, 3, generator=generator)
    history = torch.randn(2, 10, 2, 2, generator=generator)
    targets = torch.bernoulli(torch.full_like(history, 0.2), generator=generator)
    fit = fit_batched_oracle(
        design,
        history,
        targets,
        ridge=1.0,
        max_iterations=3,
    )
    banked = predict(design, history, fit)
    shared = predict(design, history[0], fit)
    assert banked.shape == targets.shape
    assert shared.shape == targets.shape
    assert torch.isfinite(fit.objective_by_bank).all()


def test_bias_bisection_remains_finite_for_separated_targets() -> None:
    logits = torch.tensor([[[20.0, -20.0]], [[10.0, -10.0]]])
    targets = torch.tensor([[[0.0, 1.0]], [[0.0, 1.0]]])
    result = fit_bias_delta(logits, targets, torch.ones_like(targets, dtype=torch.bool))
    assert result.converged
    assert torch.isfinite(result.delta).all()
    assert float(result.delta.abs().max()) <= 80.0


def test_teacher_recovery_uses_pair_condition_cell_contract() -> None:
    teacher = torch.randn(2, 3, 4, 5)
    predicted = teacher[None].expand(6, -1, -1, -1, -1).clone()
    metrics = teacher_recovery(predicted, teacher)
    assert abs(float(metrics["mean_full_cosine"]) - 1.0) < 1e-12
    assert abs(float(metrics["mean_dynamic_cosine"]) - 1.0) < 1e-12


def test_null_gate_uses_bank_as_replication_unit() -> None:
    predicted = torch.randn(4, 2, 3, 4, 5)
    teacher = torch.randn(2, 3, 4, 5)
    metrics = rf_metrics(predicted, teacher)
    nulls = null_distributions(predicted, teacher, {}, draws=8, seed=9)
    gate = sample_complexity_gate(
        metrics["mean_full_cosine"],
        nulls["static"],
        ceiling=1.0,
        practical_threshold=0.8,
        seed=10,
    )
    assert len(gate["observed_by_bank"]) == 4
    assert len(gate["null_q99_by_bank"]) == 4
    assert 0.0 <= float(gate["bank_pass_fraction"]) <= 1.0


def test_data_volume_counts_stimulus_trial_sequences_as_independent_clusters() -> None:
    targets = torch.zeros(2, 6, 4, 1)
    rows = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2))

    volume = data_volume(targets, trial_count=4, rows=rows)

    assert volume["independent_stimuli_per_bank"] == 2
    assert volume["independent_sequence_clusters_per_bank"] == 8
    assert volume["postburn_bins_per_bank"] == 24
