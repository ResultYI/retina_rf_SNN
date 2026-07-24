from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from baselines.point_process_glm import fit_point_process_glm
from data.rgc_response import CellMetadata, ResponseTargetKind
from evaluation.response_metrics import compute_response_metrics
from evaluation import rf_dynamic
from training.response_data import PreparedResponseData, ResponseSplit


def test_response_metrics_preserve_stimulus_and_trial_axes() -> None:
    targets = torch.tensor(
        [
            [[[0.0], [0.0], [1.0], [1.0]]] * 2,
            [[[1.0], [1.0], [0.0], [0.0]]] * 2,
        ]
    )
    probabilities = 1 - targets.clamp(0.1, 0.9)
    logits = torch.logit(probabilities)

    metrics = compute_response_metrics(
        logits,
        targets,
        torch.ones_like(targets, dtype=torch.bool),
        ResponseTargetKind.BERNOULLI,
        torch.tensor([0.5]),
    )

    assert metrics.psth_correlation < -0.99
    assert np.isfinite(metrics.micro_bits_per_spike)
    assert np.isfinite(metrics.macro_bits_per_spike)


def test_glm_reports_validation_and_test_metrics() -> None:
    data = _prepared_response_data()

    result = fit_point_process_glm(data, device=torch.device("cpu"), steps=2)

    assert result.model.kernel.shape[1] == 16
    assert np.isfinite(result.validation_metrics.nll)
    assert np.isfinite(result.test_metrics.nll)
    assert result.best_step >= 1


@dataclass(frozen=True, slots=True)
class _FakeOutput:
    spike_logits: torch.Tensor


class _FakeRGC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("support_mask", torch.ones(1, 1, dtype=torch.bool))


class _StatefulModel(nn.Module):
    def __init__(self, decay: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.rgc = _FakeRGC()
        self.decay = decay

    def forward_sequence(
        self,
        sequence: torch.Tensor,
        *,
        observed_counts: torch.Tensor | None = None,
    ) -> tuple[_FakeOutput, torch.Tensor]:
        state = torch.zeros(sequence.shape[0], 1, device=sequence.device)
        logits = []
        for sample in sequence.unbind(dim=1):
            drive = sample.mean(dim=1, keepdim=True)
            state = self.decay * state + drive
            logits.append(self.scale * state * drive)
        return _FakeOutput(torch.stack(logits, dim=1)), state


def test_dynamic_rf_measures_reset_recovery_and_numerical_quality() -> None:
    split = _context_split()

    result = rf_dynamic.evaluate_dynamic_rf(
        _StatefulModel(0.8),
        split,
        lag_steps=2,
        recovery_delays_ms=(0, 10, 20),
        dt_ms=5.0,
        bootstrap_iterations=100,
        seed=7,
    )

    assert result.pair_count == 3
    assert result.reset_shape_distance < result.mean_shape_distance
    assert result.recovery_shape_distances[-1] <= result.recovery_shape_distances[0]
    assert result.finite_difference_relative_error <= 0.05
    assert len(result.shape_distance_ci) == 2

    initialized = rf_dynamic.evaluate_dynamic_rf(
        _StatefulModel(0.0),
        split,
        lag_steps=2,
        recovery_delays_ms=(0, 10, 20),
        dt_ms=5.0,
        bootstrap_iterations=100,
        seed=7,
    )
    comparison = rf_dynamic.compare_dynamic_rf(
        result,
        initialized,
        bootstrap_iterations=100,
        seed=7,
    )

    assert comparison.learned_shape_delta > 0
    assert comparison.status == "supported"


def test_dynamic_rf_teacher_comparison_requires_error_reduction() -> None:
    trained = _teacher_result("supported", (0.1,) * 3, (0.05,) * 3)
    initialized = _teacher_result("supported", (0.4,) * 3, (0.2,) * 3)

    comparison = rf_dynamic.compare_dynamic_rf(
        trained,
        initialized,
        bootstrap_iterations=100,
        seed=7,
    )

    assert comparison.status == "supported"
    assert comparison.teacher_primary_error_delta_ci[0] > 0
    assert comparison.teacher_recovery_error_delta_ci[0] > 0


def test_dynamic_rf_teacher_comparison_hardens_teacher_gate_samples() -> None:
    initialized = _teacher_result(
        "supported",
        (0.4,) * 3,
        (0.2,) * 3,
    )
    cases = (
        (
            _teacher_result(
                "supported",
                (0.1,) * 2,
                (0.05,) * 2,
                pair_count=2,
            ),
            _teacher_result(
                "supported",
                (0.4,) * 2,
                (0.2,) * 2,
                pair_count=2,
            ),
            "not_identifiable",
        ),
        (
            _teacher_result(
                "supported",
                (0.1,) * 3,
                (0.05,) * 3,
                direction_agreement=(True, False),
                model_signed_gains=(0.2, -0.1),
                reference_signed_gains=(0.3, 0.1),
            ),
            initialized,
            "teacher_mismatch",
        ),
        (
            _teacher_result(
                "supported",
                (0.1, float("nan"), 0.1),
                (0.05,) * 3,
            ),
            initialized,
            "not_supported",
        ),
        (
            _teacher_result("supported", (0.1,) * 3, (0.05,) * 2),
            initialized,
            "not_supported",
        ),
        (
            _teacher_result("not_supported", (0.1,) * 3, (0.05,) * 3),
            _teacher_result("not_supported", (0.4,) * 3, (0.2,) * 3),
            "supported",
        ),
    )

    for trained, initial, expected in cases:
        comparison = rf_dynamic.compare_dynamic_rf(
            trained,
            initial,
            bootstrap_iterations=100,
            seed=7,
        )
        assert comparison.status == expected


def _prepared_response_data() -> PreparedResponseData:
    cells = CellMetadata(
        ids=("cell",),
        type_ids=("midget",),
        polarities=np.asarray([0], dtype=np.int64),
        positions_degs=np.zeros((1, 2), dtype=np.float32),
        eccentricities_deg=np.asarray([4.0], dtype=np.float32),
    )
    train = _response_split(0.0, "train")
    validation = _response_split(1.0, "validation")
    test = _response_split(0.0, "test")
    return PreparedResponseData(
        train=train,
        validation=validation,
        test=test,
        cells=cells,
        cone_positions_degs=np.zeros((1, 2), dtype=np.float32),
        time_axis_seconds=np.arange(20, dtype=np.float64) * 0.005,
        target_kind=ResponseTargetKind.BERNOULLI,
        normalization_mean=np.zeros(1, dtype=np.float32),
        normalization_std=np.ones(1, dtype=np.float32),
        fingerprint="test",
    )


def _response_split(value: float, source: str) -> ResponseSplit:
    counts = torch.full((1, 2, 20, 1), value)
    return ResponseSplit(
        cone_response=torch.linspace(-1, 1, 20).view(1, 20, 1),
        spike_counts=counts,
        valid_mask=torch.ones_like(counts, dtype=torch.bool),
        source_ids=(source,),
        context_ids=("stationary",),
    )


def _context_split() -> ResponseSplit:
    sequences = []
    sources = []
    contexts = []
    probe = torch.tensor([[0.5], [1.0]])
    for source in ("a", "b", "c"):
        for context, level in (("low", -0.5), ("high", 0.5)):
            sequences.append(torch.cat((torch.full((6, 1), level), probe)))
            sources.append(source)
            contexts.append(context)
    cones = torch.stack(sequences)
    counts = torch.zeros(6, 1, 8, 1)
    return ResponseSplit(
        cone_response=cones,
        spike_counts=counts,
        valid_mask=torch.ones_like(counts, dtype=torch.bool),
        source_ids=tuple(sources),
        context_ids=tuple(contexts),
    )


def _teacher_result(
    status: str,
    primary_errors: tuple[float, ...],
    recovery_errors: tuple[float, ...],
    pair_count: int = 3,
    direction_agreement: tuple[bool, ...] = (True,),
    model_signed_gains: tuple[float, ...] = (0.2,),
    reference_signed_gains: tuple[float, ...] = (0.3,),
) -> rf_dynamic.DynamicRFResult:
    return rf_dynamic.DynamicRFResult(
        pair_count,
        0.2,
        0.2,
        (0.2, 0.2),
        (0.2, 0.2),
        0.0,
        (0.1,),
        0.0,
        0.0,
        0.0,
        (0.2,) * pair_count,
        (0.2,) * pair_count,
        status,
        primary_errors,
        recovery_errors,
        direction_agreement,
        model_signed_gains,
        reference_signed_gains,
    )
