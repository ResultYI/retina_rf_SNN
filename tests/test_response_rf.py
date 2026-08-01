from __future__ import annotations

from dataclasses import dataclass

import torch

from baselines.point_process_glm import PointProcessGLM
from evaluation.rf_dynamic import classify_dynamic_rf, evaluate_dynamic_rf
from evaluation.rf_static import StaticRFError, extract_static_rf
from training.response_data import ResponseSplit


def test_static_glm_uses_only_past_spike_history() -> None:
    model = PointProcessGLM(cone_count=2, cell_count=1, temporal_lags=2)
    cones = torch.zeros(1, 4, 2)
    baseline = torch.zeros(1, 4, 1)
    changed = baseline.clone()
    changed[:, 1] = 1

    before = model(cones, baseline)
    after = model(cones, changed)

    assert torch.equal(before[:, :2], after[:, :2])


def test_dynamic_rf_needs_three_independent_context_pairs() -> None:
    assert classify_dynamic_rf(1, 0.2, 0.2) == "not_identifiable"
    assert classify_dynamic_rf(3, 0.2, 0.2) == "supported"


def test_extract_static_rf_free_running_is_deterministic_when_no_history() -> None:
    # Given
    model = _StaticLinearModel()
    sequence = torch.tensor([[[0.25], [0.5], [0.75]]])

    # When
    first = extract_static_rf(model, sequence, lag_steps=2)
    second = extract_static_rf(model, sequence, lag_steps=2)

    # Then
    assert torch.equal(first.kernels, second.kernels)
    assert first.finite_difference_relative_error == second.finite_difference_relative_error
    assert first.identifiable is second.identifiable


def test_extract_static_rf_checks_strongest_supported_coordinate() -> None:
    model = _MixedGradientModel()
    sequence = torch.ones(1, 3, 2)

    result = extract_static_rf(model, sequence, lag_steps=2)

    assert result.identifiable
    assert result.finite_difference_relative_error <= 0.05


def test_extract_static_rf_uses_observed_history_causally() -> None:
    # Given
    model = _HistoryCausalModel()
    sequence = torch.ones(1, 4, 1)
    baseline = torch.zeros(1, 4, 1)
    changed_current = baseline.clone()
    changed_current[:, -1] = 9.0
    changed_past = baseline.clone()
    changed_past[:, -2] = 3.0

    # When
    base_rf = extract_static_rf(
        model,
        sequence,
        lag_steps=2,
        observed_counts=baseline,
    )
    current_rf = extract_static_rf(
        model,
        sequence,
        lag_steps=2,
        observed_counts=changed_current,
    )
    past_rf = extract_static_rf(
        model,
        sequence,
        lag_steps=2,
        observed_counts=changed_past,
    )

    # Then
    assert torch.equal(base_rf.kernels, current_rf.kernels)
    assert not torch.equal(base_rf.kernels, past_rf.kernels)
    assert past_rf.finite_difference_relative_error <= 0.05


def test_extract_static_rf_rejects_malformed_observed_history_shape() -> None:
    # Given
    model = _HistoryCausalModel()
    sequence = torch.ones(1, 4, 1)
    malformed = torch.zeros(1, 3, 1)

    # When / Then
    try:
        extract_static_rf(
            model,
            sequence,
            lag_steps=2,
            observed_counts=malformed,
        )
    except StaticRFError as error:
        assert "observed counts" in str(error)
    else:
        raise AssertionError("malformed observed history was accepted")


def test_dynamic_rf_conditional_trial_average_is_trial_order_invariant() -> None:
    # Given
    model = _HistoryCausalModel()
    split = _conditional_split((0.0, 2.0))
    reversed_split = _conditional_split((2.0, 0.0))

    # When
    result = evaluate_dynamic_rf(model, split, lag_steps=2, bootstrap_iterations=25)
    reversed_result = evaluate_dynamic_rf(
        model,
        reversed_split,
        lag_steps=2,
        bootstrap_iterations=25,
    )

    # Then
    assert result.mean_log_gain_shift > 0.1
    assert result.per_source_gain_shifts == reversed_result.per_source_gain_shifts
    assert result.finite_difference_relative_error <= 0.05


def test_dynamic_rf_zeroes_invalid_observed_bins_before_aggregation() -> None:
    # Given
    model = _HistoryCausalModel()
    valid_reference = _conditional_split((1.0, 1.0), invalid_value=0.0)
    invalid_changed = _conditional_split((1.0, 1.0), invalid_value=100.0)

    # When
    reference = evaluate_dynamic_rf(
        model,
        valid_reference,
        lag_steps=2,
        bootstrap_iterations=25,
    )
    changed = evaluate_dynamic_rf(
        model,
        invalid_changed,
        lag_steps=2,
        bootstrap_iterations=25,
    )

    # Then
    assert reference.mean_log_gain_shift > 0.1
    assert reference.per_source_gain_shifts == changed.per_source_gain_shifts
    assert reference.per_source_shape_distances == changed.per_source_shape_distances


def test_extract_static_rf_free_running_ignores_observed_history() -> None:
    # Given
    model = _HistoryCausalModel()
    sequence = torch.ones(1, 4, 1)
    history = torch.full((1, 4, 1), 4.0)

    # When
    free = extract_static_rf(model, sequence, lag_steps=2)
    conditional = extract_static_rf(
        model,
        sequence,
        lag_steps=2,
        observed_counts=history,
    )
    free_again = extract_static_rf(model, sequence, lag_steps=2)

    # Then
    assert torch.equal(free.kernels, free_again.kernels)
    assert not torch.equal(free.kernels, conditional.kernels)


class _StaticLinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.rgc = _StaticRGC()

    def forward_sequence(
        self,
        sequence: torch.Tensor,
    ) -> tuple[_StaticOutput, torch.Tensor]:
        logits = self.scale * sequence
        return _StaticOutput(logits), logits[:, -1]


class _MixedGradientModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights = torch.nn.Parameter(torch.tensor([1e-4, 1.0]))
        self.rgc = _StaticRGC(cone_count=2)

    def forward_sequence(
        self,
        sequence: torch.Tensor,
    ) -> tuple[_StaticOutput, torch.Tensor]:
        logits = (sequence * self.weights).sum(dim=2, keepdim=True)
        return _StaticOutput(logits), logits[:, -1]


@dataclass(frozen=True, slots=True)
class _StaticOutput:
    spike_logits: torch.Tensor


class _StaticRGC(torch.nn.Module):
    def __init__(self, cone_count: int = 1) -> None:
        super().__init__()
        self.register_buffer(
            "support_mask",
            torch.ones(1, cone_count, dtype=torch.bool),
        )


class _HistoryCausalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.rgc = _StaticRGC()

    def forward_sequence(
        self,
        sequence: torch.Tensor,
        *,
        observed_counts: torch.Tensor | None = None,
    ) -> tuple[_StaticOutput, torch.Tensor]:
        state = torch.zeros(sequence.shape[0], 1, device=sequence.device)
        logits = []
        for index, sample in enumerate(sequence.unbind(dim=1)):
            logits.append(self.scale * sample * (1.0 + state))
            if observed_counts is not None:
                state = state + observed_counts[:, index]
        return _StaticOutput(torch.stack(logits, dim=1)), state


def _conditional_split(
    low_trial_values: tuple[float, float],
    *,
    invalid_value: float = 0.0,
) -> ResponseSplit:
    cones = torch.ones(6, 4, 1)
    counts = torch.zeros(6, 2, 4, 1)
    valid_mask = torch.ones_like(counts, dtype=torch.bool)
    sources = []
    contexts = []
    for source_index, source in enumerate(("a", "b", "c")):
        low_index = source_index * 2
        high_index = low_index + 1
        counts[low_index, :, 2, 0] = torch.tensor(low_trial_values)
        counts[high_index, :, 2, 0] = 4.0
        counts[low_index : high_index + 1, :, 1, 0] = invalid_value
        valid_mask[low_index : high_index + 1, :, 1, 0] = False
        sources.extend((source, source))
        contexts.extend(("low", "high"))
    return ResponseSplit(
        cone_response=cones,
        spike_counts=counts,
        valid_mask=valid_mask,
        source_ids=tuple(sources),
        context_ids=tuple(contexts),
    )
