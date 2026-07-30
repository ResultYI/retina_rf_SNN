from __future__ import annotations

from dataclasses import dataclass

import torch

from evaluation.rf_dynamic import evaluate_dynamic_rf
from training import response_data
from training.response_data import ResponseSplit


def test_conditional_recovery_uses_observed_history() -> None:
    # Given
    model = _HistoryCausalModel()
    baseline = _split_with_recovery_history(0.0)
    changed = _split_with_recovery_history(3.0)

    # When
    base = evaluate_dynamic_rf(
        model,
        baseline,
        lag_steps=2,
        condition_on_observed=True,
        recovery_delays_ms=(0,),
        bootstrap_iterations=25,
    )
    conditional = evaluate_dynamic_rf(
        model,
        changed,
        lag_steps=2,
        condition_on_observed=True,
        recovery_delays_ms=(0,),
        bootstrap_iterations=25,
    )
    free = evaluate_dynamic_rf(
        model,
        changed,
        lag_steps=2,
        condition_on_observed=False,
        recovery_delays_ms=(0,),
        bootstrap_iterations=25,
    )
    free_again = evaluate_dynamic_rf(
        model,
        baseline,
        lag_steps=2,
        condition_on_observed=False,
        recovery_delays_ms=(0,),
        bootstrap_iterations=25,
    )

    # Then
    assert conditional.recovery_shape_distances != base.recovery_shape_distances
    assert conditional.reset_shape_distance != base.reset_shape_distance
    assert (
        conditional.recovery_mean_log_gain_shifts
        != base.recovery_mean_log_gain_shifts
    )
    assert conditional.reset_log_gain_shift != base.reset_log_gain_shift
    assert free.recovery_shape_distances == free_again.recovery_shape_distances
    assert free.reset_shape_distance == free_again.reset_shape_distance


def test_conditional_recovery_zeroes_invalid_history_bins() -> None:
    # Given
    model = _HistoryCausalModel()
    valid = _split_with_recovery_history(0.0, invalid_value=0.0)
    invalid_changed = _split_with_recovery_history(0.0, invalid_value=100.0)

    # When
    reference = evaluate_dynamic_rf(
        model,
        valid,
        lag_steps=2,
        condition_on_observed=True,
        recovery_delays_ms=(0,),
        bootstrap_iterations=25,
    )
    changed = evaluate_dynamic_rf(
        model,
        invalid_changed,
        lag_steps=2,
        condition_on_observed=True,
        recovery_delays_ms=(0,),
        bootstrap_iterations=25,
    )

    # Then
    assert reference.recovery_shape_distances == changed.recovery_shape_distances
    assert reference.reset_shape_distance == changed.reset_shape_distance


def test_invalid_spike_bins_are_zeroed_before_state_updates() -> None:
    counts = torch.tensor([[[1.0], [9.0], [1.0]]])
    mask = torch.tensor([[[True], [False], [True]]])

    history = response_data.masked_history_counts(counts, mask)

    assert history.tolist() == [[[1.0], [0.0], [1.0]]]


@dataclass(frozen=True, slots=True)
class _StaticOutput:
    spike_logits: torch.Tensor


class _StaticRGC(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("support_mask", torch.ones(1, 2, dtype=torch.bool))


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
            first = sample[:, 0:1] * (1.0 + state)
            second = sample[:, 1:2] * (1.0 - state)
            logits.append(self.scale * (first + second))
            if observed_counts is not None:
                state = state + observed_counts[:, index]
        return _StaticOutput(torch.stack(logits, dim=1)), state


def _split_with_recovery_history(
    low_history: float,
    *,
    invalid_value: float = 0.0,
) -> ResponseSplit:
    probe = torch.tensor([[1.0, 0.5], [2.0, 1.0]])
    sequences = []
    sources = []
    contexts = []
    counts = torch.zeros(6, 1, 5, 1)
    valid_mask = torch.ones_like(counts, dtype=torch.bool)
    for source_index, source in enumerate(("a", "b", "c")):
        low_index = source_index * 2
        high_index = low_index + 1
        sequences.append(torch.cat((torch.ones(3, 2), probe)))
        sequences.append(torch.cat((torch.ones(3, 2) * 2.0, probe)))
        counts[low_index, 0, 3, 0] = low_history
        counts[high_index, 0, 3, 0] = 4.0
        counts[low_index : high_index + 1, 0, 1, 0] = invalid_value
        valid_mask[low_index : high_index + 1, 0, 1, 0] = False
        sources.extend((source, source))
        contexts.extend(("low", "high"))
    return ResponseSplit(
        cone_response=torch.stack(sequences),
        spike_counts=counts,
        valid_mask=valid_mask,
        source_ids=tuple(sources),
        context_ids=tuple(contexts),
    )
