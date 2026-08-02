from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from evaluation.rf_dynamic import DynamicRFError, evaluate_dynamic_rf
from evaluation.rf_dynamic_metrics import trial_conditioned_rf
from evaluation.rf_history_contracts import (
    RF_HISTORY_CONTRACTS,
    standard_train_rate_history_counts,
)
from evaluation.rf_static import StaticRFError
from training.response_data import ResponseSplit


def test_conditional_rf_history_contracts_control_heldout_history() -> None:
    model = _HistoryCausalModel()
    split = _conditional_split((1.0, 1.0))
    mutated = _conditional_split((5.0, 1.0))
    standard = standard_train_rate_history_counts(
        _training_rate_split(1.0),
        burn_in_steps=1,
        sequence_steps=4,
    )

    zero = trial_conditioned_rf(
        model,
        split,
        1,
        2,
        history_mode="zero",
        matched_history_index=0,
        standard_history_counts=standard,
    )
    zero_mutated = trial_conditioned_rf(
        model,
        mutated,
        1,
        2,
        history_mode="zero",
        matched_history_index=0,
        standard_history_counts=standard,
    )
    standard_rf = trial_conditioned_rf(
        model,
        split,
        1,
        2,
        history_mode="standard_train_rate",
        matched_history_index=0,
        standard_history_counts=standard,
    )
    standard_mutated = trial_conditioned_rf(
        model,
        mutated,
        1,
        2,
        history_mode="standard_train_rate",
        matched_history_index=0,
        standard_history_counts=standard,
    )
    matched = trial_conditioned_rf(
        model,
        split,
        1,
        2,
        history_mode="matched_observed",
        matched_history_index=0,
        standard_history_counts=standard,
    )
    matched_mutated = trial_conditioned_rf(
        model,
        mutated,
        1,
        2,
        history_mode="matched_observed",
        matched_history_index=0,
        standard_history_counts=standard,
    )

    assert torch.equal(zero.kernels, zero_mutated.kernels)
    assert torch.equal(standard_rf.kernels, standard_mutated.kernels)
    assert not torch.equal(matched.kernels, matched_mutated.kernels)


def test_matched_history_reuses_byte_identical_low_history_for_pair() -> None:
    model = _RecordingHistoryModel()
    split = _conditional_split((2.0, 3.0))

    trial_conditioned_rf(
        model,
        split,
        0,
        2,
        history_mode="matched_observed",
        matched_history_index=0,
    )
    trial_conditioned_rf(
        model,
        split,
        1,
        2,
        history_mode="matched_observed",
        matched_history_index=0,
    )

    assert len(model.histories) == 6
    for left, right in zip(model.histories[:3], model.histories[3:], strict=True):
        assert torch.equal(left, right)


def test_standard_train_rate_history_uses_train_post_burn_only() -> None:
    train = _training_rate_split(0.5)
    heldout = _training_rate_split(1.0)

    standard = standard_train_rate_history_counts(
        train,
        burn_in_steps=1,
        sequence_steps=6,
    )
    heldout_standard = standard_train_rate_history_counts(
        train,
        burn_in_steps=1,
        sequence_steps=6,
    )
    train_changed = standard_train_rate_history_counts(
        heldout,
        burn_in_steps=1,
        sequence_steps=6,
    )

    assert set(standard.flatten().tolist()) <= {0.0, 1.0}
    assert torch.equal(standard, heldout_standard) and not torch.equal(standard, train_changed)


def test_dynamic_rf_rejects_missing_context_pair_for_history_contract() -> None:
    split = _conditional_split((1.0, 1.0))
    malformed = ResponseSplit(
        cone_response=split.cone_response[:-1],
        spike_counts=split.spike_counts[:-1],
        valid_mask=split.valid_mask[:-1],
        source_ids=split.source_ids[:-1],
        context_ids=split.context_ids[:-1],
    )

    with pytest.raises(DynamicRFError, match="complete low/high"):
        evaluate_dynamic_rf(
            _HistoryCausalModel(),
            malformed,
            lag_steps=2,
            history_mode="matched_observed",
            bootstrap_iterations=25,
        )


@pytest.mark.parametrize("history", RF_HISTORY_CONTRACTS)
def test_history_contracts_reject_missing_trial_with_domain_error(
    history: str,
) -> None:
    split = _conditional_split((1.0, 1.0))
    malformed = ResponseSplit(
        cone_response=split.cone_response,
        spike_counts=split.spike_counts[:, :0],
        valid_mask=split.valid_mask[:, :0],
        source_ids=split.source_ids,
        context_ids=split.context_ids,
    )

    with pytest.raises(DynamicRFError):
        trial_conditioned_rf(
            _HistoryCausalModel(),
            malformed,
            1,
            2,
            history_mode=history,
            matched_history_index=0,
            standard_history_counts=torch.zeros(1, 4, 1),
        )


@pytest.mark.parametrize("history", RF_HISTORY_CONTRACTS)
def test_history_contracts_reject_insufficient_history_with_domain_error(
    history: str,
) -> None:
    split = _short_history_split()

    with pytest.raises((DynamicRFError, StaticRFError)):
        trial_conditioned_rf(
            _HistoryCausalModel(),
            split,
            1,
            2,
            history_mode=history,
            matched_history_index=0,
            standard_history_counts=torch.zeros(1, 1, 1),
        )


@dataclass(frozen=True, slots=True)
class _StaticOutput:
    spike_logits: torch.Tensor


class _StaticRGC(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("support_mask", torch.ones(1, 1, dtype=torch.bool))


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


class _RecordingHistoryModel(_HistoryCausalModel):
    def __init__(self) -> None:
        super().__init__()
        self.histories: list[torch.Tensor] = []

    def forward_sequence(
        self,
        sequence: torch.Tensor,
        *,
        observed_counts: torch.Tensor | None = None,
    ) -> tuple[_StaticOutput, torch.Tensor]:
        if observed_counts is not None:
            self.histories.append(observed_counts.detach().cpu().clone())
        return super().forward_sequence(sequence, observed_counts=observed_counts)


def _conditional_split(low_trial_values: tuple[float, float]) -> ResponseSplit:
    cones = torch.ones(6, 4, 1)
    counts = torch.zeros(6, 2, 4, 1)
    sources = []
    contexts = []
    for source_index, source in enumerate(("a", "b", "c")):
        low_index = source_index * 2
        high_index = low_index + 1
        counts[low_index, :, 2, 0] = torch.tensor(low_trial_values)
        counts[high_index, :, 2, 0] = 4.0
        sources.extend((source, source))
        contexts.extend(("low", "high"))
    return ResponseSplit(
        cone_response=cones,
        spike_counts=counts,
        valid_mask=torch.ones_like(counts, dtype=torch.bool),
        source_ids=tuple(sources),
        context_ids=tuple(contexts),
    )


def _short_history_split() -> ResponseSplit:
    base = _conditional_split((1.0, 1.0))
    return ResponseSplit(
        cone_response=base.cone_response[:, :1],
        spike_counts=base.spike_counts[:, :, :1],
        valid_mask=base.valid_mask[:, :, :1],
        source_ids=base.source_ids,
        context_ids=base.context_ids,
    )


def _training_rate_split(rate: float) -> ResponseSplit:
    counts = torch.zeros(1, 2, 5, 1)
    counts[:, :, 1:, 0] = rate
    return ResponseSplit(
        cone_response=torch.ones(1, 5, 1),
        spike_counts=counts,
        valid_mask=torch.ones_like(counts, dtype=torch.bool),
        source_ids=("train",),
        context_ids=("low",),
    )
