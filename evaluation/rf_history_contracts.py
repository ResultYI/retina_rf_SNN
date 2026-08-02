from __future__ import annotations

from typing import Final, Literal, Mapping, TypeAlias, TypeVar

import torch

from evaluation.response_metrics import training_baseline_rates
from training.response_data import ResponseSplit

RFHistoryContract: TypeAlias = Literal[
    "zero",
    "matched_observed",
    "standard_train_rate",
]
RF_HISTORY_CONTRACTS: Final[tuple[RFHistoryContract, ...]] = (
    "zero",
    "matched_observed",
    "standard_train_rate",
)

T = TypeVar("T")


class RFHistoryContractError(ValueError):
    pass


def require_exact_history_contracts(
    by_history: Mapping[RFHistoryContract, T] | None,
) -> Mapping[RFHistoryContract, T]:
    if by_history is None:
        raise RFHistoryContractError(
            "RF artifact v2 requires zero, matched_observed, standard_train_rate"
        )
    expected = set(RF_HISTORY_CONTRACTS)
    actual = set(by_history)
    if actual != expected:
        raise RFHistoryContractError(
            "RF artifact v2 requires exact history keys "
            "zero, matched_observed, standard_train_rate"
        )
    return by_history


def standard_train_rate_history_counts(
    train_split: ResponseSplit,
    *,
    burn_in_steps: int,
    sequence_steps: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    rates = training_baseline_rates(
        train_split.spike_counts[:, :, burn_in_steps:].flatten(0, 1),
        train_split.valid_mask[:, :, burn_in_steps:].flatten(0, 1),
    ).clamp(0.0, 1.0)
    target_device = train_split.spike_counts.device if device is None else device
    spike_count = torch.round(rates.cpu() * sequence_steps).to(torch.int64)
    times = torch.arange(sequence_steps).view(sequence_steps, 1)
    schedule = (times < spike_count.view(1, -1)).to(dtype)
    return schedule.unsqueeze(0).to(target_device)


__all__ = [
    "RFHistoryContract",
    "RFHistoryContractError",
    "RF_HISTORY_CONTRACTS",
    "require_exact_history_contracts",
    "standard_train_rate_history_counts",
]
