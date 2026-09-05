from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch


@dataclass(frozen=True, slots=True)
class NestedSpikeBank:
    seed: int
    train_spikes: torch.Tensor
    validation_spikes: torch.Tensor
    train_sha256: str
    validation_sha256: str


@dataclass(frozen=True, slots=True)
class SpikeBudget:
    seed: int
    trials: int
    train_spikes: torch.Tensor
    validation_spikes: torch.Tensor
    train_sha256: str
    validation_sha256: str


@dataclass(frozen=True, slots=True)
class SpikeBankError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def generate_nested_spike_bank(
    train_probability: torch.Tensor,
    validation_probability: torch.Tensor,
    *,
    seed: int,
    max_trials: int = 64,
) -> NestedSpikeBank:
    _validate_probability(train_probability, max_trials)
    _validate_probability(validation_probability, max_trials)
    train_generator = torch.Generator().manual_seed(seed)
    validation_generator = torch.Generator().manual_seed(seed + 1_000_003)
    train = _sample(train_probability, max_trials, train_generator)
    validation = _sample(validation_probability, max_trials, validation_generator)
    return NestedSpikeBank(
        seed,
        train,
        validation,
        tensor_sha256(train),
        tensor_sha256(validation),
    )


def slice_spike_bank(bank: NestedSpikeBank, trials: int) -> SpikeBudget:
    if trials < 1 or trials > bank.train_spikes.shape[1]:
        raise SpikeBankError("trial budget is outside the generated master bank")
    train = bank.train_spikes[:, :trials].to(dtype=torch.float32)
    validation = bank.validation_spikes[:, :trials].to(dtype=torch.float32)
    return SpikeBudget(
        bank.seed,
        trials,
        train,
        validation,
        tensor_sha256(train),
        tensor_sha256(validation),
    )


def tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def spike_counts_by_cell(value: torch.Tensor) -> tuple[int, ...]:
    if value.ndim != 4:
        raise SpikeBankError("spike counts require [stimulus,trial,time,cell]")
    return tuple(int(item) for item in value.sum(dim=(0, 1, 2)))


def firing_rate_summary(value: torch.Tensor) -> tuple[float, float, float, float]:
    rates = value.float().mean(dim=(0, 1, 2))
    return float(rates.min()), float(rates.median()), float(rates.mean()), float(rates.max())


def _sample(
    probability: torch.Tensor,
    trials: int,
    generator: torch.Generator,
) -> torch.Tensor:
    expanded = probability[:, None].expand(-1, trials, -1, -1)
    uniforms = torch.rand(expanded.shape, generator=generator, dtype=probability.dtype)
    return uniforms < expanded


def _validate_probability(probability: torch.Tensor, trials: int) -> None:
    if probability.ndim != 3 or trials < 1:
        raise SpikeBankError("probabilities must be [stimulus,time,cell]")
    if not bool(torch.isfinite(probability).all()):
        raise SpikeBankError("probabilities must be finite")
    if bool(((probability < 0) | (probability > 1)).any()):
        raise SpikeBankError("probabilities must be in [0,1]")


__all__ = [
    "NestedSpikeBank",
    "SpikeBankError",
    "SpikeBudget",
    "firing_rate_summary",
    "generate_nested_spike_bank",
    "slice_spike_bank",
    "spike_counts_by_cell",
    "tensor_sha256",
]
