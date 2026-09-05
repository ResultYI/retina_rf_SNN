from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Final

import torch

from evaluation.mechanistic_retina.spike_banks import SpikeBudget, tensor_sha256
from evaluation.model_comparison.config import ComparisonConfig, load_comparison_config
from training.mechanistic_retina.stages import MechanisticSeedData


EXPECTED_SELECTION_SEED: Final = 19
TOTAL_TRAIN_STIMULI: Final = 112
EXPECTED_FRACTIONS: Final = (0.25, 0.5, 1.0)
EXPECTED_TRAIN_COUNTS: Final = (28, 56, 112)


@dataclass(frozen=True, slots=True)
class SampleEfficiencyProtocolError(ValueError):
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


@dataclass(frozen=True, slots=True)
class SampleEfficiencySubset:
    fraction: float
    train_count: int
    indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SampleEfficiencyBankSlice:
    seed: int
    trials: int
    train_spikes: torch.Tensor
    validation_spikes: torch.Tensor
    train_sha256: str
    validation_sha256: str


@dataclass(frozen=True, slots=True)
class SampleEfficiencyDataSlice:
    fraction: float
    train_count: int
    indices: tuple[int, ...]
    train_cones: torch.Tensor
    train_probability: torch.Tensor
    train_mask: torch.Tensor
    validation_cones: torch.Tensor
    validation_probability: torch.Tensor
    validation_mask: torch.Tensor
    train_cone_sha256: str
    train_probability_sha256: str
    train_mask_sha256: str
    validation_cone_sha256: str
    validation_probability_sha256: str
    validation_mask_sha256: str
    banks: tuple[SampleEfficiencyBankSlice, ...]


@dataclass(frozen=True, slots=True)
class SampleEfficiencyProtocol:
    canonical_config_path: Path
    canonical_config: ComparisonConfig
    output_dir: Path
    run_dir: Path
    selection_seed: int
    subsets: tuple[SampleEfficiencySubset, ...]

    @property
    def fractions(self) -> tuple[float, ...]:
        return tuple(subset.fraction for subset in self.subsets)


def load_sample_efficiency_protocol(path: Path) -> SampleEfficiencyProtocol:
    if not path.exists():
        raise SampleEfficiencyProtocolError("CONFIG_MISSING", str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SampleEfficiencyProtocolError("CONFIG_JSON_INVALID", str(path)) from exc
    if not isinstance(payload, dict):
        raise SampleEfficiencyProtocolError("CONFIG_NOT_OBJECT", str(path))

    canonical_config_path = _required_path(payload, "canonical_config_path")
    if not canonical_config_path.exists():
        raise SampleEfficiencyProtocolError(
            "CANONICAL_CONFIG_MISSING",
            str(canonical_config_path),
        )
    canonical_config = load_comparison_config(canonical_config_path)
    selection_seed = _required_int(payload, "selection_seed")
    if selection_seed != EXPECTED_SELECTION_SEED:
        raise SampleEfficiencyProtocolError(
            "SELECTION_SEED_NOT_19",
            str(selection_seed),
        )
    fractions = _required_fractions(payload)
    return SampleEfficiencyProtocol(
        canonical_config_path,
        canonical_config,
        _required_path(payload, "output_dir"),
        _required_path(payload, "run_dir"),
        selection_seed,
        _build_subsets(fractions, selection_seed),
    )


def build_sample_efficiency_slices(
    protocol: SampleEfficiencyProtocol,
    data: MechanisticSeedData,
    banks: Sequence[SpikeBudget],
) -> tuple[SampleEfficiencyDataSlice, ...]:
    _assert_train_axis(data.train_cones, "TRAIN_CONES_COUNT")
    _assert_train_axis(data.train_probability, "TRAIN_PROBABILITY_COUNT")
    _assert_train_axis(data.train_mask, "TRAIN_MASK_COUNT")
    bank_seeds = tuple(bank.seed for bank in banks)
    if bank_seeds != protocol.canonical_config.bank_seeds:
        raise SampleEfficiencyProtocolError(
            "BANK_SEEDS_MISMATCH",
            str(bank_seeds),
        )
    return tuple(_build_slice(subset, data, banks) for subset in protocol.subsets)


def _build_slice(
    subset: SampleEfficiencySubset,
    data: MechanisticSeedData,
    banks: Sequence[SpikeBudget],
) -> SampleEfficiencyDataSlice:
    index = torch.tensor(subset.indices, dtype=torch.long)
    train_cones = data.train_cones.index_select(0, index)
    train_probability = data.train_probability.index_select(0, index)
    train_mask = data.train_mask.index_select(0, index)
    return SampleEfficiencyDataSlice(
        subset.fraction,
        subset.train_count,
        subset.indices,
        train_cones,
        train_probability,
        train_mask,
        data.validation_cones,
        data.validation_probability,
        data.validation_mask,
        tensor_sha256(train_cones),
        tensor_sha256(train_probability),
        tensor_sha256(train_mask),
        tensor_sha256(data.validation_cones),
        tensor_sha256(data.validation_probability),
        tensor_sha256(data.validation_mask),
        tuple(_build_bank_slice(bank, index) for bank in banks),
    )


def _build_bank_slice(bank: SpikeBudget, index: torch.Tensor) -> SampleEfficiencyBankSlice:
    train_spikes = bank.train_spikes.index_select(0, index)
    return SampleEfficiencyBankSlice(
        bank.seed,
        bank.trials,
        train_spikes,
        bank.validation_spikes,
        tensor_sha256(train_spikes),
        bank.validation_sha256,
    )


def _build_subsets(
    fractions: tuple[float, ...],
    selection_seed: int,
) -> tuple[SampleEfficiencySubset, ...]:
    generator = torch.Generator().manual_seed(selection_seed)
    permutation = torch.randperm(TOTAL_TRAIN_STIMULI, generator=generator).tolist()
    subsets: list[SampleEfficiencySubset] = []
    for fraction in fractions:
        count = _fraction_to_count(fraction)
        selected = tuple(sorted(int(index) for index in permutation[:count]))
        subsets.append(SampleEfficiencySubset(fraction, count, selected))
    return tuple(subsets)


def _required_path(payload, key: str) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise SampleEfficiencyProtocolError("CONFIG_FIELD_INVALID", key)
    return Path(value)


def _required_int(payload, key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise SampleEfficiencyProtocolError("CONFIG_FIELD_INVALID", key)
    return value


def _required_fractions(payload) -> tuple[float, ...]:
    value = payload.get("fractions")
    if not isinstance(value, list):
        raise SampleEfficiencyProtocolError("FRACTIONS_NOT_LIST", "fractions")
    fractions = tuple(_parse_fraction(item) for item in value)
    if len(fractions) != len(set(fractions)):
        raise SampleEfficiencyProtocolError("FRACTIONS_NOT_UNIQUE", str(fractions))
    if fractions != tuple(sorted(fractions)):
        raise SampleEfficiencyProtocolError("FRACTIONS_NOT_ASCENDING", str(fractions))
    if fractions != EXPECTED_FRACTIONS:
        raise SampleEfficiencyProtocolError("FRACTIONS_NOT_FROZEN", str(fractions))
    if fractions[-1:] != (1.0,):
        raise SampleEfficiencyProtocolError("FRACTIONS_MUST_END_AT_100", str(fractions))
    previous_count = 0
    counts: list[int] = []
    for fraction in fractions:
        count = _fraction_to_count(fraction)
        if count <= previous_count:
            raise SampleEfficiencyProtocolError("FRACTIONS_NOT_NESTED", str(fractions))
        counts.append(count)
        previous_count = count
    if tuple(counts) != EXPECTED_TRAIN_COUNTS:
        raise SampleEfficiencyProtocolError("COUNTS_NOT_FROZEN", str(tuple(counts)))
    return fractions


def _parse_fraction(value) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SampleEfficiencyProtocolError("FRACTION_INVALID", str(value))
    fraction = float(value)
    if fraction <= 0.0 or fraction > 1.0:
        raise SampleEfficiencyProtocolError("FRACTION_OUT_OF_RANGE", str(fraction))
    return fraction


def _fraction_to_count(fraction: float) -> int:
    exact_count = fraction * TOTAL_TRAIN_STIMULI
    count = int(round(exact_count))
    if not math.isclose(exact_count, count, rel_tol=0.0, abs_tol=1e-9):
        raise SampleEfficiencyProtocolError("FRACTION_NOT_EXACT_COUNT", str(fraction))
    return count


def _assert_train_axis(value: torch.Tensor, code: str) -> None:
    if value.shape[0] != TOTAL_TRAIN_STIMULI:
        raise SampleEfficiencyProtocolError(code, str(value.shape[0]))


__all__ = [
    "SampleEfficiencyBankSlice",
    "SampleEfficiencyDataSlice",
    "SampleEfficiencyProtocol",
    "SampleEfficiencyProtocolError",
    "SampleEfficiencySubset",
    "build_sample_efficiency_slices",
    "load_sample_efficiency_protocol",
]
