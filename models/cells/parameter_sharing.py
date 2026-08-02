from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, assert_never

import torch

from configs.rgc_type_priors import ParameterPrior, RGCTypePrior, RGCTypePriors
from data.rgc_response import CellMetadata


class ParameterSharingError(ValueError):
    pass


ParameterSharingMode: TypeAlias = Literal[
    "type_aware",
    "type_blind",
    "cell_only",
    "shuffled_type",
    "balanced_shuffled_type",
]

RGC_PARAMETER_NAMES: Final = (
    "spatial_sigma",
    "sustained_mix",
    "membrane_tau_ms",
    "adaptation_tau_ms",
    "adaptation_gain",
    "amacrine_gain",
    "threshold",
    "subunit_tau_ms",
    "subunit_gain",
)
_PARAMETER_SHARING_MODES: Final = frozenset(
    (
        "type_aware",
        "type_blind",
        "cell_only",
        "shuffled_type",
        "balanced_shuffled_type",
    )
)


@dataclass(frozen=True, slots=True)
class ParameterSharingGroups:
    mode: ParameterSharingMode
    priors: tuple[RGCTypePrior, ...]
    group_indices: torch.Tensor
    effective_type_labels: tuple[str, ...]
    parameter_group_labels: tuple[str, ...]
    use_cell_residuals: bool
    shuffle_contract: str


def parameter_sharing_groups(
    cells: CellMetadata,
    priors: RGCTypePriors,
    mode: str,
    seed: int,
) -> ParameterSharingGroups:
    if not isinstance(mode, str) or mode not in _PARAMETER_SHARING_MODES:
        raise ParameterSharingError(
            "parameter_sharing_mode must be one of "
            f"{sorted(_PARAMETER_SHARING_MODES)}"
        )
    match mode:
        case "type_aware":
            return _type_aware_groups(cells, priors, cells.type_ids, mode)
        case "type_blind":
            return ParameterSharingGroups(
                mode=mode,
                priors=(_pooled_type_prior(priors),),
                group_indices=torch.zeros(len(cells.ids), dtype=torch.long),
                effective_type_labels=("pooled",) * len(cells.ids),
                parameter_group_labels=("pooled",),
                use_cell_residuals=True,
                shuffle_contract="none",
            )
        case "cell_only":
            return _cell_only_groups(cells, priors, mode)
        case "shuffled_type":
            return _type_aware_groups(
                cells,
                priors,
                _shuffled_type_labels(cells.type_ids, seed),
                mode,
                shuffle_contract="global_unconstrained",
            )
        case "balanced_shuffled_type":
            return _type_aware_groups(
                cells,
                priors,
                _balanced_shuffled_type_labels(
                    cells.type_ids,
                    tuple(int(value) for value in cells.polarities),
                    seed,
                ),
                mode,
                shuffle_contract="within_polarity_balanced",
            )
        case unreachable:
            assert_never(unreachable)


def _cell_only_groups(
    cells: CellMetadata,
    priors: RGCTypePriors,
    mode: ParameterSharingMode,
) -> ParameterSharingGroups:
    type_lookup = {prior.type_id: prior for prior in priors.types}
    try:
        grouped_priors = tuple(type_lookup[type_id] for type_id in cells.type_ids)
    except KeyError as exc:
        raise ParameterSharingError(f"Missing RGC type prior: {exc.args[0]}") from exc
    return ParameterSharingGroups(
        mode=mode,
        priors=grouped_priors,
        group_indices=torch.arange(len(cells.ids), dtype=torch.long),
        effective_type_labels=cells.ids,
        parameter_group_labels=cells.ids,
        use_cell_residuals=False,
        shuffle_contract="none",
    )


def _type_aware_groups(
    cells: CellMetadata,
    priors: RGCTypePriors,
    labels: tuple[str, ...],
    mode: ParameterSharingMode,
    *,
    shuffle_contract: str = "none",
) -> ParameterSharingGroups:
    type_lookup = {type_id: index for index, type_id in enumerate(priors.type_ids)}
    try:
        group_indices = torch.tensor(
            [type_lookup[type_id] for type_id in labels],
            dtype=torch.long,
        )
    except KeyError as exc:
        raise ParameterSharingError(f"Missing RGC type prior: {exc.args[0]}") from exc
    return ParameterSharingGroups(
        mode=mode,
        priors=tuple(priors.types),
        group_indices=group_indices,
        effective_type_labels=labels,
        parameter_group_labels=priors.type_ids,
        use_cell_residuals=True,
        shuffle_contract=shuffle_contract,
    )


def _pooled_type_prior(priors: RGCTypePriors) -> RGCTypePrior:
    parameters = {
        name: _pooled_parameter(tuple(prior.parameter(name) for prior in priors.types))
        for name in RGC_PARAMETER_NAMES
    }
    return RGCTypePrior(type_id="pooled", **parameters)


def _pooled_parameter(values: tuple[ParameterPrior, ...]) -> ParameterPrior:
    count = len(values)
    return ParameterPrior(
        mean=sum(prior.mean for prior in values) / count,
        lower=sum(prior.lower for prior in values) / count,
        upper=sum(prior.upper for prior in values) / count,
    )


def _shuffled_type_labels(type_ids: tuple[str, ...], seed: int) -> tuple[str, ...]:
    if len(set(type_ids)) < 2:
        raise ParameterSharingError("shuffled_type requires at least two observed types")
    labels = list(type_ids)
    random.Random(seed).shuffle(labels)
    if tuple(labels) == type_ids:
        labels = labels[1:] + labels[:1]
    return tuple(labels)


def _balanced_shuffled_type_labels(
    type_ids: tuple[str, ...],
    polarities: tuple[int, ...],
    seed: int,
) -> tuple[str, ...]:
    if len(type_ids) != len(polarities):
        raise ParameterSharingError("type labels and polarities must align")
    labels = list(type_ids)
    rng = random.Random(seed)
    for polarity in sorted(set(polarities)):
        indices = [index for index, value in enumerate(polarities) if value == polarity]
        grouped = [type_ids[index] for index in indices]
        if len(set(grouped)) < 2:
            continue
        rng.shuffle(grouped)
        if all(grouped[offset] == type_ids[index] for offset, index in enumerate(indices)):
            grouped = grouped[1:] + grouped[:1]
        for index, label in zip(indices, grouped, strict=True):
            labels[index] = label
    if tuple(labels) == type_ids:
        raise ParameterSharingError(
            "balanced_shuffled_type requires two observed types within a polarity"
        )
    return tuple(labels)


__all__ = [
    "ParameterSharingError",
    "ParameterSharingGroups",
    "ParameterSharingMode",
    "RGC_PARAMETER_NAMES",
    "parameter_sharing_groups",
]
